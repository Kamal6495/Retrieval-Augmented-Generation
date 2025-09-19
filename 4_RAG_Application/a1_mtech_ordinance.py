import os
import time
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# ---------------- Load ENV ----------------
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="M.Tech Ordinance QA", layout="wide")

st.title("Motilal Nehru National Institute of Technology Allahabad, Prayagraj")
st.subheader("Masters Course - Ordinance")

# ---------------- Load default PDF ----------------
path = os.getcwd()
pdf_file = os.path.join(path, "4_RAG_Application", "ordinance.pdf")

if not os.path.exists(pdf_file):
    st.error(f"PDF not found at {pdf_file}")
    st.stop()

# Step 1: Load PDF
loader = PyPDFium2Loader(pdf_file)
docs = list(loader.lazy_load())
st.write(f"Pages loaded: {len(docs)}")

# Sidebar options
chunk_size = st.sidebar.slider("Chunk size", 200, 1500, 800, 50)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 500, 150, 10)
top_k = st.sidebar.slider("Top K retrievals", 1, 10, 4, 1)

# Step 2: Split docs
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(docs)
st.write(f"Chunks created: {len(chunks)}")

# Step 3: Embeddings (HuggingFace, cached locally)

# Option 1: Fast & lightweight (good enough for most cases)
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     cache_folder="4_RAG_Application/huggingface"
# )

# Option 2: Higher accuracy, bigger model (if you have more RAM/CPU)
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-mpnet-base-v2",
#     cache_folder="4_RAG_Application/huggingface"
# )

# Option 3: QA-tuned embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    cache_folder="4_RAG_Application/huggingface"
)


# Use FAISS index caching
faiss_index_path = "4_RAG_Application/faiss_index"

if os.path.exists(faiss_index_path):
    vectorstore = FAISS.load_local(
        faiss_index_path,
          embeddings,
          allow_dangerous_deserialization=True)#This bypasses safety checks,if you trust the source of the FAISS index.
    st.info("Loaded FAISS index from disk")
else:
    st.info("Building FAISS vector index (may take a few seconds)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(faiss_index_path)
    st.info("Built and saved FAISS index")

# Step 4: QA Chain with Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.0,
    request_timeout=120
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# ---------------- Chat + Cache ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_cache" not in st.session_state:
    st.session_state.qa_cache = {}  # query → answer

st.subheader("Ask a question about the Ordinance")
query = st.text_input("Enter your question:", key="user_input")

if query:
    if query in st.session_state.qa_cache:
        cached = st.session_state.qa_cache[query]
        answer, sources, elapsed = cached["answer"], cached["sources"], 0.0
        from_cache = True
    else:
        with st.spinner("Generating answer..."):
            start = time.time()
            result = qa_chain(query)
            elapsed = time.time() - start

        answer = result.get("result") or ""
        sources = result.get("source_documents", [])
        from_cache = False

        st.session_state.qa_cache[query] = {
            "answer": answer,
            "sources": sources,
            "elapsed": elapsed
        }

    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.insert(0, {
        "time": timestamp,
        "question": query,
        "answer": answer,
        "sources": sources,
        "elapsed": elapsed,
        "from_cache": from_cache
    })

# ---------------- Display Chat History ----------------
st.markdown("### Chat History")
for chat in st.session_state.chat_history:
    source_tag = "(cached)" if chat.get("from_cache") else "(new)"
    st.markdown(f"[{chat['time']}] You: {chat['question']}")
    st.markdown(f"[{chat['time']}] Bot {source_tag}: {chat['answer']}")
    st.caption(f"Answered in {chat['elapsed']:.2f}s | Retrieved {len(chat['sources'])} chunks")
    st.markdown("---")

qa_chain.get_graph().print_ascii()
