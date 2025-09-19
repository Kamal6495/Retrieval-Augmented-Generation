import os
import time
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st

from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


# ---------------- Load ENV ----------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")



if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RAG PDF QA Chat", page_icon="💬", layout="wide")
st.title("💬 RAG Chatbot over PDF")

uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

chunk_size = st.sidebar.slider("Chunk size", 200, 1500, 800, 50)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 500, 150, 10)
top_k = st.sidebar.slider("Top K retrievals", 1, 10, 4, 1)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if uploaded_file:
    # Save uploaded file temporarily
    pdf_path = "temp_uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF uploaded successfully!")

    # Step 1: Load PDF
    loader = PyPDFium2Loader(pdf_path)
    docs = list(loader.lazy_load())
    st.write(f"**Pages loaded:** {len(docs)}")

    # Step 2: Split docs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    st.write(f"**Chunks created:** {len(chunks)}")

    # Step 3: Embeddings (HuggingFace, cached locally)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder="4_RAG_Application/huggingface"
    )

    st.info("🔎 Building FAISS vector index (may take a few seconds)...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

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

    # ---------------- Chat Section ----------------
    st.subheader("💬 Ask a question about your PDF")

    # Input box (always on top)
    query = st.text_input("Enter your question:", key="user_input")

    if query:
        with st.spinner("Generating answer..."):
            start = time.time()
            result = qa_chain(query)
            elapsed = time.time() - start

        answer = result.get("result") or ""
        sources = result.get("source_documents", [])

        timestamp = datetime.now().strftime("%H:%M:%S")
        # Save to chat history (new chat on top)
        st.session_state.chat_history.insert(0, {
            "time": timestamp,
            "question": query,
            "answer": answer,
            "sources": sources,
            "elapsed": elapsed
        })

    # ---------------- Display Chat History ----------------
    st.markdown("### 🗂️ Chat History")
    for chat in st.session_state.chat_history:
        st.markdown(f"**[{chat['time']}] You:** {chat['question']}")
        st.markdown(f"**[{chat['time']}] Bot:** {chat['answer']}")
        st.caption(f"⏱️ {chat['elapsed']:.2f}s | Retrieved {len(chat['sources'])} chunks")
        st.markdown("---")
