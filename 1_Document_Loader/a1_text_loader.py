from dotenv import load_dotenv                        
from langchain_community.document_loaders import TextLoader 
from langchain_core.prompts import PromptTemplate      
from langchain_core.output_parsers import StrOutputParser  
from langchain_google_genai import ChatGoogleGenerativeAI  
import os

print("Current working directory:", os.getcwd())

# Optional: set proxy if required (commented by default)
# os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
# os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"


# Load environment variables from .env (for API keys, etc.)
load_dotenv()

# Load text document (qft.txt)
loader = TextLoader("1_Document_Loader/qft.txt", encoding="utf-8")

# Load the file → returns a list of Document objects
docs = loader.load()

# Inspect loaded document
print(type(docs))          # Should be a <class 'list'>
print(len(docs))           # Number of documents loaded (likely 1 since it's a single text file)
print(docs[0])             # Show the Document object (includes content + metadata)
print(docs[0].page_content) # Print actual text content
print(docs[0].metadata)    # Print metadata (e.g., source file path)

#LLM Setup
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")   # Fetch API key from environment
llm = ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',     # Choose model
    google_api_key=GOOGLE_API_KEY,  
    temperature=0.2             # Lower temperature → more factual, less creative
)

# Define Prompt Template
prompt = PromptTemplate(
    template="Write poem with rhyme of the given text in 7 statements.\n{text}",
    input_variables=['text']   # Variable {text} will be replaced with document content
)

# Setup Output Parser
parser = StrOutputParser()   # Ensures response is returned as a plain string

# Create a chain: Prompt → LLM → Parser
chain = prompt | llm | parser

# Run the chain on our document
result = chain.invoke({'text': docs[0].page_content})

# Print Final Result (7-line summary)
print(result)
