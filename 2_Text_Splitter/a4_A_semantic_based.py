from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load API key
load_dotenv() 
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"

# Initialize embedding model (needed for SemanticChunker)
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    cache_folder="2_Text_Splitter/huggingface"
)

# Create semantic chunker
splitter = SemanticChunker(#Default- Cosine Similarity
    embeddings=embedding_model,
    breakpoint_threshold_type="standard_deviation",  # how to find breakpoints
    breakpoint_threshold_amount=0.2                   # sensitivity
)

# Example text
text = """
The little pied cormorant (Microcarbo melanoleucos) is a species of waterbird in the cormorant family, Phalacrocoracidae. It is a common bird found around the coasts, islands, estuaries and inland waters of Australia, New Guinea, New Zealand, Timor-Leste and Indonesia, and around the islands of the south-western Pacific and the subantarctic.

The Fugitive Slave Act or Fugitive Slave Law was a statute passed by the 31st United States Congress on September 18, 1850,[1] as part of the Compromise of 1850 between Southern interests in slavery and Northern Free-Soilers.

The Act was one of the most controversial elements of the 1850 compromise and heightened Northern fears of a slave power conspiracy. It required that all escaped slaves, upon capture, be returned to the slave-owner and that officials and citizens of free states had to cooperate.

Alicia is the seventh studio album by Alicia Keys (pictured) and released on September 18, 2020. Alicia's mostly low-tempo and melodically subtle music reconciles her experimental direction with bass drum–driven R&B and piano-based balladry. The songs explore identity as a multifaceted concept, sociopolitical concerns, and forms of love within multiple frameworks.
"""

# Split into semantic chunks
docs = splitter.split_text(text)

# Print results
for i, chunk in enumerate(docs):
    print(i, "->", chunk)

# import requests
# print(requests.get("https://ai.google.dev").status_code)