import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import euclidean, cityblock
from langchain_experimental.text_splitter import SemanticChunker
# from langchain_community.embeddings import HuggingFaceEmbeddings    #Deprecated
from langchain_huggingface import HuggingFaceEmbeddings

class FlexibleSemanticChunker(SemanticChunker):
    """SemanticChunker with configurable similarity metric."""

    def __init__(self, embeddings, similarity="cosine", **kwargs):
        super().__init__(embeddings=embeddings, **kwargs)
        self.similarity = similarity.lower()

    def _similarity(self, emb1, emb2):
        """Compute similarity score based on chosen metric."""
        
        if self.similarity == "cosine":
            # Cosine similarity: [-1, 1]
            return np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
        
        elif self.similarity == "dot":
            # Dot product (depends on magnitude)
            return np.dot(emb1, emb2)
        
        elif self.similarity == "euclidean":
            # Euclidean distance (smaller = closer) → invert
            return -euclidean(emb1, emb2)
        
        elif self.similarity == "manhattan":
            # Manhattan (L1) distance (smaller = closer) → invert
            return -cityblock(emb1, emb2)
        
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity}")

from dotenv import load_dotenv
import os

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    cache_folder="2_Text_Splitter/huggingface"
)

text = """
The little pied cormorant (Microcarbo melanoleucos) is a species of waterbird in the cormorant family, Phalacrocoracidae. It is a common bird found around the coasts, islands, estuaries and inland waters of Australia, New Guinea, New Zealand, Timor-Leste and Indonesia, and around the islands of the south-western Pacific and the subantarctic.

The Fugitive Slave Act or Fugitive Slave Law was a statute passed by the 31st United States Congress on September 18, 1850,[1] as part of the Compromise of 1850 between Southern interests in slavery and Northern Free-Soilers.

The Act was one of the most controversial elements of the 1850 compromise and heightened Northern fears of a slave power conspiracy. It required that all escaped slaves, upon capture, be returned to the slave-owner and that officials and citizens of free states had to cooperate.

Alicia is the seventh studio album by Alicia Keys (pictured) and released on September 18, 2020. Alicia's mostly low-tempo and melodically subtle music reconciles her experimental direction with bass drum–driven R&B and piano-based balladry. The songs explore identity as a multifaceted concept, sociopolitical concerns, and forms of love within multiple frameworks.
"""

# Cosine similarity (default in most NLP)
splitter_cosine = FlexibleSemanticChunker(
    embeddings=embedding_model,
    similarity="cosine",
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.05
)

# Euclidean similarity
splitter_euclidean = FlexibleSemanticChunker(
    embeddings=embedding_model,
    similarity="euclidean",
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=0.05
)

print("---- Cosine Similarity ----")
for i, chunk in enumerate(splitter_cosine.split_text(text)):
    print(i, "->", chunk)

print("\n---- Euclidean Similarity ----")
for i, chunk in enumerate(splitter_euclidean.split_text(text)):
    print(i, "->", chunk)
