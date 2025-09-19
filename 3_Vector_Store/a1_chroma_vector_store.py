from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma   #Deprecated from v1.0
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.schema import Document

#Created Lanchain documnets for diffrent Physicits,Mathematician,AstroPhysicists, Nuclear
doc1 = Document(
    page_content=(
        "Albert Einstein was a theoretical physicist who developed the theory of relativity, "
        "one of the two pillars of modern physics. His contributions to quantum theory, "
        "the photoelectric effect, and mass-energy equivalence fundamentally changed our "
        "understanding of space, time, and energy. He received the Nobel Prize in Physics."
    ),
    metadata={
        "name": "Albert Einstein",
        "field": "Physicist",
        "dob": "1879-03-14",
        "publications": "Annus Mirabilis papers, Relativity: The Special and General Theory",
        "college": "ETH Zurich"
    }
)

doc2 = Document(
    page_content=(
        "Isaac Newton formulated the laws of motion and universal gravitation, laying the "
        "foundations of classical mechanics. He made pioneering contributions to optics, "
        "mathematics, and calculus. His Principia Mathematica is a cornerstone of scientific "
        "knowledge that influenced generations of scientists and mathematicians."
    ),
    metadata={
        "name": "Isaac Newton",
        "field": "Mathematician/Physicist",
        "dob": "1643-01-04",
        "publications": "Philosophiæ Naturalis Principia Mathematica",
        "college": "Trinity College, Cambridge"
    }
)

doc3 = Document(
    page_content=(
        "Carl Sagan was an astrophysicist, cosmologist, and popular science communicator. "
        "He conducted research on planetary atmospheres, extraterrestrial life, and "
        "cosmology. He inspired millions through his books and the Cosmos series, making "
        "astronomy and space exploration accessible to the general public."
    ),
    metadata={
        "name": "Carl Sagan",
        "field": "Astrophysicist",
        "dob": "1934-11-09",
        "publications": "Cosmos, Pale Blue Dot",
        "college": "University of Chicago"
    }
)

doc4 = Document(
    page_content=(
        "Marie Curie conducted groundbreaking research on radioactivity, discovering polonium "
        "and radium. She was the first woman to win a Nobel Prize and remains the only person "
        "to win Nobel Prizes in two different scientific fields (Physics and Chemistry). "
        "Her work advanced medical treatments and nuclear physics."
    ),
    metadata={
        "name": "Marie Curie",
        "field": "Physicist/Chemist",
        "dob": "1867-11-07",
        "publications": "Recherches sur les Substances Radioactives",
        "college": "University of Paris"
    }
)

doc5 = Document(
    page_content=(
        "Richard Feynman was a theoretical physicist known for his work in quantum mechanics "
        "and quantum electrodynamics. He introduced Feynman diagrams, revolutionizing the "
        "way physicists calculate particle interactions. He was also a brilliant teacher "
        "who inspired generations through his lectures."
    ),
    metadata={
        "name": "Richard Feynman",
        "field": "Physicist",
        "dob": "1918-05-11",
        "publications": "The Feynman Lectures on Physics",
        "college": "MIT, Princeton University"
    }
)

doc6 = Document(
    page_content=(
        "Niels Bohr developed the Bohr model of the atom and made foundational contributions "
        "to quantum theory. He explained electron structure and energy levels, influencing "
        "atomic physics and the development of nuclear energy. Bohr received the Nobel Prize "
        "for his groundbreaking work in physics."
    ),
    metadata={
        "name": "Niels Bohr",
        "field": "Physicist",
        "dob": "1885-10-07",
        "publications": "On the Constitution of Atoms and Molecules",
        "college": "University of Copenhagen"
    }
)

doc7 = Document(
    page_content=(
        "Werner Heisenberg formulated the uncertainty principle, a cornerstone of quantum "
        "mechanics. His research in nuclear and quantum physics reshaped the understanding "
        "of atomic behavior. Heisenberg was awarded the Nobel Prize in Physics in 1932 for "
        "his pioneering contributions to theoretical physics."
    ),
    metadata={
        "name": "Werner Heisenberg",
        "field": "Physicist",
        "dob": "1901-12-05",
        "publications": "Über quantentheoretische Umdeutung kinematischer und mechanischer Beziehungen",
        "college": "University of Munich"
    }
)

doc8 = Document(
    page_content=(
        "Alan Turing was a mathematician and computer scientist who laid the theoretical "
        "foundations of modern computing. He contributed to codebreaking during World War II "
        "and developed the concept of the Turing machine, which formalized the notion of algorithms "
        "and computation."
    ),
    metadata={
        "name": "Alan Turing",
        "field": "Mathematician",
        "dob": "1912-06-23",
        "publications": "On Computable Numbers, Computing Machinery and Intelligence",
        "college": "King's College, Cambridge"
    }
)

doc9 = Document(
    page_content=(
        "Enrico Fermi was a physicist who created the first nuclear reactor and made major "
        "contributions to nuclear and particle physics. His research in beta decay and quantum "
        "theory advanced scientific understanding, and he received the Nobel Prize in Physics in 1938."
    ),
    metadata={
        "name": "Enrico Fermi",
        "field": "Physicist/Nuclear",
        "dob": "1901-09-29",
        "publications": "Nuclear Physics: A Course Given by Enrico Fermi",
        "college": "University of Pisa"
    }
)

doc10 = Document(
    page_content=(
        "Stephen Hawking made groundbreaking contributions to cosmology, black hole physics, "
        "and quantum gravity. He authored 'A Brief History of Time' and advanced understanding "
        "of singularities, Hawking radiation, and the evolution of the universe, making complex "
        "cosmological concepts accessible to the public."
    ),
    metadata={
        "name": "Stephen Hawking",
        "field": "Physicist/Cosmologist",
        "dob": "1942-01-08",
        "publications": "A Brief History of Time, The Universe in a Nutshell",
        "college": "University of Cambridge"
    }
)
docs = []
for i in range(1, 11):
    docs.append(eval(f"doc{i}"))



import os
load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN
os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    cache_folder="3_Vector_Store/huggingface"
)


vector_store=Chroma(
    embedding_function=embedding_model,
    persist_directory='3_Vector_Store/chroma_langchain_db',
    collection_name='sample'
)

vector_store.add_documents(docs)

stored_data = vector_store.get(include=['embeddings', 'metadatas', 'documents'])
for i, doc in enumerate(stored_data['documents']):
    print(f"Document {i+1}:")
    print("ID:", stored_data['ids'][i])
    print("Content:", doc[:100], "...")
    print("Metadata:", stored_data['metadatas'][i])
    #print("Embedding length:", stored_data['embeddings'][i])
    print("Embedding length:", len(stored_data['embeddings'][i]))
    print("-" * 110)

#Similarity Serach
query=vector_store.similarity_search(
    query='Who among is Physicists',
    k=3
)
print(query)

#Similarity Serach with Score
query1=vector_store.similarity_search_with_score(
    query='Who among is Physicists',
    k=3
)

#Meta-Data Filtering
query3=vector_store.similarity_search_with_score(
    query="",
    k=3,
    filter={"field":"Physicists"}

)


#Update Document
updated_doc1 = Document(
    page_content=(
        "Brian Cox is a British physicist and professor of particle physics who has made significant "
        "contributions to the ATLAS experiment at CERN and research in high-energy physics. "
        "He is also a popular science communicator, presenting numerous TV programs and lectures "
        "to make complex physics concepts accessible to the general public."
    ),
    metadata={
        "name": "Brian Cox",
        "field": "Physicist",
        "dob": "1968-03-03",
        "publications": "Why Does E=mc²?, The Quantum Universe",
        "college": "University of Manchester"
    }
)

vector_store.update_document(document_id='7a5c729c-fc48-42cb-a21d-4e4e4b4b2210',document=updated_doc1)

stored_data1=vector_store.get(include=['embeddings', 'metadatas', 'documents'])
for i, doc in enumerate(stored_data1['documents']):
    print(f"Document {i+1}:")
    print("ID:", stored_data['ids'][i])
    print("Content:", doc[:100], "...")
    print("Metadata:", stored_data['metadatas'][i])
    #print("Embedding length:", stored_data['embeddings'][i])
    print("Embedding length:", len(stored_data['embeddings'][i]))
    print("-" * 110)


#Delete Document
vector_store.delete(ids='a37fc40c-bad8-40cb-9ad9-f8b66a5bfca9')
stored_data2=vector_store.get(include=['embeddings', 'metadatas', 'documents'])
for i, doc in enumerate(stored_data2['documents']):
    print(f"Document {i+1}:")
    print("ID:", stored_data['ids'][i])
    print("Content:", doc[:100], "...")
    print("Metadata:", stored_data['metadatas'][i])
    #print("Embedding length:", stored_data['embeddings'][i])
    print("Embedding length:", len(stored_data['embeddings'][i]))
    print("-" * 110)



