from langchain.text_splitter import CharacterTextSplitter

text="""
The Higgs boson, sometimes called the Higgs particle,[9][10] is an elementary particle in the Standard Model of particle physics produced by the quantum excitation of the Higgs field,[11][12] one of the fields in particle physics theory.[12] In the Standard Model, the Higgs particle is a massive scalar boson that couples to (interacts with) particles whose mass arises from their interactions with the Higgs Field, has zero spin, even (positive) parity, no electric charge, and no colour charge.[13] It is also very unstable, decaying into other particles almost immediately upon generation.

The Higgs field is a scalar field with two neutral and two electrically charged components that form a complex doublet of the weak isospin SU(2) symmetry. Its "sombrero potential" leads it to take a nonzero value everywhere (including otherwise empty space), which breaks the weak isospin symmetry of the electroweak interaction and, via the Higgs mechanism, gives a rest mass to all massive elementary particles of the Standard Model, including the Higgs boson itself. The existence of the Higgs field became the last unverified part of the Standard Model of particle physics, and for several decades was considered "the central problem in particle physics".
"""

splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

result=splitter.split_text(text)

for i, chunk in enumerate(result): 
    print(i,"->",chunk)