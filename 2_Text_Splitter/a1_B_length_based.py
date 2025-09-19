from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFium2Loader

import os

path=os.getcwd()
pdf_file=os.path.join(path,"2_Text_Splitter","ml.pdf")

loader=PyPDFium2Loader(pdf_file)

# docs=loader.lazy_load()
splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)


# result=splitter.split_documents(docs)

for i,chunk in enumerate(splitter.split_documents(loader.lazy_load())):
    print(i,"->",chunk.page_content)

