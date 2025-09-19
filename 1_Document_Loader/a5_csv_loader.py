from dotenv import load_dotenv
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

import os
load_dotenv()

# Optional: set proxy if required (commented by default)
# os.environ["HTTP_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"
# os.environ["HTTPS_PROXY"] = "http://edcguest:edcguest@172.31.102.14:3128"

#PATH
path=os.getcwd()
csv_file=os.path.join(path,"1_Document_Loader","customers.csv")

#Loader
loader=CSVLoader(file_path=csv_file)

docs=(loader.load())#Lazy Load Save Memory

#Inspect
print(type(docs))
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)

#Prompt

prompt=PromptTemplate(
    template="Write a sonnet for the given text.\n{text}",
    input_variables=['text']
)

#LLM 
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
llm=ChatGoogleGenerativeAI(
    model='gemini-2.5-pro',
    google_api_key=GOOGLE_API_KEY,
    temperature=0.8
)

#Parser
parser=StrOutputParser()

#Chain Create
chain=prompt | llm | parser

#Execute Chain
result=chain.invoke({'text':docs[0].page_content})
print(result)
