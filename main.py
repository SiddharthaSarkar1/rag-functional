from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate

from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")

data = PyPDFLoader("document_loaders/GRU.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(docs)

# Prompt Templete
templete = ChatPromptTemplate.from_messages([
    ("system", "you are an AI that summerizes the text"),
    ("human", "{data}")
])

prompt = templete.format_messages(data=docs[0].page_content)

result = model.invoke(prompt)

print(result.content)
