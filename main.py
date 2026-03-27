from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")

# Prompt Templete
templete = ChatPromptTemplate.from_messages([
    ("system", "you are an AI that summerizes the text"),
    ("human", "{data}")
])
