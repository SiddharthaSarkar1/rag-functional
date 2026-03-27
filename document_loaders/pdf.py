from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter, RecursiveCharacterTextSplitter

# #Token based splitting (ticktoken tokenizer is used)
# splitter = TokenTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=10
# )

# RecursiveCharacterTextSplitter ["\n\n", "\n", " ", ""] - (this is a type of character based splitting)
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap=10
)

data = PyPDFLoader("document_loaders/GRU.pdf")

docs = data.load()

chunks = splitter.split_documents(docs)

print(f"Length of chunks: {len(chunks)}")
print(chunks)
print(chunks[0].page_content)