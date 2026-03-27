from langchain_community.document_loaders import TextLoader

data = TextLoader("document_loaders/notes.txt")

docs = data.load()

# docs is a list and the lost content Documents. Each Document has metadata and page_content

print(docs) 
print(docs[0])
print(docs[0].page_content)