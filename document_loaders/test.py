from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

# Character based splitting
splitter = CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1
)

data = TextLoader("document_loaders/notes.txt")

docs = data.load()

# docs is a list and the lost content Documents. Each Document has metadata and page_content

chunks = splitter.split_documents(docs)

print(docs) 
print(docs[0])
print(docs[0].page_content)

print("--------------Chunks data-----------------")

print("length of chunks: ", len(chunks))
print(chunks)
for i in chunks:
    print(i.page_content)
    print()
    print()
    print()