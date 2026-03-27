from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

# Example of embedding a single query
query = "What is the capital of France?"
query_embedding = embeddings.embed_query(query)
print(f"Embedding for query (first 5 values): {query_embedding[:5]}... (length: {len(query_embedding)})")

# Example of embedding multiple documents
documents = [
    "The capital of France is Paris.",
    "Philipp likes to eat pizza.",
    "The weather is sunny today."
]

document_embeddings = embeddings.embed_documents(documents)
print(f"\nNumber of document embeddings generated: {len(document_embeddings)}")
print(f"Embedding for document 1 (first 5 values): {document_embeddings[0][:5]}... (length: {len(document_embeddings[0])})")

# Example of calculating cosine similarity (useful in RAG applications)
# This requires the scikit-learn library (pip install scikit-learn)
print("\nCosine similarities with the query:")
for i, doc_embed in enumerate(document_embeddings):
    similarity = cosine_similarity([query_embedding], [doc_embed])[0][0]
    print(f"Document {i + 1}: '{documents[i]}'")
    print(f"Cosine similarity with query: {similarity:.4f}")
    print("---")
