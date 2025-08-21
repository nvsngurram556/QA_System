
from langchain_huggingface import HuggingFaceEmbeddings  # Requires: pip install langchain-huggingface
from rag_data_embedding import process_text, create_faiss_vector_store, create_bm25_retriever, faiss_store, bm25_retriever
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_query(query):
    query = query.lower()
    query = re.sub(r'[^\w\s]', '', query)
    tokens = query.split()
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return " ".join(filtered_tokens)

def hybrid_retrieve(query, faiss_store, bm25_retriever, top_n=5):
    processed_query = preprocess_query(query)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = embeddings.embed_query(processed_query)

    # Retrieve from FAISS
    faiss_results = faiss_store.similarity_search_by_vector(query_embedding, k=top_n)

    # Retrieve from BM25
    bm25_results = bm25_retriever.get_relevant_documents(processed_query)[:top_n]

    # Combine results by union (simple concatenation)
    combined_results = faiss_results + bm25_results
    return combined_results


sample_query = "What was EPS in FY2024-25 vs. FY2023-24?"
# Demonstrate hybrid retrieval with a sample query
results = hybrid_retrieve(sample_query, faiss_store, bm25_retriever, top_n=5)

print(f"Hybrid Retrieval Results for query '{sample_query}':")
for idx, res in enumerate(results):
    print(f"Result {idx+1}: {res.page_content if hasattr(res, 'page_content') else res}")