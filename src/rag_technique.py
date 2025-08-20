
from hybrid_retrival import results
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_with_cross_encoder(query, retrieved_docs, top_k=5):
    pairs = []
    for doc in retrieved_docs:
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        pairs.append((query, content))
    scores = cross_encoder.predict(pairs)
    scored_docs = list(zip(scores, retrieved_docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    top_docs = [doc for score, doc in scored_docs[:top_k]]
    return top_docs

sample_query = "What was EPS in FY2024-25 vs. FY2023-24?"

reranked_results = rerank_with_cross_encoder(sample_query, results, top_k=5)
print("\nReranked Results:")
for idx, res in enumerate(reranked_results):
    print(f"\nReranked Result {idx+1}: {res.page_content if hasattr(res, 'page_content') else res}\n")