import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import time
from src.rag_data_embedding import faiss_store, bm25_retriever
from src.rag_hybrid_retrival import hybrid_retrieve
from src.rag_technique import rerank_with_cross_encoder
from src.rag_gen_response import generate_response

st.title("Financial QA System")

mode = st.radio("Select Mode", ["RAG", "Fine-Tuned"])
query = st.text_input("Enter your question")

if query:
    start_time = time.time()
    if mode == "RAG":
        top_chunks = hybrid_retrieve(query, faiss_store, bm25_retriever, top_n=5)
        reranked_results = rerank_with_cross_encoder(query, top_chunks)
        answer = generate_response(query, reranked_results)
        confidence_score = sum([res.metadata.get('similarity', 0) for res in reranked_results]) / len(reranked_results) if reranked_results else 0
    else:
        st.info("Fine-tuned model response generation not implemented yet.")
        answer = ""
        confidence_score = 0
    elapsed_time = time.time() - start_time

    if answer:
        st.success(f"Answer: {answer}")
    st.write(f"Retrieval Confidence Score: {confidence_score:.4f}")
    st.write(f"Method Used: {mode}")
    st.write(f"Response Time: {elapsed_time:.2f} seconds")
