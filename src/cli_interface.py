import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import time
from src.rag_data_embedding import faiss_store, bm25_retriever
from src.rag_hybrid_retrival import hybrid_retrieve
from src.rag_technique import rerank_with_cross_encoder
from src.rag_gen_response import generate_response
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

st.title("Financial QA System")

@st.cache_resource
def load_finetuned_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("./sft_model")
        model = AutoModelForCausalLM.from_pretrained("./sft_model")
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
        return generator
    except Exception as e:
        st.error(f"Failed loading fine-tuned model: {e}")
        return None

mode = st.radio("Select Mode", ["RAG", "Fine-Tuned"])
query = st.text_input("Enter your question")

if query:
    start_time = time.time()
    if mode == "RAG":
        top_chunks = hybrid_retrieve(query, faiss_store, bm25_retriever, top_n=5)
        reranked_results = rerank_with_cross_encoder(query, top_chunks)
        answer = generate_response(query, reranked_results)
        confidence_score = sum([res.metadata.get('similarity', 0) for res in reranked_results]) / len(reranked_results) if reranked_results else 0
    else:  # Fine-Tuned mode
        generator = load_finetuned_model()
        if generator is None:
            answer = "Error loading fine-tuned model."
            confidence_score = 0
        else:
            try:
                generated = generator(query, max_length=200, num_return_sequences=1, do_sample=True, top_p=0.9, temperature=0.7, return_full_text=False)
                if generated and len(generated) > 0:
                    answer = generated[0].get('generated_text', '')
                else:
                    answer = "No response generated from fine-tuned model."
                confidence_score = 1.0
            except Exception as e:
                answer = f"Error during generation: {e}"
                confidence_score = 0

    elapsed_time = time.time() - start_time

    if answer:
        st.success(f"Answer: {answer}")
    st.write(f"Retrieval Confidence Score: {confidence_score:.4f}")
    st.write(f"Method Used: {mode}")
    st.write(f"Response Time: {elapsed_time:.2f} seconds")