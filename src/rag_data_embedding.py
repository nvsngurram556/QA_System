import os
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Requires: pip install langchain-huggingface
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from rag_data_preprocess import clean_text, segment_sections

def create_faiss_vector_store(chunks, index_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(index_path)
    return vector_store

def create_bm25_retriever(chunks):
    bm25_retriever = BM25Retriever.from_texts(chunks)
    return bm25_retriever

def process_text(text):
    preprocessed_data = clean_text(text)
    segmented_data = segment_sections(preprocessed_data)
    # Handle case where segmented_data is a dictionary by joining all values into a single string
    if isinstance(segmented_data, dict):
        segmented_text = "\n\n".join(segmented_data.values())
    else:
        segmented_text = segmented_data
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20)
    chunks = text_splitter.split_text(segmented_text)
    index_path = "data/processed/faiss_index"
    faiss_store = create_faiss_vector_store(chunks, index_path)
    bm25_retriever = create_bm25_retriever(chunks)
    return faiss_store, bm25_retriever

with open("data/preprocessed/combined_text.txt", "r", encoding="utf-8") as f:
    raw_file = f.read()

faiss_store, bm25_retriever = process_text(raw_file)
