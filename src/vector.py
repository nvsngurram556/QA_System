import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_faiss_vector_store(text, path="faiss_index"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local(path)

    # Create BM25Retriever from chunks
    bm25_retriever = BM25Retriever.from_texts(chunks)
    return vector_store, bm25_retriever

# Load FAISS vector store
def load_faiss_vector_store(path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(path, embeddings,  
                 allow_dangerous_deserialization=True)
    return vector_store

def build_qa_chain(vector_store=None, bm25_retriever=None, vector_store_path="faiss_index"):
    # If not provided, load FAISS vector store and do not use BM25
    if vector_store is None:
        vector_store = load_faiss_vector_store(vector_store_path)
        retriever = vector_store.as_retriever()
    else:
        retriever = vector_store.as_retriever()

    # If BM25 retriever is provided, ensemble with FAISS retriever
    if bm25_retriever is not None:
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever, bm25_retriever],
            weights=[0.5, 0.5]
        )
        used_retriever = ensemble_retriever
    else:
        used_retriever = retriever

    llm = Ollama(model="llama-2-7b")
    qa_chain = load_qa_chain(llm, chain_type="stuff")
    qa_chain = RetrievalQA(retriever=used_retriever, combine_documents_chain=qa_chain)
    return qa_chain



# Streamlit App
st.title("RAG Chatbot with FAISS and LLaMA")
st.write("Upload a PDF and ask questions based on its content.")

uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

# These variables will be used to hold both retrievers
qa_chain = None

if uploaded_file is not None:
    pdf_path = f"data/raw/{uploaded_file.name}"
    os.makedirs("data/raw", exist_ok=True)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    text = extract_text_from_pdf(pdf_path)

    st.info("Creating FAISS vector store and BM25 retriever...")
    vector_store, bm25_retriever = create_faiss_vector_store(text)

    st.info("Initializing chatbot...")
    qa_chain = build_qa_chain(vector_store=vector_store, bm25_retriever=bm25_retriever)
    st.success("Chatbot is ready!")

if qa_chain is not None:
    question = st.text_input("Ask a question about the uploaded PDF:")
    if question:
        st.info("Querying the document...")
        answer = qa_chain.run(question)
        st.success(f"Answer: {answer}")
