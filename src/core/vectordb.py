from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_db(documents, embedding_model, persist_directory):
    # Create the FAISS index from documents
    db = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model
    )
    # Save the FAISS index
    db.save_local(persist_directory)
    return db

def load_vector_db(persist_directory, embedding_model):
    # Load the FAISS index
    if not os.path.exists(persist_directory):
        raise ValueError(f"No database found at {persist_directory}")
    
    return FAISS.load_local(
        persist_directory, 
        embedding_model,
        allow_dangerous_deserialization=True
    )