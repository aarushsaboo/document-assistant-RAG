from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_db(documents, embedding_model, persist_directory):
    # In the newer version, Chroma automatically persists when created with persist_directory
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    # No need to call db.persist() anymore as it's handled automatically
    return db

def load_vector_db(persist_directory, embedding_model):
    db = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    return db