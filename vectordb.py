# vectordb.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_db(documents, embedding_model, persist_directory):
    db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    db.persist()
    return db

def load_vector_db(persist_directory, embedding_model):
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
