# src/main.py
import os
from core.config import GOOGLE_API_KEY, DOCUMENTS_DIR, DB_DIR
from core.document_processor import load_documents, split_documents
from core.vectordb import get_embeddings, create_vector_db, load_vector_db
from core.rag_chain import create_prompt, create_llm, setup_qa_chain

def initialize_document_db():
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)
    
    print("Loading documents...")
    documents = load_documents(DOCUMENTS_DIR)
    
    if not documents:
        print(f"No documents found in {DOCUMENTS_DIR}. Please add PDF documents before querying.")
        return None
    
    print(f"Loaded {len(documents)} document chunks")
    
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    print("Creating embeddings...")
    embeddings = get_embeddings()
    
    db = create_vector_db(chunks, embeddings, DB_DIR)
    print(f"Created and persisted vector database in {DB_DIR}")
    
    return db

def load_existing_db():
    if not os.path.exists(DB_DIR):
        print(f"No existing database found in {DB_DIR}")
        return None
    
    print("Loading existing vector database...")
    embeddings = get_embeddings()
    db = load_vector_db(DB_DIR, embeddings)
    return db

def process_query(qa_chain, query):
    print("\nSearching for relevant information...")
    try:
        result = qa_chain.invoke({"query": query})
        
        print("\nAnswer:")
        print(result["result"])
        
        print("\nSources:")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"Source {i}:")
            print(f"- Content: {doc.page_content[:150]}...")
            print(f"- Source: {doc.metadata.get('source', 'Unknown')}, Page: {doc.metadata.get('page', 'Unknown')}")
            print()
    except Exception as e:
        print(f"An error occurred: {e}")
        print("If this is a model-related error, try checking available models or using a different model.")

def main():
    print("Welcome to the RAG Document Assistant!")
    
    db = load_existing_db()
    if db is None:
        db = initialize_document_db()
        if db is None:
            return
    
    prompt = create_prompt()
    llm = create_llm(GOOGLE_API_KEY)
    qa_chain = setup_qa_chain(db, llm, prompt)
    
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query.strip():
            continue
        
        process_query(qa_chain, query)

if __name__ == "__main__":
    main()