import os
import streamlit as st
from core.config import GOOGLE_API_KEY, DOCUMENTS_DIR, DB_DIR
from core.document_processor import load_documents, split_documents
from core.vectordb import get_embeddings, create_vector_db, load_vector_db
from core.rag_chain import create_prompt, create_llm, setup_qa_chain

def initialize_document_db():
    os.makedirs(DOCUMENTS_DIR, exist_ok=True) # if it already exists, don't throw an error
    os.makedirs(DB_DIR, exist_ok=True)
    
    st.info("Loading documents...")
    documents = load_documents(DOCUMENTS_DIR)
    
    if not documents:
        st.error(f"No documents found in {DOCUMENTS_DIR}. Please add PDF documents before querying.")
        return None
    
    st.info(f"Loaded {len(documents)} document chunks")
    
    chunks = split_documents(documents)
    st.info(f"Split into {len(chunks)} chunks")
    
    st.info("Creating embeddings...")
    embeddings = get_embeddings()
    
    db = create_vector_db(chunks, embeddings, DB_DIR)
    st.success(f"Created and persisted vector database in {DB_DIR}")
    
    return db

def load_existing_db():
    # Check if the actual FAISS files exist
    index_faiss_path = os.path.join(DB_DIR, "index.faiss")
    index_pkl_path = os.path.join(DB_DIR, "index.pkl")
    
    if not os.path.exists(index_faiss_path) or not os.path.exists(index_pkl_path):
        
        st.warning(f"No existing database found in {DB_DIR}")
        return None
    
    st.info("Loading existing vector database...")
    embeddings = get_embeddings()
    db = load_vector_db(DB_DIR, embeddings)
    return db

def process_query(qa_chain, query):
    if not query:
        return
        
    with st.spinner("Searching for relevant information..."):
        try:
            result = qa_chain.invoke({"query": query}) 
            
            st.write("### Answer:")
            st.write(result["result"]) 

            st.write("### Sources:")
            for i, doc in enumerate(result["source_documents"], 1):
                with st.expander(f"Source {i}"):
                    st.write(f"**Content:** {doc.page_content[:150]}...")
                    st.write(f"**Source:** {doc.metadata.get('source', 'Unknown')}, **Page:** {doc.metadata.get('page', 'Unknown')}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("If this is a model-related error, try checking available models or using a different model.")

def main():
    st.set_page_config(page_title="RAG Document Assistant", layout="wide")
    st.title("RAG Document Assistant")
    
    # Initialize session state for the database and QA chain
    if 'db' not in st.session_state:
        st.session_state.db = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    
    # Sidebar for database initialization
    with st.sidebar:
        st.header("Database Management")
        if st.button("Initialize/Reload Database"):
            db = load_existing_db()
            if db is None:
                db = initialize_document_db()
            st.session_state.db = db
            
            if st.session_state.db is not None:
                prompt = create_prompt()
                llm = create_llm(GOOGLE_API_KEY)
                st.session_state.qa_chain = setup_qa_chain(st.session_state.db, llm, prompt)
                st.success("Database and QA chain initialized successfully!")
        
        st.markdown("---")
        st.markdown(f"Document Directory: `{DOCUMENTS_DIR}`")
        st.markdown(f"Database Directory: `{DB_DIR}`")
    
    # Main content area
    if st.session_state.db is None:
        st.warning("Please initialize the database first by clicking the button in the sidebar.")
    else:
        query = st.text_input("Enter your question:")
        if st.button("Submit") or query:
            process_query(st.session_state.qa_chain, query)

if __name__ == "__main__":
    main()