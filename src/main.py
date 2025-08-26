# Add this at the very top of main.py, before other imports
import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
from core.config import GOOGLE_API_KEY, DOCUMENTS_DIR, DB_DIR
from core.document_processor import load_documents, split_documents
from core.vectordb import get_embeddings, create_vector_db, load_vector_db
from core.rag_chain import create_prompt, create_llm, setup_qa_chain

SUBJECTS = ["English", "History", "Science", "Mathematics", "General"]

def get_subject_paths(subject):
    subject_docs_dir = os.path.join(DOCUMENTS_DIR, subject.lower())
    subject_db_dir = os.path.join(DB_DIR, subject.lower())
    return subject_docs_dir, subject_db_dir

# we're returning None if no documents are found.
def initialize_document_db(subject):
    subject_docs_dir, subject_db_dir = get_subject_paths(subject)
    os.makedirs(subject_docs_dir, exist_ok=True)
    os.makedirs(subject_db_dir, exist_ok=True)

    documents = load_documents(subject_docs_dir)
    if not documents:
        st.error(f"No documents found in {subject_docs_dir}. Please add PDF documents for {subject}.")
        return None

    chunks = split_documents(documents)
    embeddings = get_embeddings()
    db = create_vector_db(chunks, embeddings, subject_db_dir)
    return db

def load_existing_db(subject):
    _, subject_db_dir = get_subject_paths(subject)
    index_faiss_path = os.path.join(subject_db_dir, "index.faiss")
    index_pkl_path = os.path.join(subject_db_dir, "index.pkl")

    if not os.path.exists(index_faiss_path) or not os.path.exists(index_pkl_path):
        return None

    embeddings = get_embeddings()
    db = load_vector_db(subject_db_dir, embeddings)
    return db

def process_query(qa_chain, query):
    if not query:
        return
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
        st.error(f"Error: {e}")

def main():
    st.set_page_config(page_title="RAG Document Assistant", layout="wide")
    st.title("RAG Document Assistant")

    if 'db' not in st.session_state:
        st.session_state.db = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'current_subject' not in st.session_state:
        st.session_state.current_subject = None

    with st.sidebar:
        st.header("Subject Selection")
        selected_subject = st.selectbox("Choose a subject:", SUBJECTS)

        if st.session_state.current_subject != selected_subject:
            st.session_state.current_subject = selected_subject
            st.session_state.db = None
            st.session_state.qa_chain = None

        st.header("Database Management")

        if st.session_state.current_subject:
            if st.session_state.db is not None:
                st.success("Database loaded")
            else:
                st.warning("Database not loaded")

        if st.button("Reload Database"):
            if st.session_state.current_subject:
                db = load_existing_db(st.session_state.current_subject)
                if db is None:
                    db = initialize_document_db(st.session_state.current_subject)
                st.session_state.db = db

                if st.session_state.db is not None:
                    prompt = create_prompt()
                    llm = create_llm(GOOGLE_API_KEY)
                    st.session_state.qa_chain = setup_qa_chain(st.session_state.db, llm, prompt)
                    st.success("Database reloaded")

        if st.session_state.current_subject:
            subject_docs_dir, subject_db_dir = get_subject_paths(st.session_state.current_subject)
            st.markdown("### Paths:")
            st.markdown(f"Documents: `{subject_docs_dir}`")
            st.markdown(f"Database: `{subject_db_dir}`")

    if st.session_state.current_subject and st.session_state.db is None:
        db = load_existing_db(st.session_state.current_subject)
        if db is None:
            db = initialize_document_db(st.session_state.current_subject)
        st.session_state.db = db

        if st.session_state.db is not None:
            prompt = create_prompt()
            llm = create_llm(GOOGLE_API_KEY)
            st.session_state.qa_chain = setup_qa_chain(st.session_state.db, llm, prompt)

    if not st.session_state.current_subject:
        st.warning("Select a subject to begin.")
    elif st.session_state.db is None:
        st.warning(f"No database found for {st.session_state.current_subject}. Add PDFs and reload.")
    else:
        query = st.text_input("Enter your question:")
        if st.button("Submit") or query:
            process_query(st.session_state.qa_chain, query)

if __name__ == "__main__":
    main()