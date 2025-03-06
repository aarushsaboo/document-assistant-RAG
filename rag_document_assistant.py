import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import google.generativeai as genai

# Configuration
GOOGLE_API_KEY = "AIzaSyC5zEinq8gaFKWr33_Mjusxbm-fyYS0YZA"  # Replace with your actual API key
DOCUMENTS_DIR = "documents"  # Directory to store your documents
DB_DIR = "db"  # Directory to store the vector database

# Initialize the Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

def initialize_document_db():
    """Initialize the document database by loading documents and creating embeddings."""
    # Create directories if they don't exist
    os.makedirs(DOCUMENTS_DIR, exist_ok=True)
    os.makedirs(DB_DIR, exist_ok=True)
    
    # Load documents from the documents directory
    print("Loading documents...")
    loader = DirectoryLoader(DOCUMENTS_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print(f"No documents found in {DOCUMENTS_DIR}. Please add PDF documents before querying.")
        return None
    
    print(f"Loaded {len(documents)} document chunks")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    
    # Create embeddings
    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vector store
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    db.persist()
    print(f"Created and persisted vector database in {DB_DIR}")
    
    return db

def load_existing_db():
    """Load an existing vector database."""
    if not os.path.exists(DB_DIR):
        print(f"No existing database found in {DB_DIR}")
        return None
    
    print("Loading existing vector database...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    return db

def setup_rag_chain(db):
    """Set up the RAG chain with the vector database and language model."""
    # Define the prompt template
    template = """
    Answer the question based on the provided context only.
    If you don't know the answer based on the context, just say you don't know.
    Don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    # Initialize the language model - updated to use the correct class and model name
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",  # Use a valid model name
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2
    )
    
    # Create the retrieval chain
    retriever = db.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    return qa_chain

def main():
    """Main function to run the RAG application."""
    print("Welcome to the RAG Document Assistant!")
    
    # Check if we have an existing database or need to create a new one
    db = load_existing_db()
    if db is None:
        db = initialize_document_db()
        if db is None:
            return
    
    # Set up the RAG chain
    qa_chain = setup_rag_chain(db)
    
    # Interactive query loop
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query.strip():
            continue
        
        print("\nSearching for relevant information...")
        try:
            # Update to use the new invoke method instead of __call__
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

if __name__ == "__main__":
    main()