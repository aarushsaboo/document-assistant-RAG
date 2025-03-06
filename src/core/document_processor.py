# src/core/document_processor.py
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(directory):
    loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks