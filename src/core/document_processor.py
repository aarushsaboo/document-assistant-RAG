# src/core/document_processor.py
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(directory):
    loader = DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load() # reads all PDFs in the folder
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # each chunk has 1000 characters
        chunk_overlap=200, # consecutive chunks overlap by 200 characters to maintain context.
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks