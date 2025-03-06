# src/core/rag_chain.py
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

def create_prompt():
    template = """
    Answer the question based on the provided context only.
    If you don't know the answer based on the context, just say you don't know.
    Don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    
    return PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
# PromptTemplate is a wrapper that allows you to structure the prompt before you have the actual values for variables.

def create_llm(api_key):
    return GoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2
    )

def setup_qa_chain(db, llm, prompt):
    retriever = db.as_retriever(search_kwargs={"k": 4}) # This turns the database into a retriever. ie When searching for an answer, retrieve the top 4 most relevant chunks from the vector DB.. Finds relevant document chunks from db: the Chroma vector database (stores embeddings of documents).
    qa_chain = RetrievalQA.from_chain_type( # question-answering pipeline
        llm=llm,
        chain_type="stuff", # chain_type tells it how to process retrieved documents & "stuff" means concatenate all retrieved documents into one prompt
        retriever=retriever,
        return_source_documents=True, # will return the actual documents used
        chain_type_kwargs={"prompt": prompt} # ensures the custom PromptTemplate is used
    )
    return qa_chain