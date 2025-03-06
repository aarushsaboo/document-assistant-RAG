# rag_chain.py
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

def create_llm(api_key):
    return GoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2
    )

def setup_qa_chain(db, llm, prompt):
    retriever = db.as_retriever(search_kwargs={"k": 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain
