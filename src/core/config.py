# src/core/config.py
import os
from langchain_google_genai import GoogleGenerativeAI

# Get the project root directory (two levels up from config.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration variables
GOOGLE_API_KEY = "AIzaSyAPDeEAmZK7Ls5ZC2Ahr-CrmjkmDNRQnwg"
DOCUMENTS_DIR = os.path.join(PROJECT_ROOT, "documents")
DB_DIR = os.path.join(PROJECT_ROOT, "db")

# Initialize LLM
llm = GoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)