# src/core/config.py
import os
import google.generativeai as genai

# Get the project root directory (two levels up from config.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration variables
GOOGLE_API_KEY = "AIzaSyC5zEinq8gaFKWr33_Mjusxbm-fyYS0YZA"
DOCUMENTS_DIR = os.path.join(PROJECT_ROOT, "documents")
DB_DIR = os.path.join(PROJECT_ROOT, "db")

# Initialize Google API
genai.configure(api_key=GOOGLE_API_KEY)

