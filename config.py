# config.py
import os
import google.generativeai as genai

# Configuration variables
GOOGLE_API_KEY = "AIzaSyC5zEinq8gaFKWr33_Mjusxbm-fyYS0YZA"
DOCUMENTS_DIR = "documents"
DB_DIR = "db"

# Initialize Google API
genai.configure(api_key=GOOGLE_API_KEY)