import google.generativeai as genai

# Replace with your actual API key
genai.configure(api_key="AIzaSyC5zEinq8gaFKWr33_Mjusxbm-fyYS0YZA")

# List available models
models = genai.list_models()
for model in models:
    print(model.name)