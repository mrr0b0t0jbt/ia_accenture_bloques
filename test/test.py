import os
import requests
import json
from google import genai
from dotenv import load_dotenv

#cargar apikey de gemini desde .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)
client = genai.Client()

def get_gemini_response(prompt):
    try:
        response = client.interactions.create(
            model="gemini-3.5-flash-lite",
            input=prompt
        )
        return response.output_text
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Error occurred while fetching response."