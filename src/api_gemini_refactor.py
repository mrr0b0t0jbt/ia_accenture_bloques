import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

def build_gemini_prompt(prompt):
    url = "https://generativelanguage.googleapis.com/v1beta/interactions"
    api_key = os.getenv("GEMINI_API_KEY")
    payload = json.dumps({
        "model": "gemini-3.5-flash-lite",
        "input": prompt
    })
    headers = {
        'x-goog-api-key': api_key,
        'Content-Type': 'application/json'
    }
    return url, headers, payload


def send_gemini_request(url, headers, payload):
    response = requests.request("POST", url, headers=headers, data=payload)
    return response

def extract_gemini_response(response):
    data = response.json()
    if response.status_code == 200:
        return data["steps"][1]["content"][0].get("text", "No output text found.")
    else:
        print(f"Error occurred: {response.status_code} - {response.text}")
        return "Error occurred while fetching response."

def get_gemini_response(prompt):
    url, headers, payload = build_gemini_prompt(prompt)
    response = send_gemini_request(url, headers, payload)
    return extract_gemini_response(response)

print(get_gemini_response("Confirma en una frase corta que esta conexion funciona correctamente."))