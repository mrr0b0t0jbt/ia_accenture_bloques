import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

def build_gemini_prompt(prompt):
    if not prompt:
        raise ValueError("La entrada no puede estar vacía. Por favor, proporciona un prompt válido.")
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
    try:
        response = requests.post(url, headers=headers, data=payload)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Ocurrio un error de conexión: {e}")
    if not response.ok:
        raise RuntimeError(f"Ocurrio un error de conexión: {response.status_code}")
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as e:
        raise RuntimeError(f"Error decodificando la respuesta JSON o viene vacía: {e}")
        
def extract_gemini_response(data):
    try:
        return data["steps"][1]["content"][0].get("text", "No se encontro texto en la respuesta")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Error extrayendo la respuesta: {e}")

def get_gemini_response(prompt):
    url, headers, payload = build_gemini_prompt(prompt)
    response = send_gemini_request(url, headers, payload)
    return extract_gemini_response(response)

if __name__ == "__main__":
    response = get_gemini_response("Confirma en una frase corta que esta conexion funciona correctamente.")
    print(response)