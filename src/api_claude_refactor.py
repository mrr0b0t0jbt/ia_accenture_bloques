import time

import requests
import os
import json
from dotenv import load_dotenv

load_dotenv()

def build_claude_prompt(prompt):
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

def send_claude_request(url, headers, payload):
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

def calculate_cost(response, input_pricing, output_pricing):
    try:
        use = response["usage"]
        input_tokens = use["total_input_tokens"]
        output_tokens = use["total_output_tokens"]
        return (input_tokens * input_pricing) + (output_tokens * output_pricing)
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Error calculando el costo: {e}")

def extract_claude_response(data):
    try:
        return data["steps"][1]["content"][0].get("text", "No se encontro texto en la respuesta")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Error extrayendo la respuesta: {e}")

def get_claude_response(prompt):
    try:
        url, headers, payload = build_claude_prompt(prompt)
        response = send_claude_request(url, headers, payload)
        cost = calculate_cost(response, input_pricing=0.0001, output_pricing=0.0002)
        print (f"Costo calculado: {cost}")
        return extract_claude_response(response)
    except KeyError as ve:
        print(f"Error extract response: {ve}")
    except Exception as e:
        print(f"unexpected error: {e}")

def latency_test(ciclos: int = 5):
    latencies = []
    for i in range(ciclos):
        url, headers, payload = build_claude_prompt("Cuanto es 2 + 2? Dame solo el resultado")
        index = time.time()
        data = send_claude_request(url, headers, payload)
        duration = time.time() - index
        print(f"Duración de la petición {i + 1}: {duration} segundos")
        latencies.append(duration)
    prom_latency = sum(latencies) / len(latencies)
    print(f"Latencia promedio después de {ciclos} ciclos: {prom_latency} segundos")  
    
if __name__ == "__main__":
    response, cost = get_claude_response("Confirma en una frase corta que esta conexion funciona correctamente.")
    print(f"Respuesta: {response}")
    print(f"Costo: {cost}")
    latency_test(10)