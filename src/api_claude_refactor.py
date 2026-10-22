import time

import requests
import os
import json
import re
from dotenv import load_dotenv

load_dotenv()

def build_claude_prompt(prompt, system=None, output_schema=None):
    if not prompt:
        raise ValueError("La entrada no puede estar vacía. Por favor, proporciona un prompt válido.")
    url = "https://api.anthropic.com/v1/messages"
    api_key = os.getenv("CLAUDE_API_KEY")
    body = {
        "model": "claude-sonnet-5-5",
        "max_tokens": 1024,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    if system:
        body["system"] = system
    if output_schema:
        body["output_config"] = {"format": {"type": "json_schema", "schema": output_schema}}
    payload = json.dumps(body)
    headers = {
        "anthropic-version": "2023-06-01",
        'x-api-key': api_key,
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
        input_tokens = use["input_tokens"]
        output_tokens = use["output_tokens"]
        return (input_tokens * input_pricing) + (output_tokens * output_pricing)
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Error calculando el costo: {e}")

def extract_claude_response(data):
    try:
        return data["content"][0].get("text", "No se encontro texto en la respuesta")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Error extrayendo la respuesta: {e}")

def parse_json_response(text):
    # algunos modelos envuelven el JSON en un bloque markdown ```json ... ```
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        raise ValueError(f"La respuesta del modelo no es JSON válido: {e}")

def get_claude_response(prompt, system=None, output_schema=None):
    try:
        url, headers, payload = build_claude_prompt(prompt, system, output_schema)
        response = send_claude_request(url, headers, payload)
        # Sonnet 5.5: USD 2 / 10 por millón de tokens (ver docs/decision_arquitectura.md)
        cost = calculate_cost(response, input_pricing=2 / 1_000_000, output_pricing=10 / 1_000_000)
        print (f"Costo calculado: {cost}")
        text = extract_claude_response(response)
        return (parse_json_response(text) if output_schema else text), cost
    except Exception as e:
        print(f"unexpected error: {e}")
        raise

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

    pregunta = "¿Cuál es la capital legislativa de Sudafrica?"
    schema = {
        "type": "object",
        "properties": {"rta": {"type": "string"}, "fuente": {"type": "string"}},
        "required": ["rta", "fuente"],
        "additionalProperties": False,
    }
    casos = [
        ("Da respuestas de una sola palabra", None),
        ("", None),
        ("Utiliza solo formato JSON sin markdown, claves: rta:str y fuente:str", None),
        ("Da respuestas de una sola palabra", schema),
    ]
    for system, output_schema in casos:
        # un caso fallido no corta los siguientes
        try:
            print(f"Respuesta: {get_claude_response(pregunta, system, output_schema)[0]}")
        except (ValueError, RuntimeError) as e:
            print(f"Error: {e}")

    latency_test(10)