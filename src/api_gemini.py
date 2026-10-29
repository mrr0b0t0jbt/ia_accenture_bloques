import time

import os
import json
from dotenv import load_dotenv
from api_claude_refactor import send_claude_request as send_gemini_request, parse_json_response, with_retry

load_dotenv()

def build_gemini_prompt(prompt, system=None, output_schema=None):
    if not prompt:
        raise ValueError("La entrada no puede estar vacía. Por favor, proporciona un prompt válido.")
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent"
    api_key = os.getenv("GEMINI_API_KEY")
    body = {
        "contents": [
            {"role": "user", "parts": [{"text": prompt}]}
        ]
    }
    if system:
        body["system_instruction"] = {"parts": [{"text": system}]}
    if output_schema:
        body["generationConfig"] = {"responseMimeType": "application/json", "responseSchema": output_schema}
    payload = json.dumps(body)
    headers = {
        'x-goog-api-key': api_key,
        'Content-Type': 'application/json'
    }
    return url, headers, payload

def calculate_gemini_cost(response, input_pricing, output_pricing):
    try:
        use = response["usageMetadata"]
        input_tokens = use["promptTokenCount"]
        output_tokens = use["candidatesTokenCount"]
        return (input_tokens * input_pricing) + (output_tokens * output_pricing)
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Error calculando el costo: {e}")

def extract_gemini_response(data):
    try:
        return data["candidates"][0]["content"]["parts"][0].get("text", "No se encontro texto en la respuesta")
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Error extrayendo la respuesta: {e}")

def get_gemini_response(prompt, system=None, output_schema=None):
    try:
        url, headers, payload = build_gemini_prompt(prompt, system, output_schema)
        try:
            response = with_retry(lambda: send_gemini_request(url, headers, payload))
        except TimeoutError as e:
            raise RuntimeError("El servicio no respondió tras varios intentos. Intenta más tarde.") from e
        # shortcut: precios de Gemini 3.1 Pro (USD 2 / 12 por millón), no de flash-lite; actualizar al confirmar precios de flash-lite
        # Pro dio 429 por cuota (sin facturación), por eso se usa flash-lite por ahora
        cost = calculate_gemini_cost(response, input_pricing=2 / 1_000_000, output_pricing=12 / 1_000_000)
        print (f"Costo calculado: {cost}")
        text = extract_gemini_response(response)
        return (parse_json_response(text) if output_schema else text), cost
    except (ValueError, RuntimeError):
        raise
    except Exception as e:
        print(f"unexpected error: {e}")
        raise

def latency_test(ciclos: int = 5):
    latencies = []
    for i in range(ciclos):
        url, headers, payload = build_gemini_prompt("Cuanto es 2 + 2? Dame solo el resultado")
        index = time.time()
        data = send_gemini_request(url, headers, payload)
        duration = time.time() - index
        print(f"Duración de la petición {i + 1}: {duration} segundos")
        latencies.append(duration)
    prom_latency = sum(latencies) / len(latencies)
    print(f"Latencia promedio después de {ciclos} ciclos: {prom_latency} segundos")

if __name__ == "__main__":
    response, cost = get_gemini_response("Confirma en una frase corta que esta conexion funciona correctamente.")
    print(f"Respuesta: {response}")
    print(f"Costo: {cost}")

    pregunta = "¿Cuál es la capital legislativa de Sudafrica?"
    schema = {
        "type": "OBJECT",
        "properties": {"rta": {"type": "STRING"}, "fuente": {"type": "STRING"}},
        "required": ["rta", "fuente"],
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
            print(f"Respuesta: {get_gemini_response(pregunta, system, output_schema)[0]}")
        except (ValueError, RuntimeError) as e:
            print(f"Error: {e}")

    latency_test(10)
