import time
import random
import threading

import requests
import os
import json
import re
from typing import Callable, Any
from dotenv import load_dotenv

load_dotenv()

# Sonnet 5.5: USD 2 / 10 por millón de tokens (ver docs/decision_arquitectura.md)
INPUT_PRICING = 2 / 1_000_000
OUTPUT_PRICING = 10 / 1_000_000
costo_total_sesion = 0.0
_lock_costo = threading.Lock()

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
        response = requests.post(url, headers=headers, data=payload, timeout=10)
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        raise TimeoutError("No se pudo contactar al proveedor. Puede ser momentáneo.")
    if response.status_code in (401, 403):
        raise ValueError("Credencial inválida o sin permiso. Revisa la API key de tu proveedor en .env.")
    if response.status_code == 429 or response.status_code >= 500:
        raise TimeoutError("El proveedor no pudo procesar la petición. Hay que esperar y reintentar.")
    if not response.ok:
        raise RuntimeError(f"Ocurrio un error de conexión: {response.status_code}")
    try:
        return response.json()
    except requests.exceptions.JSONDecodeError as e:
        raise RuntimeError(f"Error decodificando la respuesta JSON o viene vacía: {e}")

def with_retry(func: Callable[[], Any], attempts: int = 3, base_wait: float = 2, max_wait: float = 30) -> Any:
    # solo TimeoutError (red, 429, 5xx) es transitorio; ValueError/RuntimeError no mejoran reintentando
    if attempts < 1:
        raise ValueError("attempts debe ser al menos 1")
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except TimeoutError:
            if attempt == attempts:
                raise
            wait = min(base_wait * 2 ** (attempt - 1), max_wait)
            wait_with_jitter = wait + random.uniform(0, wait * 0.1)
            print(f"Intento {attempt} de {attempts} falló. Reintentando en {wait_with_jitter:.1f}s...")
            time.sleep(wait_with_jitter)

def calculate_cost(response, input_pricing, output_pricing):
    try:
        use = response["usage"]
        input_tokens = use["input_tokens"]
        output_tokens = use["output_tokens"]
        return (input_tokens * input_pricing) + (output_tokens * output_pricing)
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Error calculando el costo: {e}")

def registrar_costo(response: dict) -> float:
    global costo_total_sesion
    cost = calculate_cost(response, INPUT_PRICING, OUTPUT_PRICING)
    total_tokens = response["usage"]["input_tokens"] + response["usage"]["output_tokens"]
    # el print va fuera del lock para no retener a otros hilos mientras se escribe en consola
    with _lock_costo:
        costo_total_sesion += cost
        mensaje = f"Operación: {total_tokens} tokens, costo estimado: ${cost:.5f} (acumulado: ${costo_total_sesion:.5f})"
    print(mensaje)
    return cost

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
        try:
            response = with_retry(lambda: send_claude_request(url, headers, payload))
        except TimeoutError as e:
            raise RuntimeError("El servicio no respondió tras varios intentos. Intenta más tarde.") from e
        cost = registrar_costo(response)
        text = extract_claude_response(response)
        return (parse_json_response(text) if output_schema else text), cost
    except (ValueError, RuntimeError):
        raise
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
        registrar_costo(data)
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