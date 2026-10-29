"""Ensayo SIMULADO de rate limit (Bloque 13).

No es un 429 real del proveedor: requests.post y time.sleep se reemplazan con
mocks (unittest.mock), así que no se usa la red, ni la API real, ni la API key.
Envía 10 peticiones a get_claude_response; en la petición 6 el falso
requests.post devuelve dos 429 seguidos y luego un 200, para ver el reintento
con backoff exponencial y el costo acumulado de la sesión.

Cómo correrlo (desde la raíz del proyecto):

    .venv/bin/python scripts/ensayo_rate_limit_simulado.py
"""
import os
import sys
from unittest.mock import patch, Mock

# clave falsa: aunque exista .env, load_dotenv no sobrescribe una variable ya definida
os.environ["CLAUDE_API_KEY"] = "clave-falsa-de-ensayo"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import api_claude_refactor

PETICIONES = 10
PETICION_CON_429 = 6

esperas = []
llamadas_post = 0
llamadas_en_peticion = 0
peticion_actual = 0
primer_429 = None


def respuesta_ok(i):
    respuesta = Mock(status_code=200, ok=True)
    respuesta.json.return_value = {
        "content": [{"type": "text", "text": f"Respuesta simulada número {i}"}],
        "usage": {"input_tokens": 20 + i, "output_tokens": 40 + 2 * i},
    }
    return respuesta


def post_falso(url, headers=None, data=None, timeout=None):
    global llamadas_post, llamadas_en_peticion, primer_429
    llamadas_post += 1
    llamadas_en_peticion += 1
    if peticion_actual == PETICION_CON_429 and llamadas_en_peticion <= 2:
        if primer_429 is None:
            primer_429 = peticion_actual
        return Mock(status_code=429, ok=False)
    return respuesta_ok(peticion_actual)


def sleep_falso(segundos):
    # no duerme: solo registra la espera pedida por with_retry
    esperas.append(segundos)


# el patch cubre toda la ejecución: ningún requests.post real llega a ejecutarse
with patch("api_claude_refactor.requests.post", side_effect=post_falso), \
        patch("api_claude_refactor.time.sleep", side_effect=sleep_falso):
    for i in range(1, PETICIONES + 1):
        peticion_actual = i
        llamadas_en_peticion = 0
        try:
            texto, _ = api_claude_refactor.get_claude_response(f"Pregunta de prueba número {i}")
            print(f"Respuesta {i}: {texto[:40]}...")
        except RuntimeError as error:
            print(f"Petición {i} falló definitivamente: {error}")

print(f"Primer 429 en la petición: {primer_429}")
print(f"Esperas pedidas a time.sleep: {', '.join(f'{e:.1f}s' for e in esperas)}")
print(f"Llamadas totales a requests.post: {llamadas_post}")
print(f"Costo total del ensayo: ${api_claude_refactor.costo_total_sesion:.5f}")
