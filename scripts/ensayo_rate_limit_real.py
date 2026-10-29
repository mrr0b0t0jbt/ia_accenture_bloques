"""Ensayo REAL de rate limit contra la API de Claude (con costo).

Envía N peticiones seguidas (máximo 30) a get_claude_response pidiendo una
respuesta de una sola palabra, para intentar provocar un 429 real gastando casi
nada. La variante concurrente (--concurrencia K, máximo 20) envía hasta 100
peticiones en olas de K peticiones simultáneas, para superar el límite por
minuto de la cuenta. requests.post y time.sleep se envuelven con espías que
solo observan: la petición real se envía igual y la respuesta vuelve intacta.

Cómo correrlo (desde la raíz del proyecto):

    python scripts/ensayo_rate_limit_real.py --confirmar-api-real
    python scripts/ensayo_rate_limit_real.py --confirmar-api-real --concurrencia 10 --n 60
"""
import argparse
import concurrent.futures
import datetime
import os
import sys
import threading
import time
from typing import Callable
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

MAX_PETICIONES = 30
MAX_PETICIONES_CONCURRENTE = 100
MAX_CONCURRENCIA = 20


def ejecutar_ensayo(n: int, tope_usd: float, enviar: Callable | None = None, concurrencia: int = 1) -> dict:
    """Envía n peticiones en olas de `concurrencia` y registra los 429, las esperas y el costo.

    Con concurrencia == 1 las peticiones van una tras otra en el hilo llamador.
    El tope de gasto se comprueba antes de cada ola, así que el sobregiro máximo
    posible es una ola completa (concurrencia peticiones).
    """
    if concurrencia < 1 or concurrencia > MAX_CONCURRENCIA:
        raise ValueError(f"concurrencia debe estar entre 1 y {MAX_CONCURRENCIA}, no {concurrencia}")
    maximo = MAX_PETICIONES if concurrencia == 1 else MAX_PETICIONES_CONCURRENTE
    if n < 1 or n > maximo:
        raise ValueError(f"n debe estar entre 1 y {maximo}, no {n}")

    import api_claude_refactor

    if enviar is None:
        enviar = api_claude_refactor.get_claude_response
    post_original = api_claude_refactor.requests.post
    sleep_original = api_claude_refactor.time.sleep
    local = threading.local()
    lock = threading.Lock()
    estado = {"llamadas_http": 0, "primer_429": None, "total_429": 0, "fallos_definitivos": 0, "costo_total": 0.0}
    esperas: list[float] = []

    # el lock solo protege los contadores: la llamada real va fuera para no serializar las peticiones
    def post_espia(*args, **kwargs):
        with lock:
            estado["llamadas_http"] += 1
        response = post_original(*args, **kwargs)
        if response.status_code == 429:
            with lock:
                estado["total_429"] += 1
                if estado["primer_429"] is None or local.peticion < estado["primer_429"]:
                    estado["primer_429"] = local.peticion
        return response

    def sleep_espia(segundos):
        with lock:
            esperas.append(segundos)
        return sleep_original(segundos)

    def ejecutar_una(i):
        local.peticion = i
        try:
            _, costo = enviar(f"Pregunta de prueba número {i}", system="Responde solo con la palabra OK")
        except RuntimeError:
            # reintentos agotados: se registra y el ensayo sigue
            with lock:
                estado["fallos_definitivos"] += 1
            return
        with lock:
            estado["costo_total"] += costo

    enviadas = 0
    olas = 0
    cortado_por_tope = False
    inicio = time.monotonic()
    with patch.object(api_claude_refactor.requests, "post", post_espia), \
            patch.object(api_claude_refactor.time, "sleep", sleep_espia):
        for primera in range(1, n + 1, concurrencia):
            if estado["costo_total"] >= tope_usd:
                cortado_por_tope = True
                break
            indices = range(primera, min(primera + concurrencia, n + 1))
            if concurrencia == 1:
                ejecutar_una(primera)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(indices)) as pool:
                    for futuro in [pool.submit(ejecutar_una, i) for i in indices]:
                        futuro.result()
            enviadas = indices[-1]
            olas += 1
    return {
        "peticiones_enviadas": enviadas,
        "llamadas_http": estado["llamadas_http"],
        "primer_429": estado["primer_429"],
        "total_429": estado["total_429"],
        "esperas": esperas,
        "fallos_definitivos": estado["fallos_definitivos"],
        "costo_total": estado["costo_total"],
        "cortado_por_tope": cortado_por_tope,
        "duracion_s": time.monotonic() - inicio,
        "concurrencia": concurrencia,
        "olas": olas,
    }


def formatear_registro(info: dict, fecha: str) -> str:
    """Arma el registro en markdown con la plantilla de docs/registro_rate_limit.md."""
    enviadas = info["peticiones_enviadas"]
    concurrente = info.get("concurrencia", 1) > 1
    if info["primer_429"] is None:
        linea_429 = "no apareció ningún 429"
        resultado = f"no se alcanzó ningún 429 en {enviadas} peticiones"
    else:
        linea_429 = f"la número {info['primer_429']}"
        resultado = f"el primer 429 apareció en la petición {info['primer_429']}; hubo {info['total_429']} respuestas 429 en total"
    esperas = ", ".join(f"{e:.1f}s" for e in info["esperas"]) or "ninguna"
    lineas = [
        "# Registro del ensayo REAL de rate limit",
        "",
        "> **Aviso:** este es un ensayo **REAL** contra la API de Claude: usó la red y la API key, y tuvo costo.",
        "",
    ]
    if concurrente:
        lineas += [
            "> **Aviso (concurrencia):** esto es una ráfaga concurrente deliberada para superar el límite por minuto de la cuenta; puede provocar 429 y reintentos.",
            "",
        ]
    lineas += [
        "## Registro del ensayo de rate limit",
        "",
        f"- Fecha: {fecha}",
        f"- Peticiones enviadas: {enviadas}",
    ]
    if concurrente:
        lineas.append(f"- Concurrencia: {info['concurrencia']} (olas: {info.get('olas')})")
    lineas += [
        f"- Petición donde apareció el primer 429: {linea_429}",
        f"- Esperas observadas: {esperas}",
        f"- Costo total del ensayo: ${info['costo_total']:.5f}",
        f"- Resultado: {resultado}",
        f"- Llamadas HTTP: {info['llamadas_http']}",
        f"- Respuestas 429 totales: {info['total_429']}",
        f"- Fallos definitivos: {info['fallos_definitivos']}",
        f"- Duración: {info['duracion_s']:.1f}s",
    ]
    if info["cortado_por_tope"]:
        lineas.append("- El ensayo se cortó al alcanzar el tope de gasto.")
    lineas += [
        "",
        "## Límites del ensayo",
        "",
        "- Puede NO aparecer ningún 429: depende de la cuota real de la cuenta en ese momento.",
        "- No mide latencia fina: la duración es el total del ensayo, incluidas las esperas.",
        "",
        "## Cómo correrlo",
        "",
        "```bash",
        "python scripts/ensayo_rate_limit_real.py --confirmar-api-real",
        "```",
        "",
    ]
    return "\n".join(lineas)


def main(argv: list[str] | None = None) -> int:
    """Ejecuta el ensayo real solo con --confirmar-api-real."""
    parser = argparse.ArgumentParser(description="Ensayo REAL de rate limit contra la API de Claude (con costo).")
    parser.add_argument("--n", type=int, default=MAX_PETICIONES)
    parser.add_argument("--tope-usd", type=float, default=0.05)
    parser.add_argument("--concurrencia", type=int, default=1)
    parser.add_argument("--salida", default=None)
    parser.add_argument("--confirmar-api-real", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirmar_api_real:
        print(
            "Este es un ensayo REAL contra la API de Claude, con costo. "
            "Para ejecutarlo pasa --confirmar-api-real.",
            file=sys.stderr,
        )
        return 2
    if args.salida is None:
        if args.concurrencia > 1:
            args.salida = "docs/registro_rate_limit_real_concurrente.md"
        else:
            args.salida = "docs/registro_rate_limit_real.md"
    directorio = os.path.dirname(os.path.abspath(args.salida))
    if not os.path.isdir(directorio):
        print(f"Error: el directorio de --salida no existe: {directorio}", file=sys.stderr)
        return 1
    try:
        info = ejecutar_ensayo(args.n, args.tope_usd, concurrencia=args.concurrencia)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Peticiones enviadas: {info['peticiones_enviadas']}")
    print(f"Primer 429 en la petición: {info['primer_429']}")
    print(f"Esperas pedidas a time.sleep: {', '.join(f'{e:.1f}s' for e in info['esperas']) or 'ninguna'}")
    print(f"Llamadas HTTP: {info['llamadas_http']}")
    print(f"Costo total del ensayo: ${info['costo_total']:.5f}")
    with open(args.salida, "w", encoding="utf-8") as archivo:
        archivo.write(formatear_registro(info, datetime.date.today().isoformat()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
