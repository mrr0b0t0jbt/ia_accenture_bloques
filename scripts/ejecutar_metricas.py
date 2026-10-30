"""Precisión, cobertura y F1 sobre salidas REALES del cliente de Claude (Bloques 16 y 17).

Envía cada consulta de JSON/casos_metricas.json a get_claude_response (llamadas
REALES a la API, con costo), compara la respuesta con la referencia y escribe
los resultados en docs/resultados_metricas.json después de cada caso.

Cómo correrlo (desde la raíz del proyecto):

    python scripts/ejecutar_metricas.py --confirmar-api-real
"""
import argparse
import json
import os
import sys
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from metricas_texto import calcular_metricas, resumir_metricas

N_MAX_CASOS = 25
TOPE_USD_MAXIMO = 0.05


def cargar_casos(ruta: str) -> list[dict]:
    """Lee la lista de casos (consulta y referencia) desde un JSON UTF-8."""
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def guardar_resultados(resultados: list[dict], ruta_salida: str) -> None:
    """Reescribe el archivo de salida completo con los resultados."""
    with open(ruta_salida, "w", encoding="utf-8") as archivo:
        json.dump(resultados, archivo, indent=2, ensure_ascii=False)


def ejecutar_casos(
    casos: list[dict], obtener_respuesta: Callable[[str], tuple], tope_usd: float, ruta_salida: str
) -> dict:
    """Consulta cada caso, calcula sus métricas y corta al llegar al tope de gasto."""
    resultados = []
    errores = []
    costo_total = 0.0
    cortado_por_tope = False
    for caso in casos:
        if costo_total >= tope_usd:
            cortado_por_tope = True
            break
        try:
            texto, costo = obtener_respuesta(caso["consulta"])
        except (RuntimeError, ValueError) as error:
            # un caso fallido no corta los siguientes
            errores.append({"consulta": caso["consulta"], "error": str(error)})
            continue
        costo_total += costo
        resultados.append({
            "consulta": caso["consulta"],
            "referencia": caso["referencia"],
            "respuesta": texto,
            **calcular_metricas(caso["referencia"], texto),
            "costo": costo,
        })
        # guardado incremental: si algo falla a mitad no se pierde lo ya pagado
        guardar_resultados(resultados, ruta_salida)
    return {
        "resultados": resultados,
        "errores": errores,
        "costo_total": costo_total,
        "cortado_por_tope": cortado_por_tope,
    }


def main(argv: list[str] | None = None) -> int:
    """Calcula las métricas contra la API real solo con --confirmar-api-real."""
    parser = argparse.ArgumentParser(description="Métricas de texto sobre respuestas REALES de Claude (con costo).")
    parser.add_argument("--casos", default="JSON/casos_metricas.json")
    parser.add_argument("--salida", default="docs/resultados_metricas.json")
    parser.add_argument("--tope-usd", type=float, default=TOPE_USD_MAXIMO)
    parser.add_argument("--confirmar-api-real", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirmar_api_real:
        print(
            "Este script hace llamadas REALES a la API de Claude, con costo. "
            "Para ejecutarlo pasa --confirmar-api-real.",
            file=sys.stderr,
        )
        return 2
    if not args.salida:
        print("Error: --salida no puede estar vacía.", file=sys.stderr)
        return 1
    directorio = os.path.dirname(os.path.abspath(args.salida))
    if not os.path.isdir(directorio):
        print(f"Error: el directorio de --salida no existe: {directorio}", file=sys.stderr)
        return 1
    if not 0 < args.tope_usd <= TOPE_USD_MAXIMO:
        print(f"Error: --tope-usd debe ser mayor que 0 y como máximo {TOPE_USD_MAXIMO}.", file=sys.stderr)
        return 1
    casos = cargar_casos(args.casos)
    if len(casos) > N_MAX_CASOS:
        print(f"Error: hay {len(casos)} casos; el máximo es {N_MAX_CASOS}.", file=sys.stderr)
        return 1

    # import diferido: sin el flag no se carga el cliente ni la API key
    from api_claude_refactor import get_claude_response

    obtener_respuesta = lambda consulta: get_claude_response(consulta, system="Responde en una sola frase.")
    info = ejecutar_casos(casos, obtener_respuesta, args.tope_usd, args.salida)
    for n, resultado in enumerate(info["resultados"], start=1):
        print(f"[Caso {n}] {resultado['consulta']}")
        print(
            f"  precisión {resultado['precision']:.2f} | cobertura {resultado['cobertura']:.2f}"
            f" | F1 {resultado['f1']:.2f}"
        )
    for error in info["errores"]:
        print(f"Error en «{error['consulta']}»: {error['error']}")
    if info["cortado_por_tope"]:
        print("La ejecución se cortó al alcanzar el tope de gasto.")
    print(f"Costo total: ${info['costo_total']:.5f}")
    resumen = resumir_metricas(info["resultados"])
    print(
        f"Promedios: precisión {resumen['precision_promedio']:.2f} | "
        f"cobertura {resumen['cobertura_promedio']:.2f} | F1 {resumen['f1_promedio']:.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
