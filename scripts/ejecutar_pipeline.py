"""Pipeline de texto sobre salidas REALES del cliente de Claude (Bloque 15).

Envía cada consulta de JSON/casos_pipeline.json a get_claude_response (llamadas
REALES a la API, con costo), compara la respuesta con la referencia antes y
después de procesar el texto y escribe el registro en docs/registro_pipeline.md.

Cómo correrlo (desde la raíz del proyecto):

    python scripts/ejecutar_pipeline.py --confirmar-api-real
"""
import argparse
import datetime
import json
import os
import sys
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pipeline_texto import comparar_antes_despues


def cargar_casos(ruta: str) -> list[dict]:
    """Lee la lista de casos (consulta y referencia) desde un JSON UTF-8."""
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def ejecutar_casos(casos: list[dict], obtener_respuesta: Callable[[str], tuple], tope_usd: float) -> dict:
    """Consulta cada caso, compara con la referencia y corta al llegar al tope de gasto."""
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
            **comparar_antes_despues(caso["referencia"], texto),
            "costo": costo,
        })
    return {
        "resultados": resultados,
        "costo_total": costo_total,
        "cortado_por_tope": cortado_por_tope,
        "errores": errores,
    }


def formatear_registro(info: dict, fecha: str) -> str:
    """Arma el registro en markdown con los porcentajes de cada caso."""
    lineas = [
        "# Registro del pipeline de texto (Bloque 15)",
        "",
        "> **Aviso:** las respuestas de este registro vienen de llamadas **REALES** a la API de Claude, con costo.",
        "",
        f"- Fecha: {fecha}",
        "",
    ]
    for n, resultado in enumerate(info["resultados"], start=1):
        respuesta = resultado["respuesta"].replace("\r\n", " ").replace("\n", " ")
        lineas += [
            f"## Caso {n}",
            "",
            f"- Consulta: {resultado['consulta']}",
            f"- Referencia: {resultado['referencia']}",
            f"- Respuesta: {respuesta}",
            f"- [Caso {n}] Coincidencia texto crudo: {resultado['pct_crudo']:.1%}",
            f"- [Caso {n}] Coincidencia texto procesado: {resultado['pct_procesado']:.1%}",
            f"- Costo del caso: ${resultado['costo']:.5f}",
            "",
        ]
    lineas += [f"**Costo total:** ${info['costo_total']:.5f}", "", "## Errores", ""]
    if info["errores"]:
        lineas += [f"- {error['consulta']}: {error['error']}" for error in info["errores"]]
    else:
        lineas.append("Ninguno")
    lineas.append("")
    if info["cortado_por_tope"]:
        lineas += ["**La ejecución se cortó al alcanzar el tope de gasto.**", ""]
    lineas += [
        "## Límites del método",
        "",
        "- La métrica no distingue el orden de las palabras: \"el perro persiguió al gato\" y \"el gato persiguió al perro\" coinciden al 100 %.",
        "- La métrica no distingue las repeticiones: compara conjuntos de palabras distintas.",
        "- Los signos de apertura (¿ ¡) se eliminan junto con el resto de la puntuación.",
        "- Un decimal pierde el punto: 9.0 pasa a 90.",
        "",
    ]
    return "\n".join(lineas)


def main(argv: list[str] | None = None) -> int:
    """Ejecuta el pipeline contra la API real solo con --confirmar-api-real."""
    parser = argparse.ArgumentParser(description="Pipeline de texto sobre respuestas REALES de Claude (con costo).")
    parser.add_argument("--casos", default="JSON/casos_pipeline.json")
    parser.add_argument("--salida", default="docs/registro_pipeline.md")
    parser.add_argument("--tope-usd", type=float, default=0.05)
    parser.add_argument("--confirmar-api-real", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirmar_api_real:
        print(
            "Este script hace llamadas REALES a la API de Claude, con costo. "
            "Para ejecutarlo pasa --confirmar-api-real.",
            file=sys.stderr,
        )
        return 2

    # import diferido: sin el flag no se carga el cliente ni la API key
    from api_claude_refactor import get_claude_response

    obtener_respuesta = lambda consulta: get_claude_response(consulta, system="Responde en una sola frase.")
    info = ejecutar_casos(cargar_casos(args.casos), obtener_respuesta, args.tope_usd)
    for n, resultado in enumerate(info["resultados"], start=1):
        print(f"[Caso {n}] Coincidencia texto crudo: {resultado['pct_crudo']:.1%}")
        print(f"[Caso {n}] Coincidencia texto procesado: {resultado['pct_procesado']:.1%}")
    print(f"Costo total: ${info['costo_total']:.5f}")
    with open(args.salida, "w", encoding="utf-8") as archivo:
        archivo.write(formatear_registro(info, datetime.date.today().isoformat()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
