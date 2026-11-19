"""Mide el formato y la exactitud de la respuesta cruda del clasificador del router (Bloque 30, Unidad 2.6).

La regla de la clase para problema_de_formato_o_tono: en una muestra de respuestas reales, el 15 % o más incumple el formato.
Aquí "formato estricto" es que la respuesta cruda sea exactamente una categoría; "formato efectivo" es que quede válida tras
la normalización del router (enrutamiento._normalizar_categoria). No llama a ninguna API; la respuesta del modelo es dato.
"""
from api_claude_refactor import INPUT_PRICING, OUTPUT_PRICING
from enrutamiento import CATEGORIAS, _normalizar_categoria

UMBRAL_FORMATO = 0.15
MAX_CARACTERES_RESPUESTA = 200


def clasificar_respuesta(cruda, categoria_esperada: str) -> dict:
    """Evalúa una respuesta cruda: saneada y recortada, si es exacta, su forma normalizada, si es válida y si acierta."""
    if not isinstance(cruda, str):
        raise ValueError("la respuesta cruda no es texto")
    normalizada = _normalizar_categoria(cruda)  # registra un warning si la categoría es inesperada: es lo esperado
    return {
        "respuesta": "".join(c if c.isprintable() else " " for c in cruda)[:MAX_CARACTERES_RESPUESTA],
        "exacta": cruda in CATEGORIAS,
        "normalizada": normalizada,
        "valida": normalizada in CATEGORIAS,
        "correcta": normalizada == categoria_esperada,
    }


def costo_de_llamada(tokens_entrada: int, tokens_salida: int) -> float:
    """Costo estimado en USD con los precios por token del proyecto."""
    return tokens_entrada * INPUT_PRICING + tokens_salida * OUTPUT_PRICING


def _proporcion(fallos: int, evaluadas: int) -> float | None:
    return round(fallos / evaluadas, 4) if evaluadas else None


def _supera_umbral(fallos: int, evaluadas: int) -> bool | None:
    # enteros: así 3 de 20 (justo 15 %) no depende del punto flotante
    return fallos * 100 >= round(UMBRAL_FORMATO * 100) * evaluadas if evaluadas else None


def resumir(resultados: list[dict]) -> dict:
    """Resume los resultados con error None y tokens; cada uno trae categoria_esperada, normalizada, exacta, valida y correcta."""
    evaluados = [r for r in resultados if r["error"] is None and r["tokens"] is not None]
    n = len(evaluados)
    exactas = sum(1 for r in evaluados if r["exacta"])
    validas = sum(1 for r in evaluados if r["valida"])
    aciertos = sum(1 for r in evaluados if r["correcta"])
    por_categoria = {}
    for categoria in CATEGORIAS:
        propios = [r for r in evaluados if r["categoria_esperada"] == categoria]
        predichas = {c: sum(1 for r in propios if r["valida"] and r["normalizada"] == c) for c in CATEGORIAS}
        predichas["otra"] = sum(1 for r in propios if not r["valida"])
        por_categoria[categoria] = {"esperadas": len(propios), "aciertos": sum(1 for r in propios if r["correcta"]), "predichas": predichas}
    return {
        "llamadas": len(resultados),
        "con_error": sum(1 for r in resultados if r["error"] is not None),
        "evaluadas": n,
        "exactas": exactas,
        "validas": validas,
        "fallos_formato_estricto": n - exactas,
        "fallos_formato_efectivo": n - validas,
        "proporcion_estricta": _proporcion(n - exactas, n),
        "proporcion_efectiva": _proporcion(n - validas, n),
        "umbral": UMBRAL_FORMATO,
        "supera_umbral_estricto": _supera_umbral(n - exactas, n),
        "supera_umbral_efectivo": _supera_umbral(n - validas, n),
        "aciertos": aciertos,
        "exactitud": round(aciertos / n, 4) if n else None,
        "por_categoria": por_categoria,
    }
