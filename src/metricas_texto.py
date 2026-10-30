"""Métricas de precisión, cobertura y F1 sobre tokens (Bloques 16 y 17)."""
from collections import Counter

from pipeline_texto import procesar_texto


def contar_coincidencias(tokens_referencia: list[str], tokens_respuesta: list[str]) -> int:
    """Cuenta los tokens compartidos respetando las repeticiones de cada lado."""
    for tokens in (tokens_referencia, tokens_respuesta):
        # Counter sobre un str contaría caracteres sin fallar
        if not isinstance(tokens, list):
            raise TypeError(f"contar_coincidencias espera list[str], no {type(tokens).__name__}")
    return sum((Counter(tokens_referencia) & Counter(tokens_respuesta)).values())


def calcular_precision_cobertura_f1(tokens_referencia: list[str], tokens_respuesta: list[str]) -> dict[str, float]:
    """Calcula precisión, cobertura y F1; con una lista vacía devuelve ceros."""
    coincidencias = contar_coincidencias(tokens_referencia, tokens_respuesta)
    if not tokens_referencia or not tokens_respuesta:
        return {"precision": 0.0, "cobertura": 0.0, "f1": 0.0}
    precision = coincidencias / len(tokens_respuesta)
    cobertura = coincidencias / len(tokens_referencia)
    f1 = 2 * precision * cobertura / (precision + cobertura) if precision + cobertura else 0.0
    return {"precision": precision, "cobertura": cobertura, "f1": f1}


def calcular_metricas(texto_referencia: str, texto_respuesta: str) -> dict[str, float]:
    """Procesa ambos textos y calcula precisión, cobertura y F1."""
    return calcular_precision_cobertura_f1(procesar_texto(texto_referencia), procesar_texto(texto_respuesta))


def resumir_metricas(resultados: list[dict]) -> dict[str, float]:
    """Promedia precisión, cobertura y F1; con lista vacía devuelve ceros."""
    if not resultados:
        return {"precision_promedio": 0.0, "cobertura_promedio": 0.0, "f1_promedio": 0.0}
    return {
        f"{clave}_promedio": sum(r[clave] for r in resultados) / len(resultados)
        for clave in ("precision", "cobertura", "f1")
    }


def ordenar_peores_primero(resultados: list[dict]) -> list[dict]:
    """Devuelve una lista nueva ordenada por F1 ascendente."""
    return sorted(resultados, key=lambda r: r["f1"])


def seleccionar_muestra_revision(resultados: list[dict], n: int = 5) -> list[dict]:
    """Toma los n peores y los n mejores por F1, sin duplicados y en orden ascendente."""
    ordenados = ordenar_peores_primero(resultados)
    return ordenados[:n] + ordenados[max(n, len(ordenados) - n):]
