"""Métricas de la recuperación por CHUNK (Bloque 27): precisión, exhaustividad y F1 en el top-k.

Un chunk se identifica con su id documento_n (por ejemplo politica_parcheo_0). La comparación es EXACTA,
sin normalizar mayúsculas ni espacios: una diferencia mínima ("Doc0" o "doc0 ") rompe la igualdad.

Desvío respecto de la clase: los promedios de precisión, exhaustividad y F1 se calculan SOLO sobre los casos
respondibles. En una pregunta sin respuesta en el corpus (ids_relevantes vacío) la exhaustividad no está
definida (None) y la precisión sería siempre 0 si se recupera algo, lo que arrastraría el promedio; esos casos
se reportan aparte (ids recuperados y distancia mínima).

Límites: la métrica ignora el orden dentro del top-k (un acierto en el rango 1 vale igual que en el k) y la
exhaustividad sin chunks relevantes es None, no 0.
"""
from collections.abc import Callable

K_MAXIMO = 5
MAX_CASOS = 20


def _es_entero(valor) -> bool:
    """Indica si el valor es un entero (un booleano no cuenta)."""
    return isinstance(valor, int) and not isinstance(valor, bool)


def _validar_ids(ids_recuperados, ids_relevantes) -> None:
    """Rechaza tipos equivocados antes de calcular: 'in' sobre un texto revisaría subcadenas."""
    if not isinstance(ids_relevantes, (set, frozenset)):
        raise TypeError("ids_relevantes debe ser un set o frozenset de textos.")
    if not all(isinstance(i, str) for i in ids_relevantes):
        raise TypeError("ids_relevantes solo puede tener textos.")
    if not all(i.strip() for i in ids_relevantes):
        raise ValueError("ids_relevantes no puede tener textos vacíos.")
    if not isinstance(ids_recuperados, list):
        raise TypeError("ids_recuperados debe ser una lista de textos.")
    if not all(isinstance(i, str) for i in ids_recuperados):
        raise TypeError("ids_recuperados solo puede tener textos.")
    if len(set(ids_recuperados)) != len(ids_recuperados):
        raise ValueError("ids_recuperados tiene un id repetido; el retriever nunca repite ids.")


def precision_en_k(ids_recuperados: list[str], ids_relevantes: set[str]) -> float:
    """Fracción de los ids recuperados que son relevantes; 0.0 si no se recuperó ninguno."""
    _validar_ids(ids_recuperados, ids_relevantes)
    if not ids_recuperados:
        return 0.0
    return sum(1 for i in ids_recuperados if i in ids_relevantes) / len(ids_recuperados)


def exhaustividad_en_k(ids_recuperados: list[str], ids_relevantes: set[str]) -> float | None:
    """Fracción de los relevantes que se recuperaron; None si no hay ids relevantes."""
    _validar_ids(ids_recuperados, ids_relevantes)
    if not ids_relevantes:
        return None
    return sum(1 for i in ids_recuperados if i in ids_relevantes) / len(ids_relevantes)


def f1(precision: float, exhaustividad: float | None) -> float | None:
    """Media armónica de precisión y exhaustividad; 0.0 si ambas son 0 y None si la exhaustividad es None."""
    if exhaustividad is None:
        return None
    if precision + exhaustividad == 0:
        return 0.0
    return 2 * precision * exhaustividad / (precision + exhaustividad)


def promediar(valores: list[float | None]) -> float:
    """Promedio que ignora los None; 0.0 si no queda ningún valor."""
    validos = [v for v in valores if v is not None]
    return sum(validos) / len(validos) if validos else 0.0


def _validar_k(k) -> None:
    if not _es_entero(k) or not 1 <= k <= K_MAXIMO:
        raise ValueError(f"k debe ser un entero entre 1 y {K_MAXIMO}.")


def evaluar_caso(ids_recuperados: list[str], ids_relevantes: set[str], ks=range(1, K_MAXIMO + 1)) -> dict[int, dict]:
    """Calcula precisión, exhaustividad y F1 del top-k para cada k pedido; las claves son los k."""
    ks = list(ks)
    if not ks:
        raise ValueError("ks no puede estar vacío.")
    for k in ks:
        _validar_k(k)
    _validar_ids(ids_recuperados, ids_relevantes)
    resultado = {}
    for k in ks:
        precision = precision_en_k(ids_recuperados[:k], ids_relevantes)
        exhaustividad = exhaustividad_en_k(ids_recuperados[:k], ids_relevantes)
        resultado[k] = {"precision": precision, "exhaustividad": exhaustividad, "f1": f1(precision, exhaustividad)}
    return resultado


def _problema_de_caso(n: int, caso, vistos: set, ids_validos) -> str | None:
    """Primer problema del caso n de la lista, o None."""
    if not isinstance(caso, dict):
        return f"casos[{n}] no es un objeto."
    for campo in ("id", "pregunta"):
        if not isinstance(caso.get(campo), str) or not caso[campo].strip():
            return f"casos[{n}] no tiene el campo {campo} como texto no vacío."
    if caso["id"] in vistos:
        return f"casos[{n}] repite el id {caso['id']!r}."
    vistos.add(caso["id"])
    relevantes = caso.get("ids_relevantes")
    if not isinstance(relevantes, list) or not all(isinstance(i, str) and i.strip() for i in relevantes):
        return f"casos[{n}] no tiene ids_relevantes como lista de textos no vacíos."
    if len(set(relevantes)) != len(relevantes):
        return f"casos[{n}] repite un id dentro de ids_relevantes."
    if not isinstance(caso.get("respondible"), bool):
        return f"casos[{n}] no tiene respondible como booleano."
    if caso["respondible"] and not relevantes:
        return f"casos[{n}] es respondible y no tiene ids_relevantes."
    if not caso["respondible"] and relevantes:
        return f"casos[{n}] no es respondible y tiene ids_relevantes."
    if "nota" in caso and not isinstance(caso["nota"], str):
        return f"casos[{n}] tiene una nota que no es texto."
    if ids_validos is not None:
        for id_chunk in relevantes:
            if id_chunk not in ids_validos:
                return f"el caso {caso['id']!r} etiqueta el id {id_chunk!r}, que no existe en el corpus."
    return None


def validar_casos_evaluacion(casos, ids_validos=None) -> str | None:
    """Devuelve el primer problema de estructura o coherencia de los casos, o None si no hay ninguno."""
    if not isinstance(casos, list) or not 1 <= len(casos) <= MAX_CASOS:
        return f"el archivo de casos debe ser una lista de 1 a {MAX_CASOS} casos."
    vistos: set = set()
    for n, caso in enumerate(casos):
        problema = _problema_de_caso(n, caso, vistos, ids_validos)
        if problema:
            return problema
    return None


def evaluar_recuperacion(recuperar_ids: Callable[[str, int], list[tuple[str, float]]], casos: list[dict], k: int = 3) -> dict:
    """Evalúa todos los casos con UNA recuperación de K_MAXIMO por caso; promedia solo los respondibles."""
    _validar_k(k)
    problema = validar_casos_evaluacion(casos)
    if problema:
        raise ValueError(problema)
    por_caso, sin_respuesta = [], []
    for caso in casos:
        pares = recuperar_ids(caso["pregunta"], K_MAXIMO)
        ids = [id_chunk for id_chunk, _ in pares]
        metricas = evaluar_caso(ids, set(caso["ids_relevantes"])) if caso["respondible"] else None
        por_caso.append(
            {
                "id": caso["id"],
                "pregunta": caso["pregunta"],
                "respondible": caso["respondible"],
                "ids_relevantes": sorted(caso["ids_relevantes"]),
                "recuperados": [{"rango": rango, "id": i, "distancia": d} for rango, (i, d) in enumerate(pares, start=1)],
                "metricas": metricas,
            }
        )
        if not caso["respondible"]:
            sin_respuesta.append({"id": caso["id"], "ids_recuperados": ids[:k], "distancia_minima": pares[0][1] if pares else None})
    respondibles = [c["metricas"] for c in por_caso if c["metricas"] is not None]
    promedios = {
        n: {metrica: promediar([m[n][metrica] for m in respondibles]) for metrica in ("precision", "exhaustividad", "f1")}
        for n in range(1, K_MAXIMO + 1)
    }
    return {"por_caso": por_caso, "promedios_respondibles": promedios, "sin_respuesta": sin_respuesta}
