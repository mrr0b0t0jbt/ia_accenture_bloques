"""Mide la latencia por etapa y el costo estimado del flujo RAG completo (Bloque 28).

Por consulta se miden tres etapas por separado: embed_query (red, Gemini), busqueda_vectorial (local, Chroma) y
generacion (red, Claude). Se embebe UNA vez y el vector se reutiliza en la búsqueda; por eso no se usa
recuperacion.recuperar, que embebe dentro de la búsqueda.

Límites de lo que se mide:
- La latencia de red varía entre corridas: una medición por combinación (pregunta, k) es un orden de magnitud, no estadística.
- El primer llamado de una corrida incluye el establecimiento de la conexión, por eso suele ser el más lento.
- Desvíos respecto de la clase: el generador es Claude (no Gemini) y los precios del proyecto son USD por TOKEN
  (api_claude_refactor.INPUT_PRICING y OUTPUT_PRICING), no por 1000 tokens. El costo de los embeddings no se estima
  (el proyecto no conoce el precio de gemini-embedding-001): se cuentan peticiones.
- Los tokens aproximados del prompt (len // 4) son una heurística; el conteo real viene de la API.
"""
import contextlib
import datetime
import json
import time

import indexacion
import recuperacion

ETAPAS = ("embed_query", "busqueda_vectorial", "generacion")
K_MAXIMO_MEDICION = 6
MAX_CARACTERES_RESPUESTA = 1000


class TopeDePeticiones(Exception):
    """Se alcanzó el tope de consultas de embedding (su texto no puede contener 429 ni RESOURCE_EXHAUSTED)."""


@contextlib.contextmanager
def medir(tiempos: dict, etapa: str, reloj=time.perf_counter):
    """Guarda en tiempos[etapa] los segundos del bloque solo si termina sin excepción."""
    inicio = reloj()
    yield
    tiempos[etapa] = reloj() - inicio


def aprox_tokens(texto: str) -> int:
    """Aproxima los tokens de un texto como len // 4 (heurística; el conteo real viene de la API)."""
    return len(texto) // 4


def calcular_costo(tokens_entrada: int, tokens_salida: int, precio_entrada: float, precio_salida: float) -> float:
    """Costo en USD con precios por token."""
    return tokens_entrada * precio_entrada + tokens_salida * precio_salida


def _es_entero(valor) -> bool:
    return isinstance(valor, int) and not isinstance(valor, bool)


def embeber_con_reintentos(
    embeddings, pregunta: str, contador: dict, max_consultas: int, dormir=None, reloj=time.perf_counter
) -> tuple[list[float], float, int]:
    """Embebe la pregunta; devuelve (vector, segundos del intento exitoso, reintentos). Cada intento cuenta contra el tope."""
    intentos = 0

    def intento() -> tuple[list[float], float]:
        nonlocal intentos
        if contador["consultas"] >= max_consultas:
            raise TopeDePeticiones(f"se alcanzó el tope de {max_consultas} consultas de embedding; no se hicieron más peticiones.")
        contador["consultas"] += 1
        intentos += 1
        tiempos = {}
        with medir(tiempos, "embed_query", reloj):
            vector = embeddings.embed_query(pregunta)
        return vector, tiempos["embed_query"]

    vector, segundos = indexacion.con_reintentos(intento, dormir=dormir)
    return vector, segundos, intentos - 1


def _validar_consulta(pregunta, k) -> None:
    if not isinstance(pregunta, str) or not pregunta.strip():
        raise ValueError("La pregunta debe ser un texto con algo más que espacios.")
    if len(pregunta) > recuperacion.MAX_LONGITUD_PREGUNTA:
        raise ValueError(f"La pregunta tiene {len(pregunta)} caracteres; el máximo es {recuperacion.MAX_LONGITUD_PREGUNTA}.")
    if not _es_entero(k) or not 1 <= k <= K_MAXIMO_MEDICION:
        raise ValueError(f"k debe ser un entero entre 1 y {K_MAXIMO_MEDICION}.")


def _ms(segundos) -> float | None:
    return None if segundos is None else round(segundos * 1000, 2)


def registrar_consulta(
    id_caso: str,
    pregunta: str,
    k: int,
    vectorstore,
    embeddings,
    modelo,
    contador: dict,
    max_consultas: int,
    precios: tuple[float, float],
    corrida: str,
    max_tokens: int,
    reloj=time.perf_counter,
    ahora=None,
    dormir=None,
) -> dict:
    """Embebe, busca y genera una respuesta midiendo cada etapa; los errores del modelo quedan en el registro."""
    _validar_consulta(pregunta, k)
    ahora = ahora or (lambda: datetime.datetime.now(datetime.timezone.utc))
    tiempos = {}
    vector, tiempos["embed_query"], reintentos = embeber_con_reintentos(embeddings, pregunta, contador, max_consultas, dormir, reloj)
    with medir(tiempos, "busqueda_vectorial", reloj):
        pares = vectorstore.similarity_search_by_vector_with_relevance_scores(vector, k=k)
    # en langchain-chroma ese método devuelve la DISTANCIA (menor = más parecido)
    documentos = [documento for documento, _ in sorted(pares, key=lambda par: float(par[1]))]
    contexto, _ = recuperacion.formatear_contexto(documentos)
    prompt = recuperacion.PROMPT_TEMPLATE.format(contexto=contexto or recuperacion.SIN_CONTEXTO, pregunta=pregunta)

    mensaje, error = None, None
    try:
        with medir(tiempos, "generacion", reloj):
            mensaje = modelo.invoke(prompt)
    except Exception as fallo:
        error = f"{type(fallo).__name__}: {str(fallo)[:300]}"

    tokens, costo, sin_datos, truncado, respuesta = None, None, False, False, ""
    if mensaje is not None:
        uso = getattr(mensaje, "usage_metadata", None) or {}
        entrada, salida = uso.get("input_tokens"), uso.get("output_tokens")
        if not _es_entero(entrada) or entrada == 0:
            # el cliente real no deja el uso vacío sino en ceros; sin uso el gasto no se controla
            sin_datos = True
        else:
            salida = salida if _es_entero(salida) else 0
            tokens = {"input_tokens": entrada, "output_tokens": salida, "total_tokens": entrada + salida}
            costo = calcular_costo(entrada, salida, *precios)
        truncado = (getattr(mensaje, "response_metadata", None) or {}).get("stop_reason") == "max_tokens"
        respuesta = recuperacion.texto_de_respuesta(mensaje)

    return {
        "timestamp": ahora().isoformat(),
        "corrida": corrida,
        "id_caso": id_caso,
        "pregunta": pregunta,
        "k": k,
        "max_tokens": max_tokens,
        "tiempos_ms": {etapa: _ms(tiempos.get(etapa)) for etapa in ETAPAS},
        "tiempo_total_ms": _ms(sum(tiempos.values())),
        "reintentos_embed": reintentos,
        "ids_recuperados": [f"{d.metadata.get('documento')}_{d.metadata.get('fragmento')}" for d in documentos],
        "caracteres_prompt": len(prompt),
        "tokens_prompt_aprox": aprox_tokens(prompt),
        "tokens": tokens,
        "costo_estimado": costo,
        "truncado": truncado,
        "sin_datos_de_uso": sin_datos,
        "admite_falta_de_contexto": recuperacion.admite_falta_de_contexto(respuesta),
        "respuesta": respuesta[:MAX_CARACTERES_RESPUESTA],
        "error": error,
    }


def _promedio(valores: list[float]) -> float:
    return round(sum(valores) / len(valores), 2)


def resumir(registros: list[dict]) -> dict:
    """Resume los registros sin error y con tokens: promedios por k, si la entrada crece con k y cuántas veces mandó la generación."""
    validos = [r for r in registros if r["error"] is None and r["tokens"] is not None]
    por_k = {}
    for k in sorted({r["k"] for r in validos}):
        grupo = [r for r in validos if r["k"] == k]
        promedio_ms = {etapa: _promedio([r["tiempos_ms"][etapa] for r in grupo]) for etapa in ETAPAS}
        promedio_ms["total"] = _promedio([r["tiempo_total_ms"] for r in grupo])
        por_k[k] = {
            "registros": len(grupo),
            "promedio_ms": promedio_ms,
            "promedio_input_tokens": _promedio([r["tokens"]["input_tokens"] for r in grupo]),
            "promedio_output_tokens": _promedio([r["tokens"]["output_tokens"] for r in grupo]),
            "costo_total": sum(r["costo_estimado"] for r in grupo),
        }
    entrada_por_caso = {}
    for r in validos:
        entrada_por_caso.setdefault(r["id_caso"], {})[r["k"]] = r["tokens"]["input_tokens"]
    comparables = {id_caso: ks for id_caso, ks in entrada_por_caso.items() if len(ks) > 1}
    no_crecen = [i for i, ks in comparables.items() if not ks[max(ks)] > ks[min(ks)]]
    mas_lenta = sum(
        1 for r in validos if r["tiempos_ms"]["generacion"] > max(r["tiempos_ms"]["embed_query"], r["tiempos_ms"]["busqueda_vectorial"])
    )
    return {
        "llamadas": len(registros),
        "con_error": sum(1 for r in registros if r["error"] is not None),
        "por_k": por_k,
        "tokens_crecen_con_k": {"si": (not no_crecen) if comparables else None, "casos_que_no_crecen": no_crecen},
        "generacion_mas_lenta": {"si": mas_lenta, "total": len(validos)},
    }


def anexar_registro(ruta: str, registro: dict) -> None:
    """Anexa el registro como una línea JSON con una sola escritura; crea el archivo si no existe."""
    with open(ruta, "a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(registro, ensure_ascii=False) + "\n")


def leer_registros(ruta: str) -> list[dict]:
    """Lee un JSONL de registros (ignora líneas en blanco); ValueError con el número de línea (desde 1) si una no es válida."""
    registros = []
    with open(ruta, encoding="utf-8") as archivo:
        for numero, linea in enumerate(archivo, start=1):
            if not linea.strip():
                continue
            try:
                registro = json.loads(linea)
            except ValueError:
                raise ValueError(f"línea {numero}: no es JSON.") from None
            if not isinstance(registro, dict):
                raise ValueError(f"línea {numero}: no es un objeto.")
            if "timestamp" not in registro or "tiempos_ms" not in registro:
                raise ValueError(f"línea {numero}: falta timestamp o tiempos_ms.")
            registros.append(registro)
    return registros
