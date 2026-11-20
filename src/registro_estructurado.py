"""Registro estructurado (Bloque 31, Unidad 2.7): logs JSON de una línea, SIN contenido, con trace_id, huellas y consumo.

Regla de gobernanza: ningún log lleva el texto de una pregunta, de un fragmento o de una respuesta del modelo;
solo longitud, huella (sha256, 12 caracteres hexadecimales) e ids. Sin red ni claves.
"""
import contextlib
import contextvars
import hashlib
import json
import logging
import math
import sys
import uuid
from collections.abc import Iterator
from datetime import datetime, timezone

from langchain_core.callbacks import BaseCallbackHandler

CLAVES_RESERVADAS = ("ts", "level", "msg", "trace_id")
MAX_LONGITUD_TRACE_ID = 64
MENSAJE_INVALIDO = "<mensaje invalido>"
trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)


def huella(texto: str) -> str:
    """Devuelve los primeros 12 caracteres hexadecimales del sha256 del texto (un surrogate suelto no lanza)."""
    if not isinstance(texto, str):
        raise ValueError("texto no es texto")
    return hashlib.sha256(texto.encode("utf-8", errors="replace")).hexdigest()[:12]


def resumen_de_texto(texto) -> dict:
    """Devuelve longitud y huella del texto, o ambas en None si no es texto; nunca el texto."""
    if isinstance(texto, str):
        return {"longitud": len(texto), "huella": huella(texto)}
    return {"longitud": None, "huella": None}


@contextlib.contextmanager
def con_trace_id(valor: str | None = None) -> Iterator[str]:
    """Fija el trace_id (generado si no se da) mientras dura el bloque y restaura el anterior al salir."""
    if valor is None:
        valor = "trace-" + uuid.uuid4().hex[:12]
    elif not (
        isinstance(valor, str)
        and 1 <= len(valor) <= MAX_LONGITUD_TRACE_ID
        and valor.isprintable()
        and valor == valor.strip()
    ):
        raise ValueError("trace_id invalido")
    token = trace_id_var.set(valor)
    try:
        yield valor
    finally:
        trace_id_var.reset(token)


def _nombre_del_tipo(valor) -> str:
    # sin repr ni str: el objeto puede traer contenido
    return f"<{type(valor).__name__}>"


class FormateadorJson(logging.Formatter):
    """Formatea cada registro como UNA línea JSON ASCII: ts, level, msg, trace_id y los datos."""

    def format(self, record: logging.LogRecord) -> str:
        try:
            mensaje = record.getMessage()
        except Exception:
            mensaje = MENSAJE_INVALIDO
        ts = datetime.fromtimestamp(record.created, timezone.utc).isoformat(timespec="milliseconds")
        base = {"ts": ts, "level": record.levelname, "msg": mensaje, "trace_id": trace_id_var.get()}
        linea = dict(base)
        if hasattr(record, "datos"):
            if isinstance(record.datos, dict):
                for clave, valor in record.datos.items():
                    linea[f"datos_{clave}" if clave in CLAVES_RESERVADAS else clave] = valor
            else:
                linea["datos_invalidos"] = True
        tipo = record.exc_info[0] if record.exc_info else None
        if tipo is not None:
            linea["excepcion"] = tipo.__name__
        try:
            return json.dumps(linea, ensure_ascii=True, allow_nan=False, default=_nombre_del_tipo)
        except (ValueError, TypeError, RecursionError):
            ilegible = {**base, "datos_ilegibles": True}
            if tipo is not None:
                ilegible["excepcion"] = tipo.__name__
            return json.dumps(ilegible, ensure_ascii=True, allow_nan=False, default=_nombre_del_tipo)


def crear_logger(nombre: str = "consulta_rag", flujo=None, nivel: int = logging.INFO) -> logging.Logger:
    """Deja el logger con un único handler JSON hacia el flujo (stderr por defecto) y sin propagar."""
    if not isinstance(nombre, str) or not nombre:
        raise ValueError("nombre invalido")
    logger = logging.getLogger(nombre)
    for anterior in list(logger.handlers):
        logger.removeHandler(anterior)
        anterior.close()
    handler = logging.StreamHandler(sys.stderr if flujo is None else flujo)
    handler.setFormatter(FormateadorJson())
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(nivel)
    return logger


def registrar_evento(logger: logging.Logger | None, nivel: int, msg: str, **datos) -> None:
    """Registra un evento con sus datos; con logger None no hace nada."""
    if logger is None:
        return
    if not isinstance(msg, str):
        raise ValueError("msg invalido")
    logger.log(nivel, msg, extra={"datos": datos})


def _es_entero(valor) -> bool:
    return isinstance(valor, int) and not isinstance(valor, bool)


def _precios_validos(precios) -> bool:
    return (
        isinstance(precios, (tuple, list))
        and len(precios) == 2
        and all(isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p) and p >= 0 for p in precios)
    )


def registrar_consumo(logger: logging.Logger | None, respuesta, precios=None) -> dict:
    """Lee el uso de tokens de la respuesta, lo registra y devuelve el consumo (uso_faltante si no hay entrada)."""
    if precios is not None and not _precios_validos(precios):
        raise ValueError("precios invalidos")
    uso = getattr(respuesta, "usage_metadata", None) or {}
    entrada = uso.get("input_tokens")
    # igual que scripts/consultar_corpus.py: el cliente real deja los tokens en 0 en vez de vacíos
    if not _es_entero(entrada) or entrada == 0:
        consumo = {"input_tokens": None, "output_tokens": None, "total_tokens": None, "costo_estimado": None, "uso_faltante": True}
        registrar_evento(logger, logging.WARNING, "consumo_sin_datos_de_uso", uso_faltante=True)
        return consumo
    salida = uso.get("output_tokens")
    salida = salida if _es_entero(salida) else 0
    total = uso.get("total_tokens")
    consumo = {
        "input_tokens": entrada,
        "output_tokens": salida,
        "total_tokens": total if _es_entero(total) else entrada + salida,
        "costo_estimado": entrada * precios[0] + salida * precios[1] if precios else None,
        "uso_faltante": False,
    }
    registrar_evento(logger, logging.INFO, "consumo_registrado", **consumo)
    return consumo


def _ids_de_ejecucion(run_id, parent_run_id, metadata) -> dict:
    datos = {"run_id": str(run_id), "parent_run_id": None if parent_run_id is None else str(parent_run_id)}
    if isinstance(metadata, dict) and isinstance(metadata.get("trace_id"), str):
        datos["trace_id_metadata"] = metadata["trace_id"]
    return datos


class HandlerDeTrazas(BaseCallbackHandler):
    """Registra el ciclo de una cadena de LangChain (cadena, retriever, modelo) sin contenido; nunca rompe la cadena."""

    raise_error = False

    def __init__(self, logger: logging.Logger | None):
        super().__init__()
        self.logger = logger

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, metadata=None, **kwargs) -> None:
        if parent_run_id is None:
            registrar_evento(self.logger, logging.INFO, "cadena_inicio", **_ids_de_ejecucion(run_id, parent_run_id, metadata))

    def on_retriever_start(self, serialized, query, *, run_id, parent_run_id=None, metadata=None, **kwargs) -> None:
        datos = _ids_de_ejecucion(run_id, parent_run_id, metadata)
        registrar_evento(self.logger, logging.INFO, "retriever_inicio", query=resumen_de_texto(query), **datos)

    def on_retriever_end(self, documents, *, run_id, parent_run_id=None, **kwargs) -> None:
        n = len(documents) if isinstance(documents, (list, tuple)) else None
        registrar_evento(self.logger, logging.INFO, "retriever_fin", n_documentos=n, **_ids_de_ejecucion(run_id, parent_run_id, None))

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, metadata=None, **kwargs) -> None:
        n = len(prompts) if isinstance(prompts, (list, tuple)) else None
        registrar_evento(self.logger, logging.INFO, "llm_inicio", n_prompts=n, **_ids_de_ejecucion(run_id, parent_run_id, metadata))

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs) -> None:
        try:
            chars = len(response.generations[0][0].text)
        except (AttributeError, IndexError, TypeError):
            chars = None
        registrar_evento(self.logger, logging.INFO, "llm_fin", chars_respuesta=chars, **_ids_de_ejecucion(run_id, parent_run_id, None))
