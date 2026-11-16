import os

import pytest

RUTA_TRAZAS_REALES = "trazas/trazas_agente.jsonl"
RUTA_INDICE_REAL = "vectorstore"


RUTA_RESULTADOS_EVALUACION_REALES = "docs/resultados_evaluacion_recuperacion.json"
RUTA_REGISTROS_LATENCIA_REALES = "docs/registros_latencia_costo.jsonl"


def _estado(ruta: str) -> tuple | None:
    """Devuelve (tamaño, mtime_ns) del archivo, o None si no existe."""
    if not os.path.exists(ruta):
        return None
    info = os.stat(ruta)
    return info.st_size, info.st_mtime_ns


def estado_de_los_resultados_de_evaluacion_reales() -> tuple | None:
    """Estado de docs/resultados_evaluacion_recuperacion.json respecto del directorio actual (None si no existe)."""
    return _estado(os.path.abspath(RUTA_RESULTADOS_EVALUACION_REALES))


def estado_de_las_trazas_reales() -> tuple | None:
    """Estado de trazas/trazas_agente.jsonl respecto del directorio actual (None si no existe)."""
    return _estado(os.path.abspath(RUTA_TRAZAS_REALES))


def estado_del_indice_real() -> int | None:
    """mtime_ns del directorio vectorstore respecto del directorio actual (None si no existe)."""
    ruta = os.path.abspath(RUTA_INDICE_REAL)
    return os.stat(ruta).st_mtime_ns if os.path.exists(ruta) else None


@pytest.fixture(autouse=True)
def _no_escribe_en_el_indice_real():
    """Falla la prueba que cree o modifique el directorio vectorstore respecto del directorio actual."""
    antes = estado_del_indice_real()
    yield
    if estado_del_indice_real() != antes:
        pytest.fail(f"La prueba escribió en {RUTA_INDICE_REAL}/: debe usar un directorio en tmp_path.")


@pytest.fixture(autouse=True)
def _no_escribe_en_los_resultados_de_evaluacion_reales():
    """Falla la prueba que cree o modifique docs/resultados_evaluacion_recuperacion.json respecto del directorio actual."""
    antes = estado_de_los_resultados_de_evaluacion_reales()
    yield
    if estado_de_los_resultados_de_evaluacion_reales() != antes:
        pytest.fail(f"La prueba escribió en {RUTA_RESULTADOS_EVALUACION_REALES}: debe pasar --salida en tmp_path.")


@pytest.fixture(autouse=True)
def _no_escribe_en_los_registros_de_latencia_reales():
    """Falla la prueba que cree o modifique docs/registros_latencia_costo.jsonl respecto del directorio actual."""
    antes = _estado(os.path.abspath(RUTA_REGISTROS_LATENCIA_REALES))
    yield
    if _estado(os.path.abspath(RUTA_REGISTROS_LATENCIA_REALES)) != antes:
        pytest.fail(f"La prueba escribió en {RUTA_REGISTROS_LATENCIA_REALES}: debe pasar --salida en tmp_path.")


@pytest.fixture(autouse=True)
def _no_escribe_en_las_trazas_reales():
    """Falla la prueba que cree o modifique trazas/trazas_agente.jsonl respecto del directorio actual."""
    antes = estado_de_las_trazas_reales()
    yield
    if estado_de_las_trazas_reales() != antes:
        pytest.fail(f"La prueba escribió en {RUTA_TRAZAS_REALES}: debe usar --sin-trazas o una ruta en tmp_path.")
