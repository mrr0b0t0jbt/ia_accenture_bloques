import os

import pytest

RUTA_TRAZAS_REALES = "trazas/trazas_agente.jsonl"


def _estado(ruta: str) -> tuple | None:
    """Devuelve (tamaño, mtime_ns) del archivo, o None si no existe."""
    if not os.path.exists(ruta):
        return None
    info = os.stat(ruta)
    return info.st_size, info.st_mtime_ns


def estado_de_las_trazas_reales() -> tuple | None:
    """Estado de trazas/trazas_agente.jsonl respecto del directorio actual (None si no existe)."""
    return _estado(os.path.abspath(RUTA_TRAZAS_REALES))


@pytest.fixture(autouse=True)
def _no_escribe_en_las_trazas_reales():
    """Falla la prueba que cree o modifique trazas/trazas_agente.jsonl respecto del directorio actual."""
    antes = estado_de_las_trazas_reales()
    yield
    if estado_de_las_trazas_reales() != antes:
        pytest.fail(f"La prueba escribió en {RUTA_TRAZAS_REALES}: debe usar --sin-trazas o una ruta en tmp_path.")
