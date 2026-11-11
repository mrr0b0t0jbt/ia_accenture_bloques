import os

import pytest

RUTA_TRAZAS_REALES = "trazas/trazas_agente.jsonl"


def _estado(ruta: str) -> tuple | None:
    """Devuelve (tamaño, mtime_ns) del archivo, o None si no existe."""
    if not os.path.exists(ruta):
        return None
    info = os.stat(ruta)
    return info.st_size, info.st_mtime_ns


@pytest.fixture(autouse=True)
def _no_escribe_en_las_trazas_reales():
    """Falla la prueba que cree o modifique trazas/trazas_agente.jsonl respecto del directorio actual."""
    ruta = os.path.abspath(RUTA_TRAZAS_REALES)
    antes = _estado(ruta)
    yield
    if _estado(ruta) != antes:
        pytest.fail(f"La prueba escribió en {RUTA_TRAZAS_REALES}: debe usar --sin-trazas o una ruta en tmp_path.")
