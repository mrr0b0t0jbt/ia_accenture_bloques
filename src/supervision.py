import copy
import select
import sys
from datetime import datetime, timezone

TIEMPO_LIMITE_POR_DEFECTO = 60


def esperar_aprobacion(mensaje: str, tiempo_limite_segundos: float = TIEMPO_LIMITE_POR_DEFECTO, entrada=None) -> bool:
    """Pide una aprobación por consola y devuelve True solo si la respuesta es exactamente "s"."""
    # el valor por defecto ante timeout, EOF o error es no aprobar; select sobre stdin solo funciona en Linux y macOS
    entrada = sys.stdin if entrada is None else entrada
    print(mensaje, end="", flush=True)
    try:
        listos, _, _ = select.select([entrada], [], [], tiempo_limite_segundos)
        if not listos:
            return False
        return entrada.readline().strip().lower() == "s"
    except (OSError, ValueError):
        return False


def denegar_siempre(mensaje: str) -> bool:
    """Aprobador para corridas desatendidas: nunca aprueba."""
    return False


def crear_aprobador_interactivo(tiempo_limite_segundos: float = TIEMPO_LIMITE_POR_DEFECTO):
    """Devuelve un aprobador que pregunta por consola con el tiempo límite indicado."""

    def aprobador(mensaje: str) -> bool:
        return esperar_aprobacion(mensaje, tiempo_limite_segundos)

    return aprobador


class RegistroAprobaciones:
    """Guarda en memoria cada decisión de aprobación, aprobada o no."""

    def __init__(self) -> None:
        self._registros: list[dict] = []

    def registrar(self, herramienta: str, argumentos: dict, aprobado: bool) -> None:
        """Agrega la decisión con una copia de los argumentos y el momento en UTC."""
        self._registros.append(
            {
                "herramienta": herramienta,
                "argumentos": dict(argumentos),
                "aprobado": bool(aprobado),
                "momento": datetime.now(timezone.utc).isoformat(),
            }
        )

    @property
    def registros(self) -> list[dict]:
        """Devuelve una copia profunda de la lista de decisiones."""
        return copy.deepcopy(self._registros)
