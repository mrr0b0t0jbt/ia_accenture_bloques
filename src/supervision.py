import copy
import math
import os
import select
import sys
import termios
import time
from datetime import datetime, timezone

TIEMPO_LIMITE_POR_DEFECTO = 60
TIEMPO_LIMITE_MAXIMO = 3600
MAX_BYTES_LINEA = 1024
# select no admite descriptores mayores o iguales a FD_SETSIZE
MAX_DESCRIPTOR = 1024
MAX_APROBACIONES_POR_TURNO = 3
MAX_APROBACIONES_POR_CORRIDA = 10


def _leer_linea(fd: int, tiempo_limite_segundos: float) -> bytes | None:
    """Lee un byte por vez hasta el salto de línea; None si hay EOF, línea parcial o más de 1024 bytes, y TimeoutError si vence el plazo."""
    # un byte por vez: leer más consumiría líneas posteriores que pertenecen a la siguiente pregunta
    limite = time.monotonic() + tiempo_limite_segundos
    linea = b""
    while True:
        restante = limite - time.monotonic()
        if restante <= 0 or not select.select([fd], [], [], restante)[0]:
            raise TimeoutError
        byte = os.read(fd, 1)
        if not byte:
            return None
        if byte == b"\n":
            return linea
        linea += byte
        if len(linea) > MAX_BYTES_LINEA:
            return None


def esperar_aprobacion(
    mensaje: str, tiempo_limite_segundos: float = TIEMPO_LIMITE_POR_DEFECTO, fd_entrada: int | None = None
) -> bool:
    """Pide una aprobación por consola y devuelve True solo si la respuesta es "s"; el plazo (máximo 3600 s) cubre toda la lectura."""
    # el valor por defecto ante plazo vencido, EOF, línea parcial, entrada inválida o error es no aprobar (nunca lanza,
    # salvo KeyboardInterrupt); select sobre un descriptor solo funciona en Linux y macOS
    if type(tiempo_limite_segundos) not in (int, float) or not math.isfinite(tiempo_limite_segundos):
        return False
    if not 0 < tiempo_limite_segundos <= TIEMPO_LIMITE_MAXIMO:
        return False
    try:
        fd = sys.stdin.fileno() if fd_entrada is None else fd_entrada
        if type(fd) is not int or not 0 <= fd < MAX_DESCRIPTOR:
            return False
        # en una terminal se descarta lo tecleado antes: una tecla suelta no puede aprobar una pregunta que aún no se vio
        if os.isatty(fd):
            termios.tcflush(fd, termios.TCIFLUSH)
        print(f"{mensaje}(plazo {tiempo_limite_segundos:g} s) ", end="", flush=True)
        try:
            linea = _leer_linea(fd, tiempo_limite_segundos)
        except TimeoutError:
            print("\nTiempo agotado: no aprobado.", flush=True)
            return False
        return linea is not None and linea.strip().lower() == b"s"
    except Exception:
        return False


def denegar_siempre(mensaje: str) -> bool:
    """Aprobador para corridas desatendidas: nunca aprueba."""
    return False


def crear_aprobador_interactivo(tiempo_limite_segundos: float = TIEMPO_LIMITE_POR_DEFECTO):
    """Devuelve un aprobador que pregunta por consola con el tiempo límite indicado."""

    def aprobador(mensaje: str) -> bool:
        return esperar_aprobacion(mensaje, tiempo_limite_segundos)

    return aprobador


class LimiteAprobaciones:
    """Cuenta las aprobaciones solicitadas por turno y por corrida y recuerda los hosts negados en el turno."""

    def __init__(
        self, max_por_turno: int = MAX_APROBACIONES_POR_TURNO, max_por_corrida: int = MAX_APROBACIONES_POR_CORRIDA
    ) -> None:
        self.max_por_turno = max_por_turno
        self.max_por_corrida = max_por_corrida
        self._en_el_turno = 0
        self._en_la_corrida = 0
        self.negados: set[str] = set()

    def nuevo_turno(self) -> None:
        """Reinicia el contador del turno y los hosts negados; el de la corrida sigue."""
        self._en_el_turno = 0
        self.negados.clear()

    def agotado(self) -> bool:
        """Dice si ya se solicitó el máximo de aprobaciones del turno o de la corrida."""
        return self._en_el_turno >= self.max_por_turno or self._en_la_corrida >= self.max_por_corrida

    def contar(self) -> None:
        """Anota una aprobación solicitada."""
        self._en_el_turno += 1
        self._en_la_corrida += 1


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
