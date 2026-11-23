"""Compara la vía previa (supervision.esperar_aprobacion) con el grafo de supervisión del despliegue (Bloque 32).

Las MISMAS respuestas, por una tubería real, se dan a las dos vías; la decisión debe ser idéntica. Sin red ni claves:
el registro es una carpeta temporal.

    python scripts/comparar_supervision.py      # sale con 1 si alguna decisión difiere
"""
import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from supervision import esperar_aprobacion  # noqa: E402
from supervision_despliegue import SupervisionDespliegue  # noqa: E402

PLAZO = 0.2
# (nombre, bytes que llegan por la entrada; None = no llega nada y el plazo vence)
CASOS = (
    ("s", b"s\n"),
    ("S", b"S\n"),
    ("s con espacios", b" s \n"),
    ("n", b"n\n"),
    ("linea vacia", b"\n"),
    ("si", b"si\n"),
    ("EOF", b""),
    ("linea parcial sin salto", b"s"),
    ("linea de mas de 1024 bytes", b"s" * 1100 + b"\n"),
    ("plazo vencido", None),
)


def _entrada(datos):
    """Devuelve (descriptor de lectura, función que cierra ambos extremos) con los bytes ya escritos."""
    lectura, escritura = os.pipe()
    if datos:
        os.write(escritura, datos)
    if datos is not None:
        os.close(escritura)
        escritura = None

    def cerrar():
        os.close(lectura)
        if escritura is not None:
            os.close(escritura)

    return lectura, cerrar


def decision_previa(datos) -> bool:
    fd, cerrar = _entrada(datos)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return esperar_aprobacion("", PLAZO, fd)
    finally:
        cerrar()


def decision_grafo(datos, indice: int) -> bool:
    fd, cerrar = _entrada(datos)
    try:
        with tempfile.TemporaryDirectory() as carpeta:
            supervision = SupervisionDespliegue(os.path.join(carpeta, "registro.jsonl"))
            with contextlib.redirect_stdout(io.StringIO()):
                resultado = supervision.ejecutar(
                    "prompt", f"texto {indice}", "comparador", "comparacion", lambda mensaje: esperar_aprobacion(mensaje, PLAZO, fd)
                )
            return resultado["estado"] == "desplegado"
    finally:
        cerrar()


def comparar() -> list[dict]:
    filas = []
    for indice, (nombre, datos) in enumerate(CASOS):
        previa = decision_previa(datos)
        grafo = decision_grafo(datos, indice)
        filas.append({"caso": nombre, "previa": previa, "grafo": grafo, "igual": previa == grafo})
    return filas


def main() -> int:
    filas = comparar()
    print(f"{'caso':<30}{'previa':<10}{'grafo':<10}igual")
    for fila in filas:
        print(f"{fila['caso']:<30}{str(fila['previa']):<10}{str(fila['grafo']):<10}{fila['igual']}")
    return 0 if all(fila["igual"] for fila in filas) else 1


if __name__ == "__main__":
    sys.exit(main())
