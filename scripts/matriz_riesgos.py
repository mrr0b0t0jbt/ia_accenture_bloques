"""Matriz de riesgos del proyecto (Bloque 32): muestra la tabla ordenada y comprueba la matriz contra el codigo.

No llama a ninguna API ni lee claves. No escribe nada: solo lee docs/matriz_riesgos.json y los archivos .py citados como evidencia.

Como correrlo (desde cualquier directorio):

    python scripts/matriz_riesgos.py            # sale con 1 si la matriz es invalida o una evidencia no existe en el codigo
    python scripts/matriz_riesgos.py --matriz otra.json
"""
import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import matriz_riesgos  # noqa: E402


def _argumentos(argv):
    parser = argparse.ArgumentParser(description="Matriz de riesgos validada contra el codigo (sin API).")
    parser.add_argument("--matriz", default=str(RAIZ / matriz_riesgos.RUTA_MATRIZ_POR_DEFECTO), help="ruta del JSON de la matriz")
    parser.add_argument("--raiz", default=str(RAIZ), help="carpeta desde la que se resuelven las evidencias")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    argumentos = _argumentos(argv)
    try:
        filas = matriz_riesgos.leer_matriz(argumentos.matriz)
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    for linea in matriz_riesgos.tabla(filas):
        print(linea)
    problemas = matriz_riesgos.problemas_de_evidencias(filas, argumentos.raiz)
    if problemas:
        print()
        for problema in problemas:
            print(f"ERROR: {problema}", file=sys.stderr)
        return 1
    al_limite = sum(1 for fila in matriz_riesgos.filas_calculadas(filas) if fila["al_limite"])
    print()
    print(f"{len(filas)} riesgos, {al_limite} al limite de otra categoria; todas las evidencias existen en el codigo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
