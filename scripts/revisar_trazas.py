"""Revisión automática de las trazas del agente (Bloque 23): solo lectura, sin API y sin costo.

Cómo correrlo (desde la raíz del proyecto):

    python scripts/revisar_trazas.py --trazas trazas/trazas_agente.jsonl

Las marcas son candidatos a revisar, no conclusiones: el criterio es de una persona.
"""
import argparse
import os
import sys

_DIRECTORIO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIRECTORIO, "..", "src"))

from trazas import RUTA_TRAZAS, leer_trazas, marcar_repeticiones_sospechosas, marcar_riesgo_alto_sin_aprobacion  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Imprime las marcas de revisión de las trazas y devuelve 0; 1 si el archivo falta o está corrupto."""
    parser = argparse.ArgumentParser(description="Marca candidatos a revisar en las trazas del agente.")
    parser.add_argument("--trazas", default=RUTA_TRAZAS)
    args = parser.parse_args(argv)
    try:
        registros = leer_trazas(args.trazas)
    except FileNotFoundError:
        print(f"Error: no se encontró {args.trazas}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    marcas = marcar_repeticiones_sospechosas(registros) + marcar_riesgo_alto_sin_aprobacion(registros)
    print("\n".join(marcas) if marcas else "Sin marcas.")
    print(f"Registros revisados: {len(registros)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
