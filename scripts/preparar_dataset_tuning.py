"""Prepara y valida el dataset de tuning (Bloque 29, Unidad 2.6) SIN ninguna API: no usa claves ni red, no lee .env y no escribe ningún archivo.

Carga JSON/dataset_tuning_clasificacion.json, lo valida de forma estricta y muestra el balance por categoría y prioridad.
Sale con 1 si el dataset tiene problemas o si falta algún valor permitido (categoría o prioridad sin ejemplos); un desbalance sin ausentes es solo un aviso.
El contenido del archivo es dato: todo lo que viene de él se limpia de caracteres de control antes de llegar a la consola.

--demostrar-validador arma en memoria los ejemplos del dataset más DOS ejemplos rotos y comprueba que el validador
detecta exactamente los cuatro problemas esperados (no aplica el chequeo de balance).

Cómo correrlo (desde cualquier directorio):

    python scripts/preparar_dataset_tuning.py
    python scripts/preparar_dataset_tuning.py --dataset ruta/al/dataset.json
    python scripts/preparar_dataset_tuning.py --demostrar-validador
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import dataset_tuning  # noqa: E402  (Python puro: no carga langchain ni ninguna API)


def _limpiar(texto) -> str:
    """Reemplaza por un espacio todo carácter no imprimible (incluidos los saltos de línea)."""
    return "".join(c if c.isprintable() else " " for c in str(texto))


def _imprimir(linea: str) -> None:
    print(_limpiar(linea))


def _error(mensaje: str, codigo: int = 1) -> int:
    print(f"Error: {_limpiar(mensaje)}", file=sys.stderr)
    return codigo


def _resumen(n: int, problemas: list[str]) -> None:
    _imprimir(f"{n} ejemplos, {len(problemas)} problemas encontrados")
    for problema in problemas:
        _imprimir(f" - {problema}")


def _demostrar_validador(ejemplos: list) -> int:
    """Agrega dos ejemplos rotos y exige que el validador detecte exactamente los cuatro problemas esperados."""
    n = len(ejemplos)
    rotos = [
        SimpleNamespace(text_input="", output=json.dumps({"categoria": "incidente"})),
        SimpleNamespace(text_input=ejemplos[0].text_input, output="no es json"),
    ]
    problemas = dataset_tuning.validar_dataset(ejemplos + rotos)
    _resumen(n + len(rotos), problemas)
    esperados = [
        f"ejemplo {n}: text_input vacio",
        f'ejemplo {n}: output no tiene la clave "prioridad"',
        f"ejemplo {n + 1}: text_input duplicado del ejemplo 0 con otra salida",
        f"ejemplo {n + 1}: output no es JSON valido",
    ]
    return 0 if problemas == esperados else _error("el validador no detecto lo esperado.")


def _crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepara y valida el dataset de tuning (no usa ninguna API).", allow_abbrev=False)
    parser.add_argument("--dataset", default=str(RAIZ / "JSON" / "dataset_tuning_clasificacion.json"))
    parser.add_argument("--demostrar-validador", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Valida el dataset y muestra su balance; devuelve 0 si está listo, 1 si hay problemas o no se pudo leer."""
    args = _crear_parser().parse_args(argv)
    try:
        datos = dataset_tuning.cargar_dataset(args.dataset)
        problemas_de_forma = dataset_tuning.validar_datos(datos)
        if problemas_de_forma:
            _resumen(len(datos.get("ejemplos", [])) if isinstance(datos.get("ejemplos"), list) else 0, problemas_de_forma)
            return _error(f"el dataset tiene {len(problemas_de_forma)} problemas; no esta listo para enviarse a entrenar.")
        ejemplos = dataset_tuning.construir_ejemplos(datos)
    except ValueError as error:
        return _error(str(error))
    if args.demostrar_validador:
        return _demostrar_validador(ejemplos)
    problemas = dataset_tuning.validar_dataset(ejemplos)
    _resumen(len(ejemplos), problemas)
    resumenes = {}
    for clave in dataset_tuning.CLAVES:
        resumen = dataset_tuning.resumir_balance(dataset_tuning.contar_por_clave(ejemplos, clave), dataset_tuning.VALORES_PERMITIDOS[clave])
        resumenes[clave] = resumen
        _imprimir(f"{clave}:")
        for valor, cuenta in resumen["conteo"].items():
            _imprimir(f"  {valor}: {cuenta}")
        _imprimir(f"  razon max/min: {resumen['razon']}x" if resumen["razon"] is not None else "  razon max/min: n/d")
    for aviso in dataset_tuning.avisos_de_balance(resumenes):
        _imprimir(aviso)
    codigo = 0
    if problemas:
        codigo = _error(f"el dataset tiene {len(problemas)} problemas; no esta listo para enviarse a entrenar.")
    for clave, resumen in resumenes.items():
        if resumen["ausentes"]:
            codigo = _error(f"{clave} no tiene ejemplos de: {', '.join(resumen['ausentes'])}; no esta listo para enviarse a entrenar.")
    return codigo


if __name__ == "__main__":
    sys.exit(main())
