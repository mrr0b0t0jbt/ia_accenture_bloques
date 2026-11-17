"""Chequeo previo del flujo RAG SIN ninguna API: no usa embeddings ni claves, no lee .env, no crea ni borra archivos y no cambia los datos ni el conteo del índice.
(Abrir la colección con Chroma reescribe unos bytes de chroma.sqlite3 y las fechas de modificación, igual que cualquier script que abre el índice.)

Comprueba, en este orden: que el directorio del índice exista, que el corpus se cargue y se parta, que la colección
de Chroma exista, no esté vacía y tenga EXACTAMENTE los chunks (conteo e ids) que produce el corpus, y que el .gitignore
ignore el directorio del índice (leyendo sus líneas: la última que coincide gana y una negación la anula).

Cómo correrlo (desde cualquier directorio):

    python scripts/preflight_rag.py
    python scripts/preflight_rag.py --directorio vectorstore --coleccion corpus_miniapp
"""
import argparse
import fnmatch
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import chromadb  # noqa: E402
from chromadb.config import Settings  # noqa: E402

import indexacion  # noqa: E402
import recuperacion  # noqa: E402

MAX_EJEMPLOS = 3


def _ejemplos(ids: set[str]) -> str:
    return ", ".join(sorted(ids)[:MAX_EJEMPLOS])


def _problema_de_indice(ruta: str, coleccion: str, ids_corpus: list[str] | None) -> str | None:
    """Abre la colección sin función de embeddings y la compara con los ids del corpus (None si no se pudo cargar)."""
    try:
        indexacion.validar_nombre_coleccion(coleccion)
        cliente = chromadb.PersistentClient(
            path=ruta, settings=Settings(anonymized_telemetry=False, is_persistent=True, persist_directory=ruta)
        )
        ids_indice = cliente.get_collection(coleccion).get(include=[])["ids"]
    except ValueError as error:
        return str(error)
    except Exception as error:  # colección inexistente (el tipo de error cambia entre versiones de chromadb)
        return f"no se pudo abrir la colección {coleccion!r} del índice en {ruta}: {str(error)[:200]} Indexa con scripts/indexar_corpus.py --confirmar-api-real."
    if not ids_indice:
        return f"la colección {coleccion!r} del índice está vacía: reindexa con scripts/indexar_corpus.py --confirmar-api-real."
    if ids_corpus is None:
        return None
    faltan, sobran = set(ids_corpus) - set(ids_indice), set(ids_indice) - set(ids_corpus)
    if len(ids_indice) != len(ids_corpus) or faltan or sobran:
        return (
            f"el índice tiene {len(ids_indice)} chunks y el corpus produce {len(ids_corpus)}: faltan {len(faltan)} "
            f"(p. ej. {_ejemplos(faltan)}) y sobran {len(sobran)} (p. ej. {_ejemplos(sobran)}); "
            "reindexa con scripts/indexar_corpus.py --confirmar-api-real."
        )
    return None


def _esta_ignorado(directorio: str, lineas: list[str]) -> bool:
    """Aplica las líneas del .gitignore al nombre del directorio: gana la última que coincide y '!' la anula."""
    nombre = Path(indexacion.ruta_absoluta(directorio)).name
    ignorado = False
    for linea in lineas:
        patron = linea.strip()
        if not patron or patron.startswith("#"):
            continue
        negacion = patron.startswith("!")
        patron = patron.lstrip("!").strip("/")
        if patron and fnmatch.fnmatchcase(nombre, patron):
            ignorado = not negacion
    return ignorado


def preflight_check(directorio: str, coleccion: str, corpus: str, gitignore: str) -> list[str]:
    """Devuelve los problemas (vacía si todo está en orden): sin APIs ni .env, sin crear ni borrar archivos y sin cambiar los datos ni el conteo del índice; abrir Chroma reescribe unos bytes de chroma.sqlite3 y las fechas de modificación."""
    problemas = []
    ruta = indexacion.ruta_absoluta(directorio)
    hay_indice = os.path.isdir(ruta)
    if not hay_indice:
        problemas.append(f"no existe el directorio del índice {ruta}: indexa primero con scripts/indexar_corpus.py --confirmar-api-real.")
    ids_corpus = None
    try:
        documentos = indexacion.cargar_corpus(corpus)
        _, ids_corpus = indexacion.dividir_corpus(documentos, indexacion.CHUNK_SIZE_POR_DEFECTO, indexacion.CHUNK_OVERLAP_POR_DEFECTO)
    except ValueError as error:
        problemas.append(str(error))
    if hay_indice:
        problema = _problema_de_indice(ruta, coleccion, ids_corpus)
        if problema:
            problemas.append(problema)
    try:
        with open(gitignore, encoding="utf-8") as archivo:
            lineas = archivo.read().splitlines()
    except (OSError, UnicodeDecodeError):
        problemas.append(f"no se pudo leer el .gitignore {gitignore}: el índice podría subirse al repositorio.")
    else:
        if not _esta_ignorado(directorio, lineas):
            problemas.append(f"el .gitignore no ignora el directorio del índice ({Path(ruta).name}): podría subirse al repositorio.")
    return problemas


def _crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chequeo previo del flujo RAG (no usa ninguna API).", allow_abbrev=False)
    parser.add_argument("--directorio", default=str(RAIZ / "vectorstore"))
    parser.add_argument("--coleccion", default=indexacion.COLECCION_POR_DEFECTO)
    parser.add_argument("--corpus", default=str(RAIZ / "JSON" / "corpus_miniapp.json"))
    parser.add_argument("--gitignore", default=str(RAIZ / ".gitignore"))
    return parser


def main(argv: list[str] | None = None) -> int:
    """Imprime el resultado del chequeo y devuelve 0 si todo está en orden o 1 si hay problemas."""
    args = _crear_parser().parse_args(argv)
    directorio, corpus, gitignore = (os.path.expanduser(ruta) for ruta in (args.directorio, args.corpus, args.gitignore))
    problemas = preflight_check(directorio, args.coleccion, corpus, gitignore)
    if not problemas:
        print(f"Preflight OK: el índice {recuperacion.limpiar_para_consola(args.coleccion)} coincide con el corpus y está ignorado por git.")
        return 0
    print("Preflight con problemas:")
    for problema in problemas:
        print(f" - {recuperacion.limpiar_para_consola(problema).replace(chr(10), ' ')}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
