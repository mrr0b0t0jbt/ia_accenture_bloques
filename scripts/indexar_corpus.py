"""Indexa el corpus en un vector store persistente de Chroma con embeddings de Gemini (Bloques 24 y 25).

Sin --confirmar-api-real es el modo revisión: carga el corpus, lo divide y verifica
la integridad sin llamar a ninguna API ni escribir ningún índice. Con el flag
llama a la API de embeddings de Gemini (REAL) y escribe el índice.

Cómo correrlo (desde cualquier directorio):

    python scripts/indexar_corpus.py
    python scripts/indexar_corpus.py --comparar-tamanos 120,200,300
    python scripts/indexar_corpus.py --confirmar-api-real
"""
import argparse
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, os.path.join(RAIZ, "src"))

import indexacion  # noqa: E402
from chromadb.errors import InvalidArgumentError  # noqa: E402

NOTA_COSTO = "Los embeddings usan gemini-embedding-001; el proyecto no tiene su precio, por eso no se estima un costo en dólares."
AVISO_LIMITE = "Se agotó el límite de peticiones de la API: espera un minuto y vuelve a ejecutar."
AVISO_DIMENSION = "El índice se creó con otro modelo de embeddings: borra la carpeta del índice y reindexa."


def _tamanos(texto: str) -> list[int]:
    """Convierte '120,200' en [120, 200]; ValueError si algún elemento no es un entero."""
    try:
        return [int(parte) for parte in texto.split(",")]
    except ValueError:
        raise ValueError(f"--comparar-tamanos debe ser una lista de enteros separada por comas, no {texto!r}.") from None


def _revisar(documentos: list[dict], args) -> int:
    """Modo revisión: divide, verifica e imprime sin tocar la API ni el disco."""
    print("Modo revisión: no se llama a ninguna API ni se escribe ningún índice.")
    chunks, _ = indexacion.dividir_corpus(documentos, args.chunk_size, args.chunk_overlap)
    informe = indexacion.verificar_integridad(documentos, chunks)
    print(f"Corpus: {len(documentos)} documentos, {informe['caracteres_corpus']} caracteres")
    print(f"Chunks (tamaño {args.chunk_size}, solapamiento {args.chunk_overlap}): {informe['chunks']} chunks, {informe['caracteres_chunks']} caracteres")
    for documento in documentos:
        print(f"  {documento['id']}: {sum(1 for c in chunks if c.metadata['documento'] == documento['id'])} chunks")
    if informe["ok"]:
        print("Integridad: sin vacíos, sin duplicados, sin chunks sin fuente")
    else:
        print(f"Integridad: problemas: vacíos {informe['vacios']}, duplicados {informe['duplicados']}, sin fuente {informe['sin_fuente']}")
    if args.comparar_tamanos:
        for tamano in _tamanos(args.comparar_tamanos):
            otros, _ = indexacion.dividir_corpus(documentos, tamano, args.chunk_overlap)
            print(f"tamaño {tamano}: {len(otros)} chunks, {sum(len(c.page_content) for c in otros)} caracteres")
    return 0 if informe["ok"] else 1


def _indexar(documentos: list[dict], args) -> int:
    """Indexa con embeddings REALES de Gemini."""
    # import diferido: sin el flag no se carga langchain_google_genai ni se lee la clave
    import embeddings_gemini

    informe = indexacion.indexar(
        documentos,
        embeddings_gemini.crear_embeddings(),
        args.directorio,
        coleccion=args.coleccion,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"Indexados {informe['chunks']} chunks ({informe['caracteres_chunks']} caracteres) en {informe['directorio']} (colección {informe['coleccion']})")
    print(f"Eliminados: {informe['eliminados']}")
    print(f"Chunks en el índice: {informe['total_en_indice']}")
    print(NOTA_COSTO)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Revisa el corpus sin API o, con --confirmar-api-real, lo indexa en Chroma."""
    parser = argparse.ArgumentParser(description="Indexa el corpus en Chroma con embeddings de Gemini (modo revisión sin API por defecto).")
    parser.add_argument("--corpus", default=str(RAIZ / "JSON" / "corpus_miniapp.json"))
    parser.add_argument("--directorio", default=str(RAIZ / "vectorstore"))
    parser.add_argument("--coleccion", default=indexacion.COLECCION_POR_DEFECTO)
    parser.add_argument("--chunk-size", type=int, default=indexacion.CHUNK_SIZE_POR_DEFECTO)
    parser.add_argument("--chunk-overlap", type=int, default=indexacion.CHUNK_OVERLAP_POR_DEFECTO)
    parser.add_argument("--comparar-tamanos", help="Tamaños de chunk a comparar, separados por comas (solo modo revisión).")
    parser.add_argument("--confirmar-api-real", action="store_true")
    args = parser.parse_args(argv)
    try:
        documentos = indexacion.cargar_corpus(args.corpus)
        return _indexar(documentos, args) if args.confirmar_api_real else _revisar(documentos, args)
    except InvalidArgumentError as error:
        print(f"Error: {error}\n{AVISO_DIMENSION}", file=sys.stderr)
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
    except Exception as error:
        if "429" not in str(error) and "RESOURCE_EXHAUSTED" not in str(error):
            raise
        print(f"Error: {error}\n{AVISO_LIMITE}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
