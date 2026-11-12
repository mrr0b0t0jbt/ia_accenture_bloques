import json
import os
import re
import time

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

COLECCION_POR_DEFECTO = "corpus_miniapp"
CHUNK_SIZE_POR_DEFECTO = 200
CHUNK_OVERLAP_POR_DEFECTO = 30
MAX_DOCUMENTOS = 50
MAX_CHUNKS = 200
PATRON_NOMBRE_COLECCION = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]")
PATRON_ID_DOCUMENTO = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}")


def _texto_no_vacio(valor) -> bool:
    """Indica si el valor es una cadena con algo más que espacios."""
    return isinstance(valor, str) and bool(valor.strip())


def cargar_corpus(ruta: str) -> list[dict]:
    """Lee y valida el corpus JSON: una lista de documentos con id, fuente y texto."""
    try:
        with open(ruta, encoding="utf-8") as archivo:
            documentos = json.load(archivo)
    except FileNotFoundError:
        raise ValueError(f"Corpus: no existe el archivo {ruta}.") from None
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"Corpus: no se pudo leer {ruta}: {error}.") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"Corpus: {ruta} no es un JSON válido: {error}.") from None
    if not isinstance(documentos, list) or not documentos:
        raise ValueError("Corpus: el archivo debe ser una lista con al menos un documento.")
    if len(documentos) > MAX_DOCUMENTOS:
        raise ValueError(f"Corpus: tiene {len(documentos)} documentos; el máximo es {MAX_DOCUMENTOS}.")
    vistos = set()
    for n, documento in enumerate(documentos):
        if not isinstance(documento, dict):
            raise ValueError(f"Corpus: el elemento {n} no es un objeto.")
        for clave in ("id", "fuente", "texto"):
            if not _texto_no_vacio(documento.get(clave)):
                raise ValueError(f"Corpus: el elemento {n} no tiene la clave {clave} como texto no vacío.")
        id_documento = documento["id"]
        if not PATRON_ID_DOCUMENTO.fullmatch(id_documento):
            raise ValueError(f"Corpus: el id {id_documento!r} no es válido (letras, números, _ y -; hasta 64).")
        if id_documento in vistos:
            raise ValueError(f"Corpus: el id {id_documento!r} está repetido.")
        vistos.add(id_documento)
    return documentos


def validar_nombre_coleccion(nombre: str) -> str:
    """Devuelve el nombre si es una colección válida de Chroma; si no, lanza ValueError."""
    if not isinstance(nombre, str) or not PATRON_NOMBRE_COLECCION.fullmatch(nombre):
        raise ValueError(
            "El nombre de la colección debe tener de 3 a 512 caracteres [a-zA-Z0-9._-] "
            "y empezar y terminar con letra o número."
        )
    return nombre


def _es_entero(valor) -> bool:
    return isinstance(valor, int) and not isinstance(valor, bool)


def dividir_corpus(documentos: list[dict], chunk_size: int, chunk_overlap: int) -> tuple[list[Document], list[str]]:
    """Parte cada documento en chunks con metadata e ids deterministas f"{id}_{n}"."""
    if not _es_entero(chunk_size) or not 50 <= chunk_size <= 1000:
        raise ValueError("chunk_size debe ser un entero entre 50 y 1000.")
    if not _es_entero(chunk_overlap) or not 0 <= chunk_overlap <= chunk_size // 2:
        raise ValueError("chunk_overlap debe ser un entero entre 0 y la mitad de chunk_size.")
    divisor = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks, ids = [], []
    for documento in documentos:
        for n, texto in enumerate(divisor.split_text(documento["texto"])):
            metadata = {"fuente": documento["fuente"], "documento": documento["id"], "fragmento": n}
            chunks.append(Document(page_content=texto, metadata=metadata))
            ids.append(f"{documento['id']}_{n}")
    return chunks, ids


def verificar_integridad(documentos: list[dict], chunks: list[Document]) -> dict:
    """Revisa que los chunks no estén vacíos, repetidos ni sin fuente, y cuenta caracteres."""
    textos = [chunk.page_content for chunk in chunks]
    vacios = [n for n, texto in enumerate(textos) if not texto.strip()]
    sin_fuente = [n for n, chunk in enumerate(chunks) if not _texto_no_vacio(chunk.metadata.get("fuente"))]
    duplicados = len(set(textos)) != len(textos)
    return {
        "chunks": len(chunks),
        "vacios": vacios,
        "duplicados": duplicados,
        "sin_fuente": sin_fuente,
        "caracteres_corpus": sum(len(documento["texto"]) for documento in documentos),
        "caracteres_chunks": sum(len(texto) for texto in textos),
        "ok": not vacios and not duplicados and not sin_fuente,
    }


def con_reintentos(operacion, max_intentos: int = 4, espera_inicial: float = 5, dormir=None):
    """Ejecuta la operación y reintenta con espera creciente solo ante un 429 o RESOURCE_EXHAUSTED."""
    espera = espera_inicial
    for intento in range(1, max_intentos + 1):
        try:
            return operacion()
        except Exception as error:
            limite = "429" in str(error) or "RESOURCE_EXHAUSTED" in str(error)
            if not limite or intento == max_intentos:
                raise
            # se resuelve al llamar, para que monkeypatch sobre time.sleep funcione
            (dormir or time.sleep)(espera)
            espera *= 2


def abrir_vectorstore(embeddings, directorio: str, coleccion: str) -> Chroma:
    """Abre (o crea) la colección de Chroma en la ruta absoluta, sin telemetría."""
    validar_nombre_coleccion(coleccion)
    ruta = os.path.abspath(directorio)
    return Chroma(
        collection_name=coleccion,
        embedding_function=embeddings,
        persist_directory=ruta,
        client_settings=Settings(anonymized_telemetry=False, is_persistent=True, persist_directory=ruta),
    )


def indexar(
    documentos: list[dict],
    embeddings,
    directorio: str,
    coleccion: str = COLECCION_POR_DEFECTO,
    chunk_size: int = CHUNK_SIZE_POR_DEFECTO,
    chunk_overlap: int = CHUNK_OVERLAP_POR_DEFECTO,
    dormir=None,
) -> dict:
    """Indexa el corpus con ids deterministas: reindexar no duplica y borra lo que ya no está."""
    validar_nombre_coleccion(coleccion)
    chunks, ids = dividir_corpus(documentos, chunk_size, chunk_overlap)
    if len(chunks) > MAX_CHUNKS:
        raise ValueError(f"El corpus produce {len(chunks)} chunks; el máximo es {MAX_CHUNKS}.")
    ruta = os.path.abspath(directorio)
    if os.path.exists(ruta) and not os.path.isdir(ruta):
        raise ValueError(f"El directorio del índice no es un directorio: {ruta}")
    vectorstore = abrir_vectorstore(embeddings, ruta, coleccion)
    con_reintentos(lambda: vectorstore.add_documents(chunks, ids=ids), dormir=dormir)
    # los huérfanos se borran después de agregar: si la adición falla, el índice anterior queda intacto
    huerfanos = sorted(set(vectorstore.get()["ids"]) - set(ids))
    if huerfanos:
        vectorstore.delete(ids=huerfanos)
    return {
        "documentos": len(documentos),
        "chunks": len(chunks),
        "caracteres_corpus": sum(len(documento["texto"]) for documento in documentos),
        "caracteres_chunks": sum(len(chunk.page_content) for chunk in chunks),
        "directorio": ruta,
        "coleccion": coleccion,
        "total_en_indice": vectorstore._collection.count(),
        "eliminados": len(huerfanos),
    }


def cargar_vectorstore(embeddings, directorio: str, coleccion: str = COLECCION_POR_DEFECTO) -> Chroma:
    """Abre un índice ya creado; falla si el directorio no existe o la colección está vacía."""
    ruta = os.path.abspath(directorio)
    # Chroma crea el directorio sin avisar, por eso se comprueba antes
    if not os.path.isdir(ruta):
        raise FileNotFoundError(f"No existe el índice en {ruta}: ejecútalo primero con scripts/indexar_corpus.py.")
    vectorstore = abrir_vectorstore(embeddings, ruta, coleccion)
    if vectorstore._collection.count() == 0:
        raise ValueError(f"El índice en {ruta} está vacío (colección {coleccion}): reindexa con scripts/indexar_corpus.py.")
    return vectorstore
