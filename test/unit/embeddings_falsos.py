import math
import re
import unicodedata

from langchain_core.embeddings import Embeddings


def _palabras(texto: str) -> set[str]:
    """Palabras del texto en minúsculas y sin tildes."""
    sin_tildes = "".join(c for c in unicodedata.normalize("NFD", texto.lower()) if not unicodedata.combining(c))
    return set(re.findall(r"[a-z0-9]+", sin_tildes))


class EmbeddingsPorTema(Embeddings):
    """Embeddings falsos y deterministas: una dimensión por tema (palabras presentes) y una constante final.

    No entiende significado: solo permite probar el orden de recuperación con temas controlados.
    """

    def __init__(self, temas: dict[str, list[str]]):
        self.temas = {tema: set(palabras) for tema, palabras in temas.items()}
        self.llamadas_documentos = 0
        self.llamadas_consulta = 0

    def _vector(self, texto: str) -> list[float]:
        palabras = _palabras(texto)
        vector = [float(len(tema & palabras)) for tema in self.temas.values()] + [0.01]
        norma = math.sqrt(sum(x * x for x in vector))
        return [x / norma for x in vector]

    def embed_documents(self, textos: list[str]) -> list[list[float]]:
        self.llamadas_documentos += 1
        return [self._vector(texto) for texto in textos]

    def embed_query(self, texto: str) -> list[float]:
        self.llamadas_consulta += 1
        return self._vector(texto)
