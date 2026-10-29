"""Pipeline de limpieza y tokenización de texto (Bloque 15)."""
import re
from collections import Counter


def limpiar_texto(texto: str) -> str:
    """Pasa a minúsculas, quita la puntuación y colapsa los espacios."""
    if not isinstance(texto, str):
        raise TypeError(f"limpiar_texto espera str, no {type(texto).__name__}")
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    return re.sub(r"\s+", " ", texto).strip()


def tokenizar(texto_limpio: str) -> list[str]:
    """Separa el texto limpio en palabras."""
    return texto_limpio.split()


def procesar_texto(texto: str) -> list[str]:
    """Limpia y tokeniza el texto."""
    return tokenizar(limpiar_texto(texto))


def vectorizar(tokens: list[str]) -> Counter:
    """Cuenta cuántas veces aparece cada token."""
    return Counter(tokens)


def coincidencia_por_palabras(tokens_referencia: list[str], tokens_respuesta: list[str]) -> float:
    """Proporción de palabras distintas de la referencia presentes en la respuesta."""
    referencia = set(tokens_referencia)
    if not referencia:
        return 0.0
    return len(referencia & set(tokens_respuesta)) / len(referencia)


def comparar_antes_despues(referencia: str, respuesta: str) -> dict:
    """Coincidencia con el texto crudo (split) y con el texto procesado."""
    return {
        "pct_crudo": coincidencia_por_palabras(referencia.split(), respuesta.split()),
        "pct_procesado": coincidencia_por_palabras(procesar_texto(referencia), procesar_texto(respuesta)),
    }
