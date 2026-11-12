import os

from dotenv import load_dotenv

MODELO_EMBEDDINGS = "models/gemini-embedding-001"


def describir_error_api(error: Exception) -> tuple[str, str]:
    """Clasifica un error de la API en (categoría, mensaje en español): limite, clave u otro."""
    texto = str(error)
    if "429" in texto or "RESOURCE_EXHAUSTED" in texto:
        return "limite", "Se agotó el límite de peticiones de la API: espera un minuto y vuelve a ejecutar."
    if any(marca in texto for marca in ("API key", "PERMISSION_DENIED", "UNAUTHENTICATED", "401", "403")):
        return "clave", "La API rechazó la clave o no tiene permiso para usar embeddings: revisa GEMINI_API_KEY en .env."
    return "otro", f"No se pudo completar la operación con la API ({type(error).__name__}): {texto[:300]}"


def crear_embeddings():
    """Crea los embeddings de Gemini con la clave de .env; el modelo de embeddings no admite timeout propio."""
    load_dotenv()
    clave = os.getenv("GEMINI_API_KEY")
    if not clave:
        raise ValueError("Falta GEMINI_API_KEY en .env.")
    # import perezoso: importar este módulo no carga langchain_google_genai
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDINGS, api_key=clave)
