import os

from dotenv import load_dotenv

MODELO_EMBEDDINGS = "models/gemini-embedding-001"


def crear_embeddings():
    """Crea los embeddings de Gemini con la clave de .env; el modelo de embeddings no admite timeout propio."""
    load_dotenv()
    clave = os.getenv("GEMINI_API_KEY")
    if not clave:
        raise ValueError("Falta GEMINI_API_KEY en .env.")
    # import perezoso: importar este módulo no carga langchain_google_genai
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=MODELO_EMBEDDINGS, api_key=clave)
