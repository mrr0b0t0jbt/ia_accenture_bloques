import os

from dotenv import load_dotenv

MODELO_GEMINI = "gemini-3.5-flash-lite"


def crear_modelo_gemini(max_output_tokens: int = 2048, timeout: float = 30, thinking_level: str | None = None):
    """Crea el ChatGoogleGenerativeAI de Gemini 3.5 Flash-Lite con la clave de .env."""
    load_dotenv()
    clave = os.getenv("GEMINI_API_KEY")
    if not clave:
        raise ValueError("Falta GEMINI_API_KEY en .env.")
    # import perezoso: importar este módulo no carga langchain_google_genai
    from langchain_google_genai import ChatGoogleGenerativeAI

    # este modelo usa muestreo fijo: no se pasa temperature
    opciones = {}
    # solo se pasa si se pide: el nivel por defecto del modelo es minimal, y la documentación
    # de Gemini recomienda medium o high para agentes con herramientas
    if thinking_level is not None:
        opciones["thinking_level"] = thinking_level
    return ChatGoogleGenerativeAI(
        model=MODELO_GEMINI,
        api_key=clave,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        max_retries=2,
        **opciones,
    )
