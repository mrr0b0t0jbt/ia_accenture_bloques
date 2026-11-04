import os

from dotenv import load_dotenv


def crear_modelo(temperature: float = 0, max_tokens: int = 1024, timeout: float = 30):
    """Crea el ChatAnthropic de Sonnet 5.5 con la clave de .env."""
    load_dotenv()
    clave = os.getenv("CLAUDE_API_KEY")
    if not clave:
        raise ValueError("Falta CLAUDE_API_KEY en .env.")
    # import perezoso: importar este módulo no carga langchain_anthropic
    from langchain_anthropic import ChatAnthropic

    return ChatAnthropic(
        model="claude-sonnet-5-5",
        api_key=clave,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=2,
    )
