import os

from dotenv import load_dotenv


def crear_modelo(max_tokens: int = 2048, timeout: float = 30):
    """Crea el ChatAnthropic de Sonnet 5.5 con la clave de .env."""
    load_dotenv()
    clave = os.getenv("CLAUDE_API_KEY")
    if not clave:
        raise ValueError("Falta CLAUDE_API_KEY en .env.")
    # import perezoso: importar este módulo no carga langchain_anthropic
    from langchain_anthropic import ChatAnthropic

    # Sonnet 5.5 solo admite la temperatura por defecto
    return ChatAnthropic(
        model="claude-sonnet-5-5",
        api_key=clave,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=2,
    )
