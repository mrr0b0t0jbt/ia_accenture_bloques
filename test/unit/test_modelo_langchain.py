import modelo_langchain


def test_crear_modelo_configura_dos_reintentos(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "clave-falsa-de-prueba")
    monkeypatch.setattr(modelo_langchain, "load_dotenv", lambda: None)
    assert modelo_langchain.crear_modelo().max_retries == 2
