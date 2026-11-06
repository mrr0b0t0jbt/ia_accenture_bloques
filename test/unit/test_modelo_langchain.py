from langchain_core.messages import HumanMessage

import modelo_langchain


def test_crear_modelo_configura_dos_reintentos(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "clave-falsa-de-prueba")
    monkeypatch.setattr(modelo_langchain, "load_dotenv", lambda: None)
    assert modelo_langchain.crear_modelo().max_retries == 2


def test_payload_del_modelo_pasa_la_validacion_de_sonnet_5_5_sin_red(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "clave-falsa-de-prueba")
    monkeypatch.setattr(modelo_langchain, "load_dotenv", lambda: None)
    # método interno a propósito: es la única forma de ejercitar la validación del modelo sin red
    payload = modelo_langchain.crear_modelo()._get_request_payload([HumanMessage(content="hola")])
    assert payload.get("temperature", 1) == 1


def test_payload_del_modelo_usa_max_tokens_2048_por_defecto(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "clave-falsa-de-prueba")
    monkeypatch.setattr(modelo_langchain, "load_dotenv", lambda: None)
    modelo = modelo_langchain.crear_modelo()
    assert modelo._get_request_payload([HumanMessage(content="hola")])["max_tokens"] == 2048
