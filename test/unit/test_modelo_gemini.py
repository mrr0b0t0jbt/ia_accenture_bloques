import importlib
import sys
import warnings

import pytest
from langchain_core.messages import HumanMessage

import modelo_gemini


@pytest.fixture
def clave_falsa(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "clave-falsa-de-prueba")
    monkeypatch.setattr(modelo_gemini, "load_dotenv", lambda: None)


def test_sin_clave_lanza_value_error_con_el_mensaje_exacto(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(modelo_gemini, "load_dotenv", lambda: None)
    with pytest.raises(ValueError) as error:
        modelo_gemini.crear_modelo_gemini()
    assert str(error.value) == "Falta GEMINI_API_KEY en .env."


def test_con_clave_vacia_lanza_value_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setattr(modelo_gemini, "load_dotenv", lambda: None)
    with pytest.raises(ValueError):
        modelo_gemini.crear_modelo_gemini()


def test_con_clave_falsa_devuelve_el_modelo_configurado(clave_falsa):
    from langchain_google_genai import ChatGoogleGenerativeAI

    modelo = modelo_gemini.crear_modelo_gemini()
    assert isinstance(modelo, ChatGoogleGenerativeAI)
    assert modelo.model == "gemini-3.5-flash-lite"
    assert modelo.max_output_tokens == 2048
    assert modelo.max_retries == 2


def test_max_output_tokens_se_conserva(clave_falsa):
    assert modelo_gemini.crear_modelo_gemini(max_output_tokens=512).max_output_tokens == 512


def test_la_peticion_se_arma_sin_red_y_sin_aviso_de_muestreo(clave_falsa):
    modelo = modelo_gemini.crear_modelo_gemini()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # método interno a propósito: es la única forma de ejercitar el armado de la petición sin red
        modelo._prepare_request([HumanMessage(content="hola")])


def test_importar_el_modulo_no_requiere_clave_ni_la_libreria(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "langchain_google_genai", None)
    importlib.reload(modelo_gemini)
