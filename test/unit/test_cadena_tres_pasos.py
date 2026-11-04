from unittest.mock import MagicMock

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage

import modelo_langchain
from cadena_tres_pasos import PLANTILLA_SIMPLIFICAR, construir_cadena, construir_pasos

AVISO = {"texto": "Aviso ficticio: el producto Ejemplo 1.0 tiene una falla en su acceso remoto."}


def _modelo_falso(*respuestas: str) -> MagicMock:
    return MagicMock(side_effect=[AIMessage(content=r) for r in respuestas])


def test_cadena_devuelve_la_ultima_respuesta_tras_tres_llamadas():
    modelo = _modelo_falso("puntos", "texto simple", "- [ ] checklist")
    assert construir_cadena(modelo).invoke(AVISO) == "- [ ] checklist"
    assert modelo.call_count == 3


def test_salida_de_cada_paso_llega_al_prompt_del_siguiente():
    respuestas = iter(["PUNTOS-DEL-PASO-1", "TEXTO-DEL-PASO-2", "checklist"])
    prompts = []

    def registrar(entrada):
        prompts.append(str(entrada))
        return AIMessage(content=next(respuestas))

    construir_cadena(MagicMock(side_effect=registrar)).invoke(AVISO)
    assert "PUNTOS-DEL-PASO-1" in prompts[1]
    assert "TEXTO-DEL-PASO-2" in prompts[2]


def test_plantilla_del_paso_2_sin_la_variable_puntos_lanza_key_error():
    with pytest.raises(KeyError):
        PLANTILLA_SIMPLIFICAR.invoke({})


def test_cadena_y_cada_paso_suelto_devuelven_str():
    resultado = construir_cadena(_modelo_falso("uno", "dos", "tres")).invoke(AVISO)
    extraer, simplificar, formatear = construir_pasos(_modelo_falso("uno", "dos", "tres"))
    salidas = [
        resultado,
        extraer.invoke(AVISO),
        simplificar.invoke({"puntos": "uno"}),
        formatear.invoke({"texto_simple": "dos"}),
    ]
    # StrOutputParser devuelve una subclase de str (TextAccessor), por eso isinstance y no type
    assert all(isinstance(salida, str) and not hasattr(salida, "content") for salida in salidas)


def test_crear_modelo_sin_clave_lanza_value_error_sin_exponer_claves(monkeypatch):
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    monkeypatch.setattr(modelo_langchain, "load_dotenv", lambda: None)
    with pytest.raises(ValueError) as error:
        modelo_langchain.crear_modelo()
    assert str(error.value) == "Falta CLAUDE_API_KEY en .env."


def test_crear_modelo_con_clave_falsa_devuelve_chat_anthropic(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "clave-falsa-de-prueba")
    monkeypatch.setattr(modelo_langchain, "load_dotenv", lambda: None)
    modelo = modelo_langchain.crear_modelo()
    assert isinstance(modelo, ChatAnthropic)
    assert modelo.model == "claude-sonnet-5-5"
    assert modelo.temperature is None
