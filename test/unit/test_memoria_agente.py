import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agente import AlmacenHistoriales, construir_agente_con_memoria, construir_ejecutor, normalizar_salida
from modelo_falso_herramientas import ModeloFalsoHerramientas

BLOQUE_THINKING = {"type": "thinking", "thinking": "", "signature": "firma-opaca"}
BLOQUE_TEXTO = {"type": "text", "text": "Su severidad era media."}


def _sesion(session_id: str) -> dict:
    return {"configurable": {"session_id": session_id}}


def _agente_con_memoria(*respuestas: str) -> tuple:
    modelo = ModeloFalsoHerramientas(guion=[AIMessage(content=respuesta) for respuesta in respuestas])
    almacen = AlmacenHistoriales()
    return construir_agente_con_memoria(construir_ejecutor(modelo), almacen), almacen, modelo


def test_obtener_reutiliza_la_instancia_de_cada_sesion():
    almacen = AlmacenHistoriales()
    assert almacen.obtener("a") is almacen.obtener("a")
    assert almacen.obtener("a") is not almacen.obtener("b")


def test_mensajes_de_una_sesion_inexistente_devuelve_0_sin_crearla():
    almacen = AlmacenHistoriales()
    assert almacen.mensajes("x") == 0
    # se mira el atributo privado: la interfaz pública no permite saber si la sesión se creó
    assert "x" not in almacen._historiales


def test_el_segundo_turno_recibe_el_historial_del_primero():
    agente, _, modelo = _agente_con_memoria("respuesta-uno", "respuesta-dos")
    agente.invoke({"entrada": "pregunta-uno"}, config=_sesion("a"))
    agente.invoke({"entrada": "pregunta-dos"}, config=_sesion("a"))
    assert "pregunta-uno" in modelo.texto_recibido(1)
    assert "respuesta-uno" in modelo.texto_recibido(1)


def test_el_historial_crece_con_cada_turno():
    agente, almacen, _ = _agente_con_memoria("respuesta-uno", "respuesta-dos")
    agente.invoke({"entrada": "pregunta-uno"}, config=_sesion("a"))
    assert almacen.mensajes("a") == 2
    agente.invoke({"entrada": "pregunta-dos"}, config=_sesion("a"))
    assert almacen.mensajes("a") == 4


def test_las_sesiones_quedan_aisladas():
    agente, almacen, modelo = _agente_con_memoria("respuesta-a", "respuesta-b")
    agente.invoke({"entrada": "dato-secreto-de-a"}, config=_sesion("a"))
    agente.invoke({"entrada": "pregunta de b"}, config=_sesion("b"))
    assert len(modelo.entradas) == 2
    assert "dato-secreto-de-a" not in modelo.texto_recibido(1)
    assert almacen.mensajes("b") == 2
    assert almacen.mensajes("a") == 2


def test_invocar_sin_session_id_lanza_error_que_lo_menciona():
    agente, _, _ = _agente_con_memoria("respuesta")
    with pytest.raises(ValueError, match="session_id"):
        agente.invoke({"entrada": "pregunta"})


def test_cerrar_elimina_la_sesion_y_es_idempotente():
    agente, almacen, _ = _agente_con_memoria("respuesta")
    agente.invoke({"entrada": "pregunta"}, config=_sesion("a"))
    almacen.cerrar("a")
    assert almacen.mensajes("a") == 0
    almacen.cerrar("a")
    almacen.cerrar("inexistente")


@pytest.mark.parametrize(
    "salida, esperado",
    [
        ("hola", "hola"),
        ([BLOQUE_THINKING, BLOQUE_TEXTO], "Su severidad era media."),
        (["a", {"type": "text", "text": "b"}], "ab"),
    ],
    ids=["texto", "bloques_thinking_y_text", "str_y_bloque"],
)
def test_normalizar_salida_devuelve_texto(salida, esperado):
    assert normalizar_salida({"output": salida})["output"] == esperado


def test_normalizar_salida_conserva_las_demas_claves_y_no_modifica_el_original():
    original = {"output": [BLOQUE_TEXTO], "intermediate_steps": ["paso"]}
    resultado = normalizar_salida(original)
    assert resultado == {"output": "Su severidad era media.", "intermediate_steps": ["paso"]}
    assert original["output"] == [BLOQUE_TEXTO]


def _agente_con_bloques() -> tuple:
    modelo = ModeloFalsoHerramientas(guion=[AIMessage(content=[BLOQUE_THINKING, BLOQUE_TEXTO]), AIMessage(content="listo")])
    almacen = AlmacenHistoriales()
    return construir_agente_con_memoria(construir_ejecutor(modelo), almacen), almacen, modelo


def test_respuesta_en_bloques_se_guarda_como_texto_en_el_historial():
    agente, almacen, _ = _agente_con_bloques()
    resultado = agente.invoke({"entrada": "pregunta"}, config=_sesion("a"))
    assert resultado["output"] == "Su severidad era media."
    assert almacen.mensajes("a") == 2
    humano, ia = almacen.obtener("a").messages
    assert (type(humano), type(ia)) == (HumanMessage, AIMessage)


def test_segundo_turno_tras_respuesta_en_bloques_no_falla_ni_filtra_la_firma():
    agente, _, modelo = _agente_con_bloques()
    agente.invoke({"entrada": "pregunta-uno"}, config=_sesion("a"))
    agente.invoke({"entrada": "pregunta-dos"}, config=_sesion("a"))
    assert "Su severidad era media." in modelo.texto_recibido(1)
    assert "signature" not in modelo.texto_recibido(1)
