import importlib
import sys

from langchain_core.messages import AIMessage

import agente
import herramientas_agente
from agente import construir_ejecutor
from herramientas_agente import DOCUMENTACION, HERRAMIENTAS, calcular
from modelo_falso_herramientas import ModeloFalsoHerramientas, pedir_herramienta


def test_el_agente_usa_la_herramienta_y_responde_con_lo_observado():
    modelo = ModeloFalsoHerramientas(
        guion=[pedir_herramienta("calcular", {"expresion": "340 + 215"}), AIMessage(content="El resultado es 555.")]
    )
    resultado = construir_ejecutor(modelo).invoke({"entrada": "Cuanto es 340 mas 215?"})
    assert resultado["output"] == "El resultado es 555."
    [(accion, observacion)] = resultado["intermediate_steps"]
    assert accion.tool == "calcular"
    assert accion.tool_input == {"expresion": "340 + 215"}
    assert observacion == "555"


def test_una_pregunta_general_se_responde_sin_herramientas():
    modelo = ModeloFalsoHerramientas(guion=[AIMessage(content="Hola.")])
    resultado = construir_ejecutor(modelo).invoke({"entrada": "Hola"})
    assert resultado["intermediate_steps"] == []
    assert len(modelo.entradas) == 1


def test_el_agente_encadena_dos_herramientas_y_razona_con_lo_observado():
    modelo = ModeloFalsoHerramientas(
        guion=[
            pedir_herramienta("buscar_en_documentacion", {"consulta": "reintentos"}),
            pedir_herramienta("calcular", {"expresion": "3 * 2"}),
            AIMessage(content="Fin."),
        ]
    )
    resultado = construir_ejecutor(modelo).invoke({"entrada": "Duplica la espera de los reintentos"})
    assert [accion.tool for accion, _ in resultado["intermediate_steps"]] == ["buscar_en_documentacion", "calcular"]
    assert DOCUMENTACION["reintentos"] in modelo.texto_recibido(1)


def test_una_herramienta_inexistente_no_lanza_y_se_le_avisa_al_modelo():
    modelo = ModeloFalsoHerramientas(guion=[pedir_herramienta("inexistente", {}), AIMessage(content="Fin.")])
    resultado = construir_ejecutor(modelo).invoke({"entrada": "Usa una herramienta inventada"})
    assert resultado["output"] == "Fin."
    # la librería no lanza: entrega una observación de herramienta inválida que el modelo lee en su turno siguiente
    [(accion, observacion)] = resultado["intermediate_steps"]
    assert accion.tool == "inexistente"
    assert "is not a valid tool" in observacion
    assert observacion in modelo.texto_recibido(1)


def test_max_iterations_detiene_al_agente_sin_lanzar():
    modelo = ModeloFalsoHerramientas(guion=[pedir_herramienta("calcular", {"expresion": "1 + 1"})] * 10)
    resultado = construir_ejecutor(modelo, max_iterations=3).invoke({"entrada": "Calcula sin parar"})
    assert resultado["output"] == "Agent stopped due to max iterations."
    assert len(resultado["intermediate_steps"]) <= 3


def test_construir_ejecutor_usa_las_tres_herramientas_o_las_indicadas():
    modelo = ModeloFalsoHerramientas(guion=[])
    assert [herramienta.name for herramienta in construir_ejecutor(modelo).tools] == [
        herramienta.name for herramienta in HERRAMIENTAS
    ]
    assert [herramienta.name for herramienta in construir_ejecutor(modelo, herramientas=[calcular]).tools] == ["calcular"]


def test_importar_los_modulos_no_crea_modelo_ni_requiere_clave(monkeypatch):
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)
    importlib.reload(herramientas_agente)
    importlib.reload(agente)
