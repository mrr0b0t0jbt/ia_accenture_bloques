import logging
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import enrutamiento
from enrutamiento import (
    MENSAJE_ERROR,
    RESPUESTA_GENERICA,
    construir_cadena_con_enrutamiento,
    construir_flujo,
    construir_ruta_segura,
    procesar_mensaje_seguro,
)

ENTRADA = {"mensaje": "mensaje de prueba"}


def _lanzar_error_de_conexion(_entrada: dict) -> str:
    raise ConnectionError("simulado")


RUTA_ROTA = RunnableLambda(_lanzar_error_de_conexion)


def _registros_error(caplog) -> list[logging.LogRecord]:
    return [registro for registro in caplog.records if registro.levelno == logging.ERROR]


def _espiar_generica(monkeypatch) -> MagicMock:
    espia = MagicMock(return_value=RESPUESTA_GENERICA)
    monkeypatch.setattr(enrutamiento, "respuesta_generica", RunnableLambda(espia))
    return espia


def test_ruta_rota_devuelve_la_respuesta_generica_sin_lanzar():
    assert "equipo" in construir_ruta_segura(RUTA_ROTA).invoke(ENTRADA)


def test_con_dos_respaldos_el_segundo_responde_y_la_generica_no_se_ejecuta(monkeypatch):
    generica = _espiar_generica(monkeypatch)
    respaldo_ok = RunnableLambda(lambda _: "respaldo ok")
    assert construir_ruta_segura(RUTA_ROTA, [RUTA_ROTA, respaldo_ok]).invoke(ENTRADA) == "respaldo ok"
    generica.assert_not_called()


def test_si_el_primer_respaldo_funciona_no_se_ejecuta_lo_que_sigue(monkeypatch):
    generica = _espiar_generica(monkeypatch)
    segundo = MagicMock(return_value="segundo")
    ruta = construir_ruta_segura(RUTA_ROTA, [RunnableLambda(lambda _: "primero"), RunnableLambda(segundo)])
    assert ruta.invoke(ENTRADA) == "primero"
    segundo.assert_not_called()
    generica.assert_not_called()


def test_camino_normal_no_ejecuta_el_respaldo():
    respaldo = MagicMock(return_value="respaldo")
    ruta = construir_ruta_segura(RunnableLambda(lambda _: "ruta ok"), [RunnableLambda(respaldo)])
    assert ruta.invoke(ENTRADA) == "ruta ok"
    respaldo.assert_not_called()


def test_clasificacion_rota_devuelve_el_texto_fijo_sin_propagar():
    cadena = construir_cadena_con_enrutamiento(MagicMock(side_effect=ConnectionError("simulado")))
    assert "no pudimos procesar" in procesar_mensaje_seguro(cadena, "mensaje de prueba").lower()


def test_clasificacion_rota_registra_error_con_traceback(caplog):
    cadena = construir_cadena_con_enrutamiento(MagicMock(side_effect=ConnectionError("simulado")))
    with caplog.at_level(logging.ERROR, logger="enrutamiento"):
        procesar_mensaje_seguro(cadena, "mensaje de prueba")
    registro = caplog.records[-1]
    assert registro.levelno == logging.ERROR
    assert "Error procesando mensaje" in registro.getMessage()
    assert registro.exc_info


def test_flujo_completo_el_respaldo_de_la_ruta_actua_antes_del_try_except():
    modelo = MagicMock(side_effect=[AIMessage(content="incidente"), ConnectionError("simulado")])
    assert procesar_mensaje_seguro(construir_flujo(modelo), "mensaje de prueba") == RESPUESTA_GENERICA
    assert modelo.call_count == 2


@pytest.mark.parametrize("tipo_error", [NameError, ValueError, TypeError])
def test_procesar_mensaje_seguro_captura_cualquier_excepcion_y_la_registra(tipo_error, caplog):
    cadena = RunnableLambda(MagicMock(side_effect=tipo_error("simulado")))
    with caplog.at_level(logging.ERROR, logger="enrutamiento"):
        assert procesar_mensaje_seguro(cadena, "mensaje de prueba") == MENSAJE_ERROR
    assert any(registro.levelno == logging.ERROR for registro in caplog.records)


@pytest.mark.parametrize("tipo_error", [ValueError, KeyError])
def test_ruta_segura_responde_generica_ante_errores_que_no_son_de_red(tipo_error):
    ruta_rota = RunnableLambda(MagicMock(side_effect=tipo_error("simulado")))
    assert construir_ruta_segura(ruta_rota).invoke(ENTRADA) == RESPUESTA_GENERICA


def test_error_de_programacion_en_una_ruta_queda_registrado_con_traceback(caplog):
    ruta_rota = RunnableLambda(MagicMock(side_effect=NameError("simulado")))
    with caplog.at_level(logging.ERROR, logger="enrutamiento"):
        assert construir_ruta_segura(ruta_rota).invoke(ENTRADA) == RESPUESTA_GENERICA
    errores = _registros_error(caplog)
    assert len(errores) == 1
    assert "Fallo una ruta" in errores[0].getMessage()
    assert errores[0].exc_info


def test_ruta_y_respaldo_rotos_dejan_dos_registros_y_la_respuesta_generica(caplog):
    with caplog.at_level(logging.ERROR, logger="enrutamiento"):
        assert construir_ruta_segura(RUTA_ROTA, [RUTA_ROTA]).invoke(ENTRADA) == RESPUESTA_GENERICA
    assert len(_registros_error(caplog)) == 2


def test_camino_normal_de_la_ruta_segura_no_registra_errores(caplog):
    with caplog.at_level(logging.ERROR, logger="enrutamiento"):
        assert construir_ruta_segura(RunnableLambda(lambda _: "ruta ok")).invoke(ENTRADA) == "ruta ok"
    assert _registros_error(caplog) == []
