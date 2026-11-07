import os
from unittest.mock import MagicMock

import pydantic
import pytest

from herramientas_agente import (
    CVES_LABORATORIO,
    DOCUMENTACION,
    HERRAMIENTAS,
    buscar_en_documentacion,
    calcular,
    consultar_cve,
)

CVE_CONOCIDO = "CVE-2099-10002"


def test_calcular_suma_devuelve_el_entero_sin_decimales():
    assert calcular.invoke({"expresion": "340 + 215"}) == "555"


def test_calcular_respeta_los_parentesis():
    assert calcular.invoke({"expresion": "2 * (3 + 4)"}) == "14"


def test_calcular_division_no_entera_devuelve_el_decimal():
    assert calcular.invoke({"expresion": "10 / 4"}) == "2.5"


@pytest.mark.parametrize("expresion", ["__import__('os').system('ls')", "2 ** 10", "9" * 150])
def test_calcular_rechaza_lo_no_permitido_sin_ejecutarlo(expresion, monkeypatch):
    sistema = MagicMock()
    monkeypatch.setattr(os, "system", sistema)
    assert calcular.invoke({"expresion": expresion}) == "Expresión no permitida."
    sistema.assert_not_called()


@pytest.mark.parametrize(
    "expresion, verificar",
    [
        ("10 / 0", lambda texto: "cero" in texto),
        ("340 +", lambda texto: texto.startswith("Error al evaluar")),
    ],
)
def test_calcular_devuelve_texto_ante_errores_de_evaluacion(expresion, verificar):
    assert verificar(calcular.invoke({"expresion": expresion}))


@pytest.mark.parametrize("expresion", ["True + 1", "1j", "abs(1)"])
def test_calcular_rechaza_booleanos_complejos_y_llamadas(expresion):
    assert calcular.invoke({"expresion": expresion}) == "Expresión no permitida."


@pytest.mark.parametrize(
    "expresion, esperado",
    [("1+" * 49 + "11", "60"), ("1+" * 50 + "1", "Expresión no permitida.")],
    ids=["100_caracteres", "101_caracteres"],
)
def test_calcular_aplica_el_limite_de_longitud_exacto(expresion, esperado):
    assert calcular.invoke({"expresion": expresion}) == esperado


@pytest.mark.parametrize(
    "expresion, esperado",
    [("-5 + +2", "-3"), ("7 // 2", "3"), ("7 % 3", "1"), ("10 / 2", "5")],
)
def test_calcular_resuelve_unarios_division_entera_modulo_y_division_exacta(expresion, esperado):
    assert calcular.invoke({"expresion": expresion}) == esperado


def test_buscar_devuelve_el_texto_de_reintentos():
    respuesta = buscar_en_documentacion.invoke({"consulta": "reintentos"})
    assert respuesta == DOCUMENTACION["reintentos"]
    assert "2 s" in respuesta


def test_buscar_devuelve_el_texto_de_costo():
    assert buscar_en_documentacion.invoke({"consulta": "como se calcula el costo"}) == DOCUMENTACION["costo"]


def test_buscar_sin_coincidencia_avisa_que_no_encontro():
    assert "No se encontró información" in buscar_en_documentacion.invoke({"consulta": "tema ajeno"})


def test_buscar_es_insensible_a_mayusculas():
    consulta = "¿Cómo funcionan los REINTENTOS?"
    assert buscar_en_documentacion.invoke({"consulta": consulta}) == DOCUMENTACION["reintentos"]


def test_cve_conocido_sin_mitigacion_trae_la_severidad_y_no_la_mitigacion():
    respuesta = consultar_cve.invoke({"cve_id": CVE_CONOCIDO})
    assert CVES_LABORATORIO[CVE_CONOCIDO]["severidad"] in respuesta
    assert CVES_LABORATORIO[CVE_CONOCIDO]["mitigacion"] not in respuesta


def test_cve_conocido_con_mitigacion_la_incluye():
    respuesta = consultar_cve.invoke({"cve_id": CVE_CONOCIDO, "incluir_mitigacion": True})
    assert CVES_LABORATORIO[CVE_CONOCIDO]["mitigacion"] in respuesta


def test_cve_se_normaliza_antes_de_buscarlo():
    respuesta = consultar_cve.invoke({"cve_id": " cve-2099-10001 "})
    assert CVES_LABORATORIO["CVE-2099-10001"]["resumen"] in respuesta


def test_cve_desconocido_avisa_que_no_se_encontro():
    assert "No se encontró el CVE de laboratorio" in consultar_cve.invoke({"cve_id": "CVE-2099-99999"})


def test_cve_con_argumento_invalido_falla_en_la_validacion():
    with pytest.raises(pydantic.ValidationError):
        consultar_cve.invoke({"cve_id": CVE_CONOCIDO, "incluir_mitigacion": "tal vez"})


def test_cve_tiene_nombre_propio_y_campos_descritos():
    assert consultar_cve.name == "consultar_cve_de_laboratorio"
    assert all(campo.description for campo in consultar_cve.args_schema.model_fields.values())


def test_las_tres_herramientas_tienen_descripcion_y_devuelven_texto():
    argumentos = [{"consulta": "costo"}, {"expresion": "1 + 1"}, {"cve_id": CVE_CONOCIDO}]
    for herramienta, entrada in zip(HERRAMIENTAS, argumentos, strict=True):
        assert herramienta.description
        assert isinstance(herramienta.invoke(entrada), str)
