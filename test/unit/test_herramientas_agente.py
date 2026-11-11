import os
from unittest.mock import MagicMock

import pydantic
import pytest

from herramientas_agente import (
    CVES_LABORATORIO,
    DOCUMENTACION,
    HERRAMIENTAS,
    RIESGO,
    buscar_en_documentacion,
    calcular,
    construir_herramientas,
    consultar_cve,
    limpiar_motivo,
)
from supervision import RegistroAprobaciones

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


@pytest.mark.parametrize("expresion", ["1e308 * 10", "1e999 - 1e999"])
def test_calcular_rechaza_resultados_no_finitos(expresion):
    assert calcular.invoke({"expresion": expresion}) == "Resultado fuera de rango."


@pytest.mark.parametrize(
    "expresion, esperado",
    [(" 1 + 1 ", "2"), ("", "Expresión no permitida."), ("   ", "Expresión no permitida.")],
    ids=["espacios_en_los_bordes", "vacia", "solo_espacios"],
)
def test_calcular_tolera_espacios_en_los_bordes_y_la_expresion_vacia(expresion, esperado):
    assert calcular.invoke({"expresion": expresion}) == esperado


@pytest.mark.parametrize(
    "expresion, esperado",
    [
        ("1e23", "1e+23"),
        ("1e15", "1000000000000000"),
        ("9007199254740993 * 1.0", "9007199254740992.0"),
        ("9" * 30, "9" * 30),
    ],
    ids=["float_enorme", "float_entero_seguro", "float_en_2_a_la_53", "entero_enorme"],
)
def test_calcular_no_inventa_digitos_de_ruido_en_numeros_grandes(expresion, esperado):
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


def test_cve_desconocido_muy_largo_refleja_solo_el_inicio_del_identificador():
    respuesta = consultar_cve.invoke({"cve_id": "a" * 200})
    assert respuesta.startswith("No se encontró el CVE de laboratorio")
    assert len(respuesta) < 140
    assert "…" in respuesta


def test_cve_desconocido_de_64_caracteres_se_refleja_completo():
    respuesta = consultar_cve.invoke({"cve_id": "a" * 64})
    assert respuesta == f"No se encontró el CVE de laboratorio {'A' * 64}."


def test_cve_con_argumento_invalido_falla_en_la_validacion():
    with pytest.raises(pydantic.ValidationError):
        consultar_cve.invoke({"cve_id": CVE_CONOCIDO, "incluir_mitigacion": "tal vez"})


def test_cve_tiene_nombre_propio_y_campos_descritos():
    assert consultar_cve.name == "consultar_cve_de_laboratorio"
    assert all(campo.description for campo in consultar_cve.args_schema.model_fields.values())


def test_las_tres_herramientas_tienen_descripcion_y_devuelven_texto():
    argumentos = [{"consulta": "costo"}, {"expresion": "1 + 1"}, {"cve_id": CVE_CONOCIDO}]
    # solo las tres de riesgo bajo: la cuarta pide aprobación y tiene sus propias pruebas
    for herramienta, entrada in zip(HERRAMIENTAS[:3], argumentos, strict=True):
        assert herramienta.description
        assert isinstance(herramienta.invoke(entrada), str)


HOST_VALIDO = "host-lab-01.ejemplo.test"


class _AprobadorEspia:
    """Aprobador de prueba que responde lo indicado y guarda los mensajes que recibió."""

    def __init__(self, respuesta: bool) -> None:
        self.respuesta = respuesta
        self.mensajes: list[str] = []

    def __call__(self, mensaje: str) -> bool:
        self.mensajes.append(mensaje)
        return self.respuesta


def _aislar(respuesta: bool = True):
    """Devuelve (herramienta, aprobador espía, registro) con la herramienta de aislamiento ya construida."""
    aprobador, registro = _AprobadorEspia(respuesta), RegistroAprobaciones()
    return construir_herramientas(aprobador, registro)[3], aprobador, registro


def test_riesgo_cubre_exactamente_las_herramientas_con_niveles_validos():
    assert set(RIESGO) == {herramienta.name for herramienta in HERRAMIENTAS}
    assert set(RIESGO.values()) <= {"bajo", "medio", "alto"}
    assert [RIESGO[nombre] for nombre in ("buscar_en_documentacion", "calcular", "consultar_cve_de_laboratorio")] == ["bajo"] * 3
    assert RIESGO["aislar_host_de_laboratorio"] == "alto"


def test_aislar_con_aprobacion_simula_el_aislamiento_y_registra_la_decision():
    herramienta, _, registro = _aislar(True)
    respuesta = herramienta.invoke({"host": HOST_VALIDO, "motivo": "beaconing"})
    assert "aislado" in respuesta and "simulado" in respuesta
    [entrada] = registro.registros
    assert entrada["aprobado"] is True
    assert entrada["argumentos"] == {"host": HOST_VALIDO, "motivo": "beaconing"}


def test_aislar_sin_aprobacion_no_aisla_y_registra_la_negativa():
    herramienta, _, registro = _aislar(False)
    assert "no aprobado" in herramienta.invoke({"host": HOST_VALIDO, "motivo": "beaconing"})
    [entrada] = registro.registros
    assert entrada["aprobado"] is False


@pytest.mark.parametrize(
    "host", ["produccion.empresa.com", "host-lab-01.ejemplo.test.evil.com", "ejemplo.test"]
)
def test_aislar_rechaza_hosts_que_no_son_de_laboratorio_sin_pedir_aprobacion_y_registra_el_intento(host):
    herramienta, aprobador, registro = _aislar(True)
    assert herramienta.invoke({"host": host, "motivo": "x"}).startswith("Host no permitido")
    assert aprobador.mensajes == []
    [entrada] = registro.registros
    assert entrada["aprobado"] is False
    assert entrada["argumentos"] == {"host": host, "motivo": "x", "rechazo": "host no permitido"}


def test_aislar_registra_un_host_rechazado_largo_y_con_controles_limpio_y_sin_llamar_al_aprobador():
    herramienta, aprobador, registro = _aislar(True)
    herramienta.invoke({"host": "a\n\x1b[2K" * 60, "motivo": "x"})
    [entrada] = registro.registros
    host = entrada["argumentos"]["host"]
    assert entrada["aprobado"] is False and entrada["argumentos"]["rechazo"] == "host no permitido"
    assert 0 < len(host) <= 100 and "\n" not in host and "\x1b" not in host
    assert aprobador.mensajes == []


def test_aislar_con_un_aprobador_que_falla_propaga_la_excepcion_sin_aislar_ni_registrar():
    registro = RegistroAprobaciones()

    def aprobador_roto(mensaje: str) -> bool:
        raise RuntimeError("fallo del aprobador")

    herramienta = construir_herramientas(aprobador_roto, registro)[3]
    with pytest.raises(RuntimeError):
        herramienta.invoke({"host": HOST_VALIDO, "motivo": "x"})
    assert registro.registros == []


@pytest.mark.parametrize(
    "host, aceptado",
    [("a" * 87 + ".ejemplo.test", True), ("a" * 88 + ".ejemplo.test", False)],
    ids=["100_caracteres", "101_caracteres"],
)
def test_aislar_aplica_el_limite_de_100_caracteres_del_host(host, aceptado):
    herramienta, aprobador, registro = _aislar(True)
    respuesta = herramienta.invoke({"host": host, "motivo": "x"})
    assert (len(aprobador.mensajes), len(registro.registros)) == ((1, 1) if aceptado else (0, 1))
    assert respuesta.startswith("Host no permitido") is not aceptado


def test_aislar_pide_aprobacion_con_un_mensaje_que_trae_host_y_motivo():
    herramienta, aprobador, _ = _aislar(True)
    herramienta.invoke({"host": HOST_VALIDO, "motivo": "beaconing"})
    [mensaje] = aprobador.mensajes
    assert HOST_VALIDO in mensaje and "beaconing" in mensaje
    assert mensaje.endswith("(s/n): ")


def _mensaje_con_motivo(motivo: str) -> str:
    herramienta, aprobador, _ = _aislar(True)
    herramienta.invoke({"host": HOST_VALIDO, "motivo": motivo})
    return aprobador.mensajes[0]


def test_aislar_no_deja_saltos_de_linea_ni_retorno_de_carro_en_el_mensaje():
    mensaje = _mensaje_con_motivo("a\nb\rc")
    assert "\n" not in mensaje and "\r" not in mensaje


def test_aislar_no_deja_secuencias_ansi_en_el_mensaje():
    assert "\x1b" not in _mensaje_con_motivo("fuga\x1b[2KAPROBACION PREVIA")


def test_aislar_no_deja_marcas_bidireccionales_en_el_mensaje():
    mensaje = _mensaje_con_motivo("a\u202eb\u2066c")
    assert "\u202e" not in mensaje and "\u2066" not in mensaje


def test_aislar_muestra_el_motivo_entre_comillas_tras_la_etiqueta_de_no_verificado():
    mensaje = _mensaje_con_motivo("beaconing")
    assert 'Motivo (texto del modelo, no verificado): "beaconing"' in mensaje
    assert mensaje.endswith("(s/n): ")


@pytest.mark.parametrize("motivo", ["", "  \n\t  "], ids=["vacio", "solo_blancos"])
def test_aislar_con_motivo_vacio_usa_sin_motivo_en_el_mensaje_y_el_registro(motivo):
    herramienta, aprobador, registro = _aislar(True)
    herramienta.invoke({"host": HOST_VALIDO, "motivo": motivo})
    assert '"(sin motivo)"' in aprobador.mensajes[0]
    assert registro.registros[0]["argumentos"]["motivo"] == "(sin motivo)"


def test_aislar_registra_un_motivo_largo_con_saltos_de_linea_sin_saltos_y_de_a_lo_mas_200():
    herramienta, _, registro = _aislar(True)
    herramienta.invoke({"host": HOST_VALIDO, "motivo": "ab\n" * 170})
    motivo = registro.registros[0]["argumentos"]["motivo"]
    assert 0 < len(motivo) <= 200 and "\n" not in motivo


def test_limpiar_motivo_colapsa_los_espacios_y_tabuladores():
    assert limpiar_motivo("a   b\t\tc") == "a b c"


def test_limpiar_motivo_reemplaza_el_escape_por_un_espacio():
    assert limpiar_motivo("x\x1b[2Ky") == "x [2Ky"


def test_limpiar_motivo_conserva_acentos_y_enes():
    assert limpiar_motivo("tráfico de señalización") == "tráfico de señalización"


@pytest.mark.parametrize(
    "motivo, esperado",
    [("a" + " " * 300 + "b", "a b"), ("x" * 199 + " y", "x" * 199)],
    ids=["colapsa_antes_de_recortar", "strip_despues_de_recortar"],
)
def test_limpiar_motivo_sanea_colapsa_recorta_y_hace_strip_en_ese_orden(motivo, esperado):
    assert limpiar_motivo(motivo) == esperado


def test_aislar_recorta_el_motivo_a_200_caracteres_en_el_mensaje_y_en_el_registro():
    herramienta, aprobador, registro = _aislar(True)
    herramienta.invoke({"host": HOST_VALIDO, "motivo": "m" * 500})
    assert "m" * 200 in aprobador.mensajes[0] and "m" * 201 not in aprobador.mensajes[0]
    assert registro.registros[0]["argumentos"]["motivo"] == "m" * 200


def test_aislar_normaliza_el_host_antes_de_validarlo_y_registrarlo():
    herramienta, _, registro = _aislar(True)
    assert "aislado" in herramienta.invoke({"host": " HOST-LAB-01.EJEMPLO.TEST ", "motivo": "x"})
    assert registro.registros[0]["argumentos"]["host"] == HOST_VALIDO


def test_construir_herramientas_conserva_el_orden_y_la_global_niega_por_defecto():
    nombres = [herramienta.name for herramienta in construir_herramientas()]
    assert nombres == [
        "buscar_en_documentacion",
        "calcular",
        "consultar_cve_de_laboratorio",
        "aislar_host_de_laboratorio",
    ]
    assert "no aprobado" in HERRAMIENTAS[3].invoke({"host": HOST_VALIDO, "motivo": "x"})
