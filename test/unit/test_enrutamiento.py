import importlib
import logging
import sys
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

import enrutamiento
from enrutamiento import construir_cadena_con_enrutamiento, construir_paso_clasificacion


def _rutas_falsas() -> dict:
    return {
        "cadena_incidente": MagicMock(return_value="respuesta incidente"),
        "cadena_vulnerabilidad": MagicMock(return_value="respuesta vulnerabilidad"),
        "cadena_consulta": MagicMock(return_value="respuesta consulta"),
    }


def _invocar(categoria: str, rutas: dict) -> str:
    paso = MagicMock(return_value=categoria)
    return construir_cadena_con_enrutamiento(paso, **rutas).invoke({"mensaje": "mensaje de prueba"})


def test_categoria_incidente_ejecuta_su_ruta():
    rutas = _rutas_falsas()
    assert _invocar("incidente", rutas) == "respuesta incidente"
    rutas["cadena_incidente"].assert_called_once()
    rutas["cadena_consulta"].assert_not_called()


def test_categoria_vulnerabilidad_ejecuta_su_ruta():
    rutas = _rutas_falsas()
    assert _invocar("vulnerabilidad", rutas) == "respuesta vulnerabilidad"
    rutas["cadena_vulnerabilidad"].assert_called_once()
    rutas["cadena_consulta"].assert_not_called()


def test_categoria_consulta_ejecuta_su_ruta():
    rutas = _rutas_falsas()
    assert _invocar("consulta", rutas) == "respuesta consulta"
    rutas["cadena_consulta"].assert_called_once()
    rutas["cadena_incidente"].assert_not_called()


def test_categoria_inventada_cae_en_la_ruta_por_defecto():
    rutas = _rutas_falsas()
    assert _invocar("categoria_inventada", rutas) == "respuesta consulta"
    rutas["cadena_consulta"].assert_called_once()


def test_clasificacion_normaliza_espacios_y_mayusculas():
    modelo = MagicMock(return_value=AIMessage(content=" Incidente \n"))
    assert construir_paso_clasificacion(modelo).invoke({"mensaje": "mensaje de prueba"}) == "incidente"


@pytest.mark.parametrize(
    "respuesta, esperada",
    [
        ("Incidente.", "incidente"),
        ("**incidente**", "incidente"),
        ("'vulnerabilidad'", "vulnerabilidad"),
        (" Consulta! \n", "consulta"),
    ],
)
def test_clasificacion_quita_la_puntuacion_de_los_extremos(respuesta, esperada):
    modelo = MagicMock(return_value=AIMessage(content=respuesta))
    assert construir_paso_clasificacion(modelo).invoke({"mensaje": "mensaje de prueba"}) == esperada


def test_categoria_con_frase_se_devuelve_tal_cual_y_deja_un_warning(caplog):
    modelo = MagicMock(return_value=AIMessage(content="incidente de seguridad"))
    with caplog.at_level(logging.WARNING, logger="enrutamiento"):
        categoria = construir_paso_clasificacion(modelo).invoke({"mensaje": "mensaje de prueba"})
    assert categoria == "incidente de seguridad"
    assert any(
        registro.levelno == logging.WARNING and "Categoria inesperada" in registro.getMessage()
        for registro in caplog.records
    )


def test_categoria_conocida_no_deja_warning(caplog):
    modelo = MagicMock(return_value=AIMessage(content="incidente"))
    with caplog.at_level(logging.WARNING, logger="enrutamiento"):
        construir_paso_clasificacion(modelo).invoke({"mensaje": "mensaje de prueba"})
    assert not [registro for registro in caplog.records if registro.levelno == logging.WARNING]


def test_ruta_elegida_recibe_el_mensaje_original_y_la_categoria():
    rutas = _rutas_falsas()
    _invocar("incidente", rutas)
    assert rutas["cadena_incidente"].call_args.args[0] == {"mensaje": "mensaje de prueba", "categoria": "incidente"}


def test_plantilla_de_incidente_limita_la_longitud_y_conserva_el_mensaje():
    prompt = enrutamiento.PLANTILLAS_RUTAS["incidente"].invoke({"mensaje": "texto de prueba"}).to_string()
    assert "10 pasos" in prompt
    assert "350 palabras" in prompt
    assert "texto de prueba" in prompt


def test_limite_del_incidente_no_esta_en_las_otras_plantillas_y_las_tres_siguen_distintas():
    plantillas = enrutamiento.PLANTILLAS_RUTAS
    assert "350 palabras" not in plantillas["vulnerabilidad"].template
    assert "350 palabras" not in plantillas["consulta"].template
    assert len({plantilla.template for plantilla in plantillas.values()}) == 3


def test_plantilla_de_vulnerabilidad_limita_la_longitud_y_conserva_el_mensaje():
    prompt = enrutamiento.PLANTILLAS_RUTAS["vulnerabilidad"].invoke({"mensaje": "texto de prueba"}).to_string()
    assert "8 pasos" in prompt
    assert "400 palabras" in prompt
    assert "texto de prueba" in prompt


def test_cada_limite_de_longitud_queda_solo_en_su_plantilla():
    plantillas = enrutamiento.PLANTILLAS_RUTAS
    assert "350 palabras" not in plantillas["consulta"].template
    assert "400 palabras" not in plantillas["consulta"].template
    assert "350 palabras" in plantillas["incidente"].template
    assert "10 pasos" in plantillas["incidente"].template
    assert "350 palabras" not in plantillas["vulnerabilidad"].template


def test_importar_el_modulo_no_crea_modelo_ni_requiere_clave(monkeypatch):
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)
    importlib.reload(enrutamiento)
