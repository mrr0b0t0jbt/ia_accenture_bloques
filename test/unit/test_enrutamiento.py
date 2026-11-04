import importlib
import sys
from unittest.mock import MagicMock

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


def test_ruta_elegida_recibe_el_mensaje_original_y_la_categoria():
    rutas = _rutas_falsas()
    _invocar("incidente", rutas)
    assert rutas["cadena_incidente"].call_args.args[0] == {"mensaje": "mensaje de prueba", "categoria": "incidente"}


def test_importar_el_modulo_no_crea_modelo_ni_requiere_clave(monkeypatch):
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)
    importlib.reload(enrutamiento)
