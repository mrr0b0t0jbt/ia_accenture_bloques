import importlib.util
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from pipeline_texto import comparar_antes_despues

RAIZ = Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location("ejecutar_pipeline", RAIZ / "scripts" / "ejecutar_pipeline.py")
ejecutar_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ejecutar_pipeline)

CASOS = [
    {"consulta": "consulta 1", "referencia": "referencia uno"},
    {"consulta": "consulta 2", "referencia": "referencia dos"},
    {"consulta": "consulta 3", "referencia": "referencia tres"},
]


# --- comparar_antes_despues ---

def test_comparar_antes_despues_cielo_azul():
    resultado = comparar_antes_despues("El cielo es azul.", "el cielo es azul")
    assert resultado["pct_crudo"] == pytest.approx(0.5)
    assert resultado["pct_procesado"] == pytest.approx(1.0)


def test_comparar_antes_despues_respuesta_vacia():
    resultado = comparar_antes_despues("El cielo es azul.", "")
    assert resultado["pct_crudo"] == 0.0
    assert resultado["pct_procesado"] == 0.0


# --- ejecutar_casos ---

def test_ejecutar_casos_desempaqueta_la_tupla_de_get_claude_response():
    obtener_respuesta = Mock(return_value=("texto", 0.01))
    info = ejecutar_pipeline.ejecutar_casos([{"consulta": "q", "referencia": "Texto."}], obtener_respuesta, 1.0)
    resultado = info["resultados"][0]
    assert resultado["respuesta"] == "texto"
    assert resultado["costo"] == pytest.approx(0.01)
    assert info["costo_total"] == pytest.approx(0.01)
    # "Texto." vs "texto": crudo no coincide, procesado sí
    assert resultado["pct_crudo"] == pytest.approx(0.0)
    assert resultado["pct_procesado"] == pytest.approx(1.0)


def test_ejecutar_casos_un_caso_fallido_no_corta_los_siguientes():
    obtener_respuesta = Mock(side_effect=[("uno", 0.01), RuntimeError("simulado"), ("tres", 0.02)])
    info = ejecutar_pipeline.ejecutar_casos(CASOS, obtener_respuesta, 1.0)
    assert len(info["resultados"]) == 2
    assert info["errores"] == [{"consulta": "consulta 2", "error": "simulado"}]
    assert info["costo_total"] == pytest.approx(0.03)
    assert info["cortado_por_tope"] is False


def test_ejecutar_casos_respeta_el_tope_de_gasto():
    obtener_respuesta = Mock(return_value=("respuesta", 0.03))
    info = ejecutar_pipeline.ejecutar_casos(CASOS, obtener_respuesta, 0.05)
    assert len(info["resultados"]) == 2
    assert info["cortado_por_tope"] is True

    info = ejecutar_pipeline.ejecutar_casos(CASOS, obtener_respuesta, 1.0)
    assert len(info["resultados"]) == 3
    assert info["cortado_por_tope"] is False


def test_ejecutar_casos_corta_cuando_el_costo_es_igual_al_tope():
    obtener_respuesta = Mock(return_value=("r", 0.05))
    info = ejecutar_pipeline.ejecutar_casos(CASOS, obtener_respuesta, 0.05)
    assert len(info["resultados"]) == 1
    assert obtener_respuesta.call_count == 1
    assert info["cortado_por_tope"] is True


def test_ejecutar_casos_registra_value_error_como_error_del_caso():
    obtener_respuesta = Mock(side_effect=[("uno", 0.01), ValueError("valor"), ("tres", 0.02)])
    info = ejecutar_pipeline.ejecutar_casos(CASOS, obtener_respuesta, 1.0)
    assert len(info["resultados"]) == 2
    assert info["errores"] == [{"consulta": "consulta 2", "error": "valor"}]
    assert info["costo_total"] == pytest.approx(0.03)


# --- formatear_registro ---

def test_formatear_registro_incluye_porcentajes_y_aviso_de_llamada_real():
    info = {
        "resultados": [{
            "consulta": "¿Qué es X?",
            "referencia": "X es algo.",
            "respuesta": "X es\nalgo",
            "pct_crudo": 0.5714,
            "pct_procesado": 1.0,
            "costo": 0.001,
        }],
        "costo_total": 0.001,
        "cortado_por_tope": False,
        "errores": [],
    }
    registro = ejecutar_pipeline.formatear_registro(info, "2026-10-28")
    assert "[Caso 1] Coincidencia texto crudo: 57.1%" in registro
    assert "[Caso 1] Coincidencia texto procesado: 100.0%" in registro
    assert "REALES" in registro
    assert "2026-10-28" in registro


# --- cargar_casos ---

def test_cargar_casos_lee_utf8_con_tildes(tmp_path):
    casos = [{"consulta": "¿Qué año?", "referencia": "El año pequeño, canción."}]
    ruta = tmp_path / "casos.json"
    ruta.write_text(json.dumps(casos, ensure_ascii=False), encoding="utf-8")
    assert ejecutar_pipeline.cargar_casos(ruta) == casos


def test_casos_pipeline_json_tiene_tres_casos_con_consulta_y_referencia():
    casos = ejecutar_pipeline.cargar_casos(RAIZ / "JSON" / "casos_pipeline.json")
    assert len(casos) == 3
    for caso in casos:
        assert caso["consulta"]
        assert caso["referencia"]


# --- main ---

def test_main_sin_confirmar_api_real_devuelve_2_sin_ejecutar_casos(monkeypatch, capsys):
    falso = Mock()
    monkeypatch.setattr(ejecutar_pipeline, "ejecutar_casos", falso)
    assert ejecutar_pipeline.main([]) == 2
    assert capsys.readouterr().err
    falso.assert_not_called()


def test_main_salida_en_directorio_inexistente_devuelve_1_sin_llamar_a_la_api(monkeypatch, capsys, tmp_path):
    falso = Mock()
    monkeypatch.setattr(ejecutar_pipeline, "ejecutar_casos", falso)
    salida = tmp_path / "no_existe" / "registro.md"
    assert ejecutar_pipeline.main(["--confirmar-api-real", "--salida", str(salida)]) == 1
    assert capsys.readouterr().err
    falso.assert_not_called()
    assert not (tmp_path / "no_existe").exists()
