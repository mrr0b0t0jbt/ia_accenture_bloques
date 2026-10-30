import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest

RAIZ = Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location("ejecutar_metricas", RAIZ / "scripts" / "ejecutar_metricas.py")
ejecutar_metricas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ejecutar_metricas)

CASOS = [
    {"consulta": "consulta 1", "referencia": "referencia uno"},
    {"consulta": "consulta 2", "referencia": "referencia dos"},
    {"consulta": "consulta 3", "referencia": "referencia tres"},
]


def _escribir_casos(tmp_path: Path, cantidad: int) -> Path:
    ruta = tmp_path / "casos.json"
    casos = [{"consulta": f"consulta {n}", "referencia": f"referencia {n}"} for n in range(cantidad)]
    ruta.write_text(json.dumps(casos, ensure_ascii=False), encoding="utf-8")
    return ruta


def _inyectar_cliente_falso(monkeypatch, costo: float) -> Mock:
    falso = Mock(side_effect=lambda consulta, system=None: (f"respuesta a {consulta}", costo))
    modulo = types.ModuleType("api_claude_refactor")
    modulo.get_claude_response = falso
    monkeypatch.setitem(sys.modules, "api_claude_refactor", modulo)
    return falso


# --- ejecutar_casos ---

def test_ejecutar_casos_desempaqueta_la_tupla_y_calcula_metricas(tmp_path):
    obtener_respuesta = Mock(return_value=("El gato duerme.", 0.01))
    casos = [{"consulta": "q", "referencia": "El gato duerme en el sofá."}]
    info = ejecutar_metricas.ejecutar_casos(casos, obtener_respuesta, 1.0, tmp_path / "salida.json")
    resultado = info["resultados"][0]
    assert resultado["respuesta"] == "El gato duerme."
    assert resultado["costo"] == pytest.approx(0.01)
    assert info["costo_total"] == pytest.approx(0.01)
    # 3 coincidencias: 3 de 3 tokens de la respuesta, 3 de 6 de la referencia
    assert resultado["precision"] == pytest.approx(1.0)
    assert resultado["cobertura"] == pytest.approx(0.5)
    assert resultado["f1"] == pytest.approx(2 / 3)


def test_ejecutar_casos_guarda_la_salida_despues_de_cada_caso(tmp_path):
    salida = tmp_path / "salida.json"
    llamadas = []

    def obtener_respuesta(consulta):
        llamadas.append(consulta)
        if len(llamadas) == 3:
            assert len(json.loads(salida.read_text(encoding="utf-8"))) == 2
        return "respuesta", 0.001

    info = ejecutar_metricas.ejecutar_casos(CASOS, obtener_respuesta, 1.0, salida)
    assert len(llamadas) == 3
    assert len(json.loads(salida.read_text(encoding="utf-8"))) == len(info["resultados"]) == 3


def test_ejecutar_casos_un_caso_fallido_no_corta_los_siguientes(tmp_path):
    obtener_respuesta = Mock(side_effect=[("uno", 0.01), RuntimeError("simulado"), ("tres", 0.02)])
    info = ejecutar_metricas.ejecutar_casos(CASOS, obtener_respuesta, 1.0, tmp_path / "salida.json")
    assert [r["consulta"] for r in info["resultados"]] == ["consulta 1", "consulta 3"]
    assert info["errores"] == [{"consulta": "consulta 2", "error": "simulado"}]
    assert info["cortado_por_tope"] is False


def test_ejecutar_casos_corta_al_superar_el_tope(tmp_path):
    obtener_respuesta = Mock(return_value=("respuesta", 0.03))
    info = ejecutar_metricas.ejecutar_casos(CASOS, obtener_respuesta, 0.05, tmp_path / "salida.json")
    assert len(info["resultados"]) == 2
    assert info["cortado_por_tope"] is True
    assert info["costo_total"] == pytest.approx(0.06)


def test_ejecutar_casos_un_value_error_no_corta_los_siguientes(tmp_path):
    obtener_respuesta = Mock(side_effect=[ValueError("respuesta inválida"), ("dos", 0.01), ("tres", 0.01)])
    info = ejecutar_metricas.ejecutar_casos(CASOS, obtener_respuesta, 1.0, tmp_path / "salida.json")
    assert info["errores"] == [{"consulta": "consulta 1", "error": "respuesta inválida"}]
    assert [r["consulta"] for r in info["resultados"]] == ["consulta 2", "consulta 3"]


def test_ejecutar_casos_la_salida_conserva_caracteres_no_ascii(tmp_path):
    salida = tmp_path / "salida.json"
    obtener_respuesta = Mock(return_value=("¿Qué? La canción.", 0.001))
    ejecutar_metricas.ejecutar_casos(CASOS[:1], obtener_respuesta, 1.0, salida)
    texto = salida.read_text(encoding="utf-8")
    assert "¿Qué?" in texto
    assert "ó" in texto
    assert "\\u00bf" not in texto


# --- main ---

def test_main_sin_confirmar_api_real_devuelve_2_sin_importar_el_cliente(monkeypatch, capsys, tmp_path):
    # con None en sys.modules, cualquier import del cliente lanzaría ImportError
    monkeypatch.setitem(sys.modules, "api_claude_refactor", None)
    casos = tmp_path / "no_existe.json"
    assert ejecutar_metricas.main(["--casos", str(casos)]) == 2
    assert "--confirmar-api-real" in capsys.readouterr().err


def test_main_salida_en_directorio_inexistente_devuelve_1(monkeypatch, tmp_path):
    falso = _inyectar_cliente_falso(monkeypatch, 0.001)
    casos = _escribir_casos(tmp_path, 1)
    salida = tmp_path / "no_existe" / "salida.json"
    assert ejecutar_metricas.main(["--confirmar-api-real", "--casos", str(casos), "--salida", str(salida)]) == 1
    falso.assert_not_called()


def test_main_tope_mayor_al_maximo_devuelve_1(monkeypatch, tmp_path):
    falso = _inyectar_cliente_falso(monkeypatch, 0.001)
    casos = _escribir_casos(tmp_path, 1)
    salida = tmp_path / "salida.json"
    argv = ["--confirmar-api-real", "--casos", str(casos), "--salida", str(salida), "--tope-usd", "0.06"]
    assert ejecutar_metricas.main(argv) == 1
    falso.assert_not_called()


def test_main_con_mas_de_25_casos_devuelve_1_sin_llamar_al_cliente(monkeypatch, tmp_path):
    falso = _inyectar_cliente_falso(monkeypatch, 0.001)
    casos = _escribir_casos(tmp_path, 26)
    salida = tmp_path / "salida.json"
    assert ejecutar_metricas.main(["--confirmar-api-real", "--casos", str(casos), "--salida", str(salida)]) == 1
    falso.assert_not_called()
    assert not salida.exists()


def test_main_con_exactamente_25_casos_los_ejecuta_todos(monkeypatch, tmp_path):
    falso = _inyectar_cliente_falso(monkeypatch, 0.0001)
    casos = _escribir_casos(tmp_path, 25)
    salida = tmp_path / "salida.json"
    assert ejecutar_metricas.main(["--confirmar-api-real", "--casos", str(casos), "--salida", str(salida)]) == 0
    assert falso.call_count == 25


def test_main_tope_usd_maximo_es_exactamente_0_05(monkeypatch, tmp_path):
    assert ejecutar_metricas.TOPE_USD_MAXIMO == 0.05
    falso = _inyectar_cliente_falso(monkeypatch, 0.001)
    casos = _escribir_casos(tmp_path, 1)
    salida = tmp_path / "salida.json"
    argv = ["--confirmar-api-real", "--casos", str(casos), "--salida", str(salida), "--tope-usd"]
    assert ejecutar_metricas.main(argv + ["0.051"]) == 1
    falso.assert_not_called()
    assert ejecutar_metricas.main(argv + ["0.05"]) == 0


@pytest.mark.parametrize("tope", ["0", "-0.01"])
def test_main_tope_no_positivo_devuelve_1(monkeypatch, tmp_path, tope):
    falso = _inyectar_cliente_falso(monkeypatch, 0.001)
    casos = _escribir_casos(tmp_path, 1)
    salida = tmp_path / "salida.json"
    argv = ["--confirmar-api-real", "--casos", str(casos), "--salida", str(salida), "--tope-usd", tope]
    assert ejecutar_metricas.main(argv) == 1
    falso.assert_not_called()


def test_main_salida_vacia_devuelve_1(monkeypatch, tmp_path):
    falso = _inyectar_cliente_falso(monkeypatch, 0.001)
    casos = _escribir_casos(tmp_path, 1)
    assert ejecutar_metricas.main(["--confirmar-api-real", "--casos", str(casos), "--salida", ""]) == 1
    falso.assert_not_called()


def test_main_con_cliente_falso_escribe_la_salida(monkeypatch, capsys, tmp_path):
    falso = _inyectar_cliente_falso(monkeypatch, 0.001)
    casos = _escribir_casos(tmp_path, 2)
    salida = tmp_path / "salida.json"
    assert ejecutar_metricas.main(["--confirmar-api-real", "--casos", str(casos), "--salida", str(salida)]) == 0
    escritos = json.loads(salida.read_text(encoding="utf-8"))
    assert [e["consulta"] for e in escritos] == ["consulta 0", "consulta 1"]
    assert escritos[0]["respuesta"] == "respuesta a consulta 0"
    assert escritos[0]["costo"] == pytest.approx(0.001)
    assert set(escritos[0]) == {"consulta", "referencia", "respuesta", "precision", "cobertura", "f1", "costo"}
    assert falso.call_args.kwargs["system"] == "Responde en una sola frase."
    assert "Costo total" in capsys.readouterr().out


def test_main_respeta_el_tope_recibido(monkeypatch, tmp_path):
    _inyectar_cliente_falso(monkeypatch, 0.001)
    casos = _escribir_casos(tmp_path, 5)
    salida = tmp_path / "salida.json"
    argv = ["--confirmar-api-real", "--casos", str(casos), "--salida", str(salida), "--tope-usd", "0.002"]
    assert ejecutar_metricas.main(argv) == 0
    assert len(json.loads(salida.read_text(encoding="utf-8"))) == 2
