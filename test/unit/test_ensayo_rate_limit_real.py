import importlib.util
import json
import threading
from pathlib import Path
from unittest.mock import patch, Mock

import pytest

RAIZ = Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location("ensayo_rate_limit_real", RAIZ / "scripts" / "ensayo_rate_limit_real.py")
ensayo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ensayo)

RESPUESTA_OK = {
    "content": [{"type": "text", "text": "OK"}],
    "usage": {"input_tokens": 1000, "output_tokens": 500},
}


@pytest.fixture(autouse=True)
def clave_falsa(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "clave-falsa-de-prueba")


def respuesta_falsa(status_code):
    respuesta = Mock(status_code=status_code, ok=status_code < 400)
    respuesta.json.return_value = RESPUESTA_OK
    return respuesta


def info_a_mano(primer_429, total_429):
    return {
        "peticiones_enviadas": 5,
        "llamadas_http": 5 + total_429,
        "primer_429": primer_429,
        "total_429": total_429,
        "esperas": [2.1] * total_429,
        "fallos_definitivos": 0,
        "costo_total": 0.0012,
        "cortado_por_tope": False,
        "duracion_s": 3.2,
    }


# --- ejecutar_ensayo ---

def test_ejecutar_ensayo_registra_el_primer_429_y_el_total():
    respuestas = [respuesta_falsa(c) for c in (200, 200, 429, 429, 200, 200)]
    with patch("api_claude_refactor.requests.post", side_effect=respuestas), \
            patch("api_claude_refactor.time.sleep"):
        info = ensayo.ejecutar_ensayo(4, 1.0)
    assert info["primer_429"] == 3
    assert info["total_429"] == 2
    assert info["llamadas_http"] == 6
    assert info["peticiones_enviadas"] == 4
    assert info["fallos_definitivos"] == 0
    assert info["costo_total"] == pytest.approx(4 * 0.007)


def test_ejecutar_ensayo_sin_429_informa_que_no_se_alcanzo():
    respuestas = [respuesta_falsa(200) for _ in range(3)]
    with patch("api_claude_refactor.requests.post", side_effect=respuestas), \
            patch("api_claude_refactor.time.sleep"):
        info = ensayo.ejecutar_ensayo(3, 1.0)
    assert info["primer_429"] is None
    assert info["total_429"] == 0
    assert "no se alcanzó ningún 429 en 3 peticiones" in ensayo.formatear_registro(info, "2026-10-28")


def test_ejecutar_ensayo_corta_al_superar_el_tope_de_gasto():
    enviar = Mock(return_value=("OK", 0.03))
    info = ensayo.ejecutar_ensayo(5, 0.05, enviar=enviar)
    assert info["peticiones_enviadas"] == 2
    assert info["cortado_por_tope"] is True
    assert enviar.call_count == 2


def test_ejecutar_ensayo_registra_las_esperas_pedidas_a_time_sleep():
    respuestas = [respuesta_falsa(c) for c in (429, 429, 200)]
    with patch("api_claude_refactor.requests.post", side_effect=respuestas), \
            patch("api_claude_refactor.time.sleep") as mock_sleep:
        info = ensayo.ejecutar_ensayo(1, 1.0)
    assert len(info["esperas"]) == 2
    assert 2 <= info["esperas"][0] <= 2.2
    assert 4 <= info["esperas"][1] <= 4.4
    # el espía delega en el sleep que estaba vigente (aquí el mock)
    assert mock_sleep.call_count == 2


def test_ejecutar_ensayo_n_fuera_de_rango_lanza_value_error():
    with pytest.raises(ValueError):
        ensayo.ejecutar_ensayo(31, 1.0)
    with pytest.raises(ValueError):
        ensayo.ejecutar_ensayo(0, 1.0)


def test_ejecutar_ensayo_cuenta_los_fallos_definitivos():
    enviar = Mock(side_effect=[("OK", 0.01), RuntimeError("x"), ("OK", 0.01), RuntimeError("x"), ("OK", 0.01)])
    info = ensayo.ejecutar_ensayo(5, 1.0, enviar=enviar)
    assert info["fallos_definitivos"] == 2
    assert info["peticiones_enviadas"] == 5
    assert info["costo_total"] == pytest.approx(0.03)
    assert info["cortado_por_tope"] is False


def test_ensayo_concurrente_registra_el_primer_429_y_el_total():
    lock = threading.Lock()
    vistas = set()
    llamadas = [0]

    def fake(*args, **kwargs):
        peticion = int(json.loads(kwargs["data"])["messages"][0]["content"].split()[-1])
        with lock:
            llamadas[0] += 1
            primera_vez = peticion not in vistas
            vistas.add(peticion)
        return respuesta_falsa(429 if primera_vez and peticion in (2, 5) else 200)

    with patch("api_claude_refactor.requests.post", side_effect=fake), \
            patch("api_claude_refactor.time.sleep"):
        info = ensayo.ejecutar_ensayo(6, 1.0, concurrencia=3)
    assert info["total_429"] == 2
    assert info["primer_429"] == 2
    assert info["llamadas_http"] == llamadas[0] == 8
    assert info["olas"] == 2
    assert info["peticiones_enviadas"] == 6
    assert info["fallos_definitivos"] == 0
    assert len(info["esperas"]) == 2


def test_ensayo_concurrente_corta_entre_olas_al_superar_el_tope():
    enviar = Mock(return_value=("OK", 0.03))
    info = ensayo.ejecutar_ensayo(10, 0.1, enviar=enviar, concurrencia=2)
    assert info["peticiones_enviadas"] == 4
    assert info["cortado_por_tope"] is True
    assert info["costo_total"] == pytest.approx(0.12)
    assert info["olas"] == 2


def test_ensayo_concurrente_valida_n_y_concurrencia():
    enviar = Mock(return_value=("OK", 0.0))
    for concurrencia in (0, 21):
        with pytest.raises(ValueError):
            ensayo.ejecutar_ensayo(5, 1.0, enviar=enviar, concurrencia=concurrencia)
    for n in (101, 0):
        with pytest.raises(ValueError):
            ensayo.ejecutar_ensayo(n, 1.0, enviar=enviar, concurrencia=5)
    with pytest.raises(ValueError):
        ensayo.ejecutar_ensayo(31, 1.0, enviar=enviar, concurrencia=1)
    assert ensayo.ejecutar_ensayo(100, 1.0, enviar=enviar, concurrencia=5)["peticiones_enviadas"] == 100
    ensayo.ejecutar_ensayo(20, 1.0, enviar=enviar, concurrencia=20)


def test_ensayo_concurrente_los_contadores_coinciden_con_las_llamadas_reales():
    lock = threading.Lock()
    contador = {"llamadas": 0, "429": 0}

    def fake(*args, **kwargs):
        with lock:
            contador["llamadas"] += 1
            es_429 = contador["llamadas"] % 3 == 0
            if es_429:
                contador["429"] += 1
        return respuesta_falsa(429 if es_429 else 200)

    with patch("api_claude_refactor.requests.post", side_effect=fake), \
            patch("api_claude_refactor.time.sleep"):
        info = ensayo.ejecutar_ensayo(60, 1.0, concurrencia=20)
    assert info["llamadas_http"] == contador["llamadas"]
    assert info["total_429"] == contador["429"]
    assert info["peticiones_enviadas"] == 60
    assert info["olas"] == 3


# --- main ---

def test_main_sin_confirmar_api_real_devuelve_2_sin_ejecutar_el_ensayo(monkeypatch, capsys):
    falso = Mock()
    monkeypatch.setattr(ensayo, "ejecutar_ensayo", falso)
    assert ensayo.main([]) == 2
    assert capsys.readouterr().err
    falso.assert_not_called()


def test_main_salida_en_directorio_inexistente_devuelve_1_sin_llamar_a_la_api(monkeypatch, capsys, tmp_path):
    falso = Mock()
    monkeypatch.setattr(ensayo, "ejecutar_ensayo", falso)
    salida = tmp_path / "no_existe" / "registro.md"
    assert ensayo.main(["--confirmar-api-real", "--salida", str(salida)]) == 1
    assert capsys.readouterr().err
    falso.assert_not_called()
    assert not (tmp_path / "no_existe").exists()


def test_main_con_concurrencia_sin_confirmar_api_real_devuelve_2(monkeypatch, capsys):
    falso = Mock()
    monkeypatch.setattr(ensayo, "ejecutar_ensayo", falso)
    assert ensayo.main(["--concurrencia", "5"]) == 2
    assert capsys.readouterr().err
    falso.assert_not_called()


# --- formatear_registro ---

def test_formatear_registro_incluye_el_aviso_de_ensayo_real():
    sin_429 = ensayo.formatear_registro(info_a_mano(None, 0), "2026-10-28")
    con_429 = ensayo.formatear_registro(info_a_mano(3, 2), "2026-10-28")
    for registro in (sin_429, con_429):
        assert "> **Aviso:**" in registro
        assert "REAL" in registro
        assert "2026-10-28" in registro
    assert "no se alcanzó ningún 429 en 5 peticiones" in sin_429
    assert "petición 3" in con_429
    assert "2 respuestas 429" in con_429


def test_formatear_registro_concurrente_incluye_concurrencia_y_aviso():
    info = dict(info_a_mano(None, 0), concurrencia=5, olas=4)
    registro = ensayo.formatear_registro(info, "2026-10-28")
    assert "- Concurrencia: 5 (olas: 4)" in registro
    assert "ráfaga concurrente deliberada" in registro
    assert "no se alcanzó ningún 429 en 5 peticiones" in registro
    secuencial = ensayo.formatear_registro(info_a_mano(None, 0), "2026-10-28")
    assert "Concurrencia" not in secuencial
    assert "ráfaga" not in secuencial
