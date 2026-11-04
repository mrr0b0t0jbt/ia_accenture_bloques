import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import modelo_langchain

RAIZ = Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location("ejecutar_cadenas", RAIZ / "scripts" / "ejecutar_cadenas.py")
ejecutar_cadenas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ejecutar_cadenas)

CASOS = str(RAIZ / "JSON" / "casos_cadenas.json")
RESPUESTAS = [
    "puntos falsos", "texto simple falso", "- [ ] checklist falsa",
    "incidente", "respuesta a incidente",
    "vulnerabilidad", "respuesta a vulnerabilidad",
    "consulta", "respuesta a consulta",
]


def _modelo_falso(llamadas: list[str]) -> GenericFakeChatModel:
    # modelo de chat real de LangChain: a diferencia de un MagicMock, dispara los callbacks de uso
    def generar():
        for texto in RESPUESTAS:
            llamadas.append(texto)
            yield AIMessage(
                content=texto,
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                response_metadata={"model_name": "modelo-falso"},
            )

    return GenericFakeChatModel(messages=generar())


def test_main_sin_confirmar_api_real_devuelve_2_sin_crear_modelo(monkeypatch, capsys):
    # con None en sys.modules, cualquier import de langchain_anthropic lanzaría ImportError
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)
    falso = MagicMock()
    monkeypatch.setattr(modelo_langchain, "crear_modelo", falso)
    assert ejecutar_cadenas.main([]) == 2
    falso.assert_not_called()
    assert "--confirmar-api-real" in capsys.readouterr().err


def test_main_tope_cero_o_negativo_devuelve_1_sin_crear_modelo(monkeypatch, tmp_path):
    falso = MagicMock()
    monkeypatch.setattr(modelo_langchain, "crear_modelo", falso)
    argv = ["--confirmar-api-real", "--casos", CASOS, "--salida", str(tmp_path / "salida.json"), "--tope-usd"]
    assert ejecutar_cadenas.main(argv + ["0"]) == 1
    assert ejecutar_cadenas.main(argv + ["-0.01"]) == 1
    falso.assert_not_called()


def test_main_salida_en_directorio_inexistente_devuelve_1_sin_crear_modelo(monkeypatch, tmp_path):
    falso = MagicMock()
    monkeypatch.setattr(modelo_langchain, "crear_modelo", falso)
    salida = tmp_path / "no_existe" / "salida.json"
    assert ejecutar_cadenas.main(["--confirmar-api-real", "--casos", CASOS, "--salida", str(salida)]) == 1
    falso.assert_not_called()


def test_main_con_modelo_falso_procesa_4_casos_con_9_llamadas_y_escribe_el_json(monkeypatch, tmp_path, capsys):
    llamadas = []
    monkeypatch.setattr(modelo_langchain, "crear_modelo", lambda: _modelo_falso(llamadas))
    salida = tmp_path / "salida.json"
    assert ejecutar_cadenas.main(["--confirmar-api-real", "--casos", CASOS, "--salida", str(salida)]) == 0
    assert len(llamadas) == 9
    escrito = json.loads(salida.read_text(encoding="utf-8"))
    aviso, *mensajes = escrito["resultados"]
    assert aviso["puntos"] == "puntos falsos"
    assert aviso["respuesta"] == "- [ ] checklist falsa"
    assert [m["categoria_obtenida"] for m in mensajes] == [m["categoria_esperada"] for m in mensajes]
    assert mensajes[0]["respuesta"] == "respuesta a incidente"
    assert aviso["tokens_entrada"] == 30 and aviso["tokens_salida"] == 15
    # 9 llamadas con 10 tokens de entrada (USD 2/M) y 5 de salida (USD 10/M)
    assert escrito["costo_total_estimado"] == pytest.approx(9 * (10 * 2 + 5 * 10) / 1_000_000)
    assert "estimación" in escrito["nota_costo"]
    assert "estimación" in capsys.readouterr().out
