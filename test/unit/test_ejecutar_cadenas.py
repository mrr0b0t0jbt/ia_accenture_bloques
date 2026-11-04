import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import modelo_langchain
from enrutamiento import RESPUESTA_GENERICA

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
AVISO_FICTICIO = {"texto": "Aviso ficticio."}
MENSAJE_FICTICIO = {"mensaje": "Mensaje ficticio.", "esperada": "consulta"}
# corre en un intérprete limpio: lo que ya importó pytest no enmascara lo que carga el script
CODIGO_SIN_FLAG = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("ejecutar_cadenas", sys.argv[1])
modulo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulo)
codigo = modulo.main([])
cargados = [n for n in sys.modules if n.startswith("langchain") or n == "api_claude_refactor"]
print(json.dumps({"codigo": codigo, "cargados": cargados}))
"""


def _modelo_falso(llamadas: list[str], respuestas: list[str] = RESPUESTAS) -> GenericFakeChatModel:
    # modelo de chat real de LangChain: a diferencia de un MagicMock, dispara los callbacks de uso
    def generar():
        for texto in respuestas:
            llamadas.append(texto)
            yield AIMessage(
                content=texto,
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                response_metadata={"model_name": "modelo-falso"},
            )

    return GenericFakeChatModel(messages=generar())


def _escribir_casos(tmp_path: Path, casos: dict) -> str:
    ruta = tmp_path / "casos.json"
    ruta.write_text(json.dumps(casos), encoding="utf-8")
    return str(ruta)


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


@pytest.mark.parametrize("tope", ["0.051", "1", "0.0501"])
def test_main_tope_sobre_el_maximo_devuelve_1_sin_crear_modelo(monkeypatch, tmp_path, tope):
    falso = MagicMock()
    monkeypatch.setattr(modelo_langchain, "crear_modelo", falso)
    argv = ["--confirmar-api-real", "--casos", CASOS, "--salida", str(tmp_path / "salida.json"), "--tope-usd", tope]
    assert ejecutar_cadenas.main(argv) == 1
    falso.assert_not_called()


def test_main_tope_igual_al_maximo_se_acepta(monkeypatch, tmp_path):
    monkeypatch.setattr(modelo_langchain, "crear_modelo", lambda: _modelo_falso([]))
    argv = ["--confirmar-api-real", "--casos", CASOS, "--salida", str(tmp_path / "salida.json"), "--tope-usd", "0.05"]
    assert ejecutar_cadenas.main(argv) == 0


def test_main_con_13_llamadas_devuelve_1_sin_crear_modelo(monkeypatch, tmp_path):
    falso = MagicMock()
    monkeypatch.setattr(modelo_langchain, "crear_modelo", falso)
    casos = _escribir_casos(tmp_path, {"avisos": [AVISO_FICTICIO], "mensajes": [MENSAJE_FICTICIO] * 5})
    assert ejecutar_cadenas.main(["--confirmar-api-real", "--casos", casos, "--salida", str(tmp_path / "salida.json")]) == 1
    falso.assert_not_called()


def test_main_con_exactamente_12_llamadas_se_ejecuta(monkeypatch, tmp_path):
    llamadas = []
    monkeypatch.setattr(modelo_langchain, "crear_modelo", lambda: _modelo_falso(llamadas, ["consulta"] * 12))
    casos = _escribir_casos(tmp_path, {"avisos": [AVISO_FICTICIO] * 2, "mensajes": [MENSAJE_FICTICIO] * 3})
    assert ejecutar_cadenas.main(["--confirmar-api-real", "--casos", casos, "--salida", str(tmp_path / "salida.json")]) == 0
    assert len(llamadas) == 12


def test_ejecutar_casos_corta_entre_casos_al_llegar_al_tope(tmp_path):
    llamadas = []
    casos = ejecutar_cadenas.cargar_casos(CASOS)
    # cada llamada cuesta 7e-5: el aviso (3 llamadas, 2.1e-4) alcanza el tope de 2e-4
    salida = ejecutar_cadenas.ejecutar_casos(casos, _modelo_falso(llamadas), 0.0002, str(tmp_path / "salida.json"))
    assert len(salida["resultados"]) == 1
    assert "aviso" in salida["resultados"][0]
    assert salida["cortado_por_tope"] is True
    assert len(llamadas) == 3


def test_ejecutar_casos_corta_cuando_el_costo_iguala_exactamente_el_tope(tmp_path):
    casos = ejecutar_cadenas.cargar_casos(CASOS)
    ruta = str(tmp_path / "salida.json")
    primera = ejecutar_cadenas.ejecutar_casos(casos, _modelo_falso([]), 1.0, ruta)
    # el tope se toma del propio resultado: recalcularlo a mano podría diferir por redondeo de punto flotante
    costo = primera["resultados"][0]["costo_estimado"]
    llamadas = []
    salida = ejecutar_cadenas.ejecutar_casos(casos, _modelo_falso(llamadas), costo, ruta)
    assert len(salida["resultados"]) == 1
    assert salida["cortado_por_tope"] is True
    assert len(llamadas) == 3


def test_ruta_que_falla_marca_hubo_fallo_con_la_respuesta_generica(tmp_path):
    modelo = MagicMock(side_effect=[AIMessage(content="incidente"), ConnectionError("simulado")])
    casos = {"avisos": [], "mensajes": [MENSAJE_FICTICIO]}
    salida = ejecutar_cadenas.ejecutar_casos(casos, modelo, 0.05, str(tmp_path / "salida.json"))
    resultado = salida["resultados"][0]
    assert resultado["hubo_fallo"] is True
    assert resultado["respuesta"] == RESPUESTA_GENERICA


def test_camino_normal_no_marca_hubo_fallo(tmp_path):
    modelo = _modelo_falso([], ["consulta", "respuesta a consulta"])
    casos = {"avisos": [], "mensajes": [MENSAJE_FICTICIO]}
    salida = ejecutar_cadenas.ejecutar_casos(casos, modelo, 0.05, str(tmp_path / "salida.json"))
    assert salida["resultados"][0]["hubo_fallo"] is False


def test_main_sin_flag_en_interprete_limpio_no_carga_langchain_ni_el_cliente(tmp_path):
    script = str(RAIZ / "scripts" / "ejecutar_cadenas.py")
    proceso = subprocess.run(
        [sys.executable, "-c", CODIGO_SIN_FLAG, script], capture_output=True, text=True, cwd=tmp_path, timeout=60
    )
    assert proceso.returncode == 0, proceso.stderr
    assert json.loads(proceso.stdout) == {"codigo": 2, "cargados": []}


@pytest.mark.parametrize(
    "casos",
    [
        {"avisos": [], "mensajes": [{"mensaje": "Mensaje ficticio."}]},
        {"avisos": [], "mensajes": [{"mensaje": "Mensaje ficticio.", "esperada": "otra"}]},
        {"avisos": [{"titulo": "Aviso ficticio."}], "mensajes": []},
        {"avisos": []},
        {"avisos": [], "mensajes": [{"mensaje": "   ", "esperada": "consulta"}]},
        {"avisos": [{"texto": "   "}], "mensajes": []},
        {"avisos": [], "mensajes": [{"mensaje": "", "esperada": "consulta"}]},
    ],
    ids=[
        "mensaje_sin_esperada",
        "esperada_desconocida",
        "aviso_sin_texto",
        "json_sin_mensajes",
        "mensaje_solo_espacios",
        "aviso_solo_espacios",
        "mensaje_vacio",
    ],
)
def test_main_con_casos_invalidos_devuelve_1_sin_crear_modelo_ni_salida(monkeypatch, tmp_path, capsys, casos):
    falso = MagicMock()
    monkeypatch.setattr(modelo_langchain, "crear_modelo", falso)
    salida = tmp_path / "salida.json"
    argv = ["--confirmar-api-real", "--casos", _escribir_casos(tmp_path, casos), "--salida", str(salida)]
    assert ejecutar_cadenas.main(argv) == 1
    falso.assert_not_called()
    assert not salida.exists()
    assert capsys.readouterr().err.startswith("Error: ")


def test_main_devuelve_1_si_ningun_caso_da_un_resultado_util(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(modelo_langchain, "crear_modelo", lambda: MagicMock(side_effect=ConnectionError("simulado")))
    salida = tmp_path / "salida.json"
    assert ejecutar_cadenas.main(["--confirmar-api-real", "--casos", CASOS, "--salida", str(salida)]) == 1
    captura = capsys.readouterr()
    assert "Costo total estimado" in captura.out
    assert "resultado útil" in captura.err
    assert len(json.loads(salida.read_text(encoding="utf-8"))["resultados"]) == 4


def test_main_devuelve_0_si_al_menos_un_caso_da_un_resultado_util(monkeypatch, tmp_path):
    error = ConnectionError("simulado")
    # el aviso falla en su primer paso, el primer mensaje responde y los otros dos fallan al clasificar
    modelo = MagicMock(side_effect=[error, AIMessage(content="incidente"), AIMessage(content="respuesta útil"), error, error])
    monkeypatch.setattr(modelo_langchain, "crear_modelo", lambda: modelo)
    salida = tmp_path / "salida.json"
    assert ejecutar_cadenas.main(["--confirmar-api-real", "--casos", CASOS, "--salida", str(salida)]) == 0
    resultados = json.loads(salida.read_text(encoding="utf-8"))["resultados"]
    assert [r.get("respuesta") == "respuesta útil" for r in resultados] == [False, True, False, False]
