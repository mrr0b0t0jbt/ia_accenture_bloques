import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import modelo_gemini
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


def _modelo_con_metadata(respuestas: list[tuple[str, dict]]) -> GenericFakeChatModel:
    mensajes = [
        AIMessage(
            content=texto,
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            response_metadata={"model_name": "modelo-falso", **metadata},
        )
        for texto, metadata in respuestas
    ]
    return GenericFakeChatModel(messages=iter(mensajes))


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
    monkeypatch.setattr(modelo_langchain, "crear_modelo", lambda **kwargs: _modelo_falso(llamadas))
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


@pytest.mark.parametrize("tope", ["0.101", "1", "0.1001"])
def test_main_tope_sobre_el_maximo_devuelve_1_sin_crear_modelo(monkeypatch, tmp_path, tope):
    falso = MagicMock()
    monkeypatch.setattr(modelo_langchain, "crear_modelo", falso)
    argv = ["--confirmar-api-real", "--casos", CASOS, "--salida", str(tmp_path / "salida.json"), "--tope-usd", tope]
    assert ejecutar_cadenas.main(argv) == 1
    falso.assert_not_called()


def test_main_tope_igual_al_maximo_se_acepta(monkeypatch, tmp_path):
    monkeypatch.setattr(modelo_langchain, "crear_modelo", lambda **kwargs: _modelo_falso([]))
    argv = ["--confirmar-api-real", "--casos", CASOS, "--salida", str(tmp_path / "salida.json"), "--tope-usd", "0.10"]
    assert ejecutar_cadenas.main(argv) == 0


def test_constantes_de_tope_y_de_max_tokens():
    assert ejecutar_cadenas.TOPE_USD_MAXIMO == 0.10
    assert ejecutar_cadenas.MAX_TOKENS_MAXIMO == 2048


@pytest.mark.parametrize("max_tokens", ["2049", "0", "-1"])
def test_main_max_tokens_fuera_de_rango_devuelve_1_sin_crear_modelo(monkeypatch, tmp_path, max_tokens):
    falso = MagicMock()
    monkeypatch.setattr(modelo_langchain, "crear_modelo", falso)
    argv = ["--confirmar-api-real", "--casos", CASOS, "--salida", str(tmp_path / "salida.json"), "--max-tokens", max_tokens]
    assert ejecutar_cadenas.main(argv) == 1
    falso.assert_not_called()


def test_main_sin_max_tokens_crea_el_modelo_con_2048(monkeypatch, tmp_path):
    capturados = {}

    def crear(**kwargs):
        capturados.update(kwargs)
        return _modelo_falso([])

    monkeypatch.setattr(modelo_langchain, "crear_modelo", crear)
    assert ejecutar_cadenas.main(["--confirmar-api-real", "--casos", CASOS, "--salida", str(tmp_path / "salida.json")]) == 0
    assert capturados == {"max_tokens": 2048}


def test_main_con_max_tokens_1000_crea_el_modelo_con_1000(monkeypatch, tmp_path):
    capturados = {}

    def crear(**kwargs):
        capturados.update(kwargs)
        return _modelo_falso([])

    monkeypatch.setattr(modelo_langchain, "crear_modelo", crear)
    argv = ["--confirmar-api-real", "--casos", CASOS, "--salida", str(tmp_path / "salida.json"), "--max-tokens", "1000"]
    assert ejecutar_cadenas.main(argv) == 0
    assert capturados == {"max_tokens": 1000}


def test_main_con_13_llamadas_devuelve_1_sin_crear_modelo(monkeypatch, tmp_path):
    falso = MagicMock()
    monkeypatch.setattr(modelo_langchain, "crear_modelo", falso)
    casos = _escribir_casos(tmp_path, {"avisos": [AVISO_FICTICIO], "mensajes": [MENSAJE_FICTICIO] * 5})
    assert ejecutar_cadenas.main(["--confirmar-api-real", "--casos", casos, "--salida", str(tmp_path / "salida.json")]) == 1
    falso.assert_not_called()


def test_main_con_exactamente_12_llamadas_se_ejecuta(monkeypatch, tmp_path):
    llamadas = []
    monkeypatch.setattr(modelo_langchain, "crear_modelo", lambda **kwargs: _modelo_falso(llamadas, ["consulta"] * 12))
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


def test_respuesta_de_ruta_cortada_marca_truncado_solo_en_su_caso(tmp_path):
    fin = {"stop_reason": "end_turn"}
    modelo = _modelo_con_metadata(
        [("puntos", fin), ("simple", fin), ("- [ ] lista", fin), ("consulta", fin), ("respuesta a medias", {"stop_reason": "max_tokens"})]
    )
    ruta = tmp_path / "salida.json"
    ejecutar_cadenas.ejecutar_casos({"avisos": [AVISO_FICTICIO], "mensajes": [MENSAJE_FICTICIO]}, modelo, 0.05, str(ruta))
    aviso, mensaje = json.loads(ruta.read_text(encoding="utf-8"))["resultados"]
    assert aviso["truncado"] is False
    assert mensaje["truncado"] is True


def test_respuestas_completas_no_marcan_truncado(tmp_path):
    fin = {"stop_reason": "end_turn"}
    modelo = _modelo_con_metadata([("puntos", fin), ("simple", fin), ("- [ ] lista", fin), ("consulta", fin), ("respuesta", fin)])
    casos = {"avisos": [AVISO_FICTICIO], "mensajes": [MENSAJE_FICTICIO]}
    salida = ejecutar_cadenas.ejecutar_casos(casos, modelo, 0.05, str(tmp_path / "salida.json"))
    assert [r["truncado"] for r in salida["resultados"]] == [False, False]


def test_finish_reason_max_tokens_marca_truncado(tmp_path):
    modelo = _modelo_con_metadata([("consulta", {}), ("respuesta a medias", {"finish_reason": "MAX_TOKENS"})])
    casos = {"avisos": [], "mensajes": [MENSAJE_FICTICIO]}
    salida = ejecutar_cadenas.ejecutar_casos(casos, modelo, 0.05, str(tmp_path / "salida.json"))
    assert salida["resultados"][0]["truncado"] is True


def test_informar_avisa_solo_de_las_respuestas_truncadas(capsys):
    salida = {"resultados": [{"respuesta": "a medias", "truncado": True}], "cortado_por_tope": False, "costo_total_estimado": 0.0}
    ejecutar_cadenas.informar(salida)
    assert "AVISO: respuesta cortada por max_tokens." in capsys.readouterr().out
    salida["resultados"][0]["truncado"] = False
    ejecutar_cadenas.informar(salida)
    assert "AVISO" not in capsys.readouterr().out


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


def test_archivo_real_de_casos_es_valido_y_su_vulnerabilidad_trae_el_id_de_laboratorio():
    casos = ejecutar_cadenas.cargar_casos(CASOS)
    assert ejecutar_cadenas.validar_casos(casos) is None
    assert len(casos["avisos"]) == 1
    assert len(casos["mensajes"]) == 3
    assert [m["esperada"] for m in casos["mensajes"]] == ["incidente", "vulnerabilidad", "consulta"]
    vulnerabilidad = casos["mensajes"][1]["mensaje"]
    assert "CVE-2099-10002" in vulnerabilidad
    assert "CVE-0000" not in vulnerabilidad


def test_main_devuelve_1_si_ningun_caso_da_un_resultado_util(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(modelo_langchain, "crear_modelo", lambda **kwargs: MagicMock(side_effect=ConnectionError("simulado")))
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
    monkeypatch.setattr(modelo_langchain, "crear_modelo", lambda **kwargs: modelo)
    salida = tmp_path / "salida.json"
    assert ejecutar_cadenas.main(["--confirmar-api-real", "--casos", CASOS, "--salida", str(salida)]) == 0
    resultados = json.loads(salida.read_text(encoding="utf-8"))["resultados"]
    assert [r.get("respuesta") == "respuesta útil" for r in resultados] == [False, True, False, False]


def test_main_proveedor_invalido_sale_con_2_sin_crear_modelo(monkeypatch, tmp_path, capsys):
    falso_claude, falso_gemini = MagicMock(), MagicMock()
    monkeypatch.setattr(modelo_langchain, "crear_modelo", falso_claude)
    monkeypatch.setattr(modelo_gemini, "crear_modelo_gemini", falso_gemini)
    argv = ["--confirmar-api-real", "--casos", CASOS, "--salida", str(tmp_path / "salida.json"), "--proveedor", "otro"]
    with pytest.raises(SystemExit) as salida:
        ejecutar_cadenas.main(argv)
    assert salida.value.code == 2
    assert "invalid choice" in capsys.readouterr().err
    falso_claude.assert_not_called()
    falso_gemini.assert_not_called()


def test_main_proveedor_gemini_crea_su_modelo_con_2048_y_no_el_de_claude(monkeypatch, tmp_path):
    falso_claude = MagicMock()
    capturados = {}

    def crear(**kwargs):
        capturados.update(kwargs)
        return _modelo_falso([])

    monkeypatch.setattr(modelo_langchain, "crear_modelo", falso_claude)
    monkeypatch.setattr(modelo_gemini, "crear_modelo_gemini", crear)
    argv = ["--confirmar-api-real", "--casos", CASOS, "--salida", str(tmp_path / "salida.json"), "--proveedor", "gemini"]
    assert ejecutar_cadenas.main(argv) == 0
    assert capturados == {"max_output_tokens": 2048}
    falso_claude.assert_not_called()


def test_main_proveedor_gemini_con_max_tokens_1000_pasa_max_output_tokens(monkeypatch, tmp_path):
    capturados = {}

    def crear(**kwargs):
        capturados.update(kwargs)
        return _modelo_falso([])

    monkeypatch.setattr(modelo_gemini, "crear_modelo_gemini", crear)
    argv = ["--confirmar-api-real", "--casos", CASOS, "--salida", str(tmp_path / "salida.json")]
    assert ejecutar_cadenas.main(argv + ["--proveedor", "gemini", "--max-tokens", "1000"]) == 0
    assert capturados == {"max_output_tokens": 1000}


def test_main_sin_proveedor_no_crea_el_modelo_de_gemini(monkeypatch, tmp_path):
    falso_gemini = MagicMock()
    monkeypatch.setattr(modelo_langchain, "crear_modelo", lambda **kwargs: _modelo_falso([]))
    monkeypatch.setattr(modelo_gemini, "crear_modelo_gemini", falso_gemini)
    assert ejecutar_cadenas.main(["--confirmar-api-real", "--casos", CASOS, "--salida", str(tmp_path / "salida.json")]) == 0
    falso_gemini.assert_not_called()


def test_costo_de_gemini_usa_sus_precios_y_el_de_claude_no_cambia(monkeypatch, tmp_path):
    casos = ejecutar_cadenas.cargar_casos(CASOS)
    ruta = str(tmp_path / "salida.json")
    with monkeypatch.context() as contexto:
        # con None en sys.modules, cualquier import de api_claude_refactor lanzaría ImportError
        contexto.setitem(sys.modules, "api_claude_refactor", None)
        gemini = ejecutar_cadenas.ejecutar_casos(casos, _modelo_falso([]), 0.05, ruta, proveedor="gemini")
    # cada llamada cuesta 10 * 0.30e-6 + 5 * 2.5e-6 = 1.55e-5
    assert gemini["costo_total_estimado"] == pytest.approx(1.395e-4)
    assert gemini["proveedor"] == "gemini"
    assert "Gemini" in gemini["nota_costo"]
    claude = ejecutar_cadenas.ejecutar_casos(casos, _modelo_falso([]), 0.05, ruta)
    assert claude["costo_total_estimado"] == pytest.approx(6.3e-4)
    assert claude["proveedor"] == "claude"


def test_main_proveedor_gemini_sin_confirmar_devuelve_2_sin_crear_modelo(monkeypatch):
    falso_gemini = MagicMock()
    monkeypatch.setattr(modelo_gemini, "crear_modelo_gemini", falso_gemini)
    assert ejecutar_cadenas.main(["--proveedor", "gemini"]) == 2
    falso_gemini.assert_not_called()


def test_main_proveedor_gemini_sin_clave_devuelve_1_sin_crear_la_salida(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(modelo_gemini, "load_dotenv", lambda: None)
    casos = _escribir_casos(tmp_path, {"avisos": [AVISO_FICTICIO], "mensajes": [MENSAJE_FICTICIO]})
    salida = tmp_path / "salida.json"
    argv = ["--confirmar-api-real", "--proveedor", "gemini", "--casos", casos, "--salida", str(salida)]
    assert ejecutar_cadenas.main(argv) == 1
    assert capsys.readouterr().err.startswith("Error: Falta GEMINI_API_KEY")
    assert not salida.exists()


def test_main_proveedor_gemini_escribe_su_proveedor_y_su_nota_en_el_json(monkeypatch, tmp_path):
    monkeypatch.setattr(modelo_gemini, "crear_modelo_gemini", lambda **kwargs: _modelo_falso([]))
    salida = tmp_path / "salida.json"
    argv = ["--confirmar-api-real", "--casos", CASOS, "--salida", str(salida), "--proveedor", "gemini"]
    assert ejecutar_cadenas.main(argv) == 0
    escrito = json.loads(salida.read_text(encoding="utf-8"))
    assert escrito["proveedor"] == "gemini"
    assert "Gemini" in escrito["nota_costo"]
