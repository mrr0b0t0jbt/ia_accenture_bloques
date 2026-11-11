import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

import modelo_langchain
from api_claude_refactor import INPUT_PRICING, OUTPUT_PRICING
from modelo_falso_herramientas import ModeloFalsoHerramientas, pedir_herramienta

RAIZ = Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location("ejecutar_agente", RAIZ / "scripts" / "ejecutar_agente.py")
ejecutar_agente = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ejecutar_agente)

CASOS = str(RAIZ / "JSON" / "casos_agente.json")
SCRIPT = str(RAIZ / "scripts" / "ejecutar_agente.py")
# usage_metadata solo no basta: UsageMetadataCallbackHandler.on_llm_end ignora el mensaje si response_metadata no trae model_name
USO = {"usage_metadata": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}, "response_metadata": {"model_name": "falso"}}
COSTO_POR_LLAMADA = 10 * INPUT_PRICING + 5 * OUTPUT_PRICING
CAMPOS_DEL_TURNO = {
    "id", "sesion", "mensaje", "respuesta", "herramientas_usadas", "herramientas_esperadas", "coincide_herramientas",
    "pasos", "llamadas", "tokens_entrada", "tokens_salida", "costo_estimado", "truncado", "mensajes_en_sesion", "hubo_fallo",
}
# corre en un intérprete limpio: lo que ya importó pytest no enmascara lo que carga el script
CODIGO_SIN_FLAG = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("ejecutar_agente", sys.argv[1])
modulo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modulo)
codigo = modulo.main([])
cargados = [n for n in sys.modules if n.startswith("langchain") or n == "api_claude_refactor"]
print(json.dumps({"codigo": codigo, "cargados": cargados}))
"""


class ModeloDeGuion(ModeloFalsoHerramientas):
    """Modelo falso que puede fallar en una llamada y que rechaza el camino con streaming."""

    fallar_en: int | None = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if len(self.entradas) == self.fallar_en:
            self.entradas.append(messages)
            raise RuntimeError("fallo simulado")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):
        # con streaming, ChatAnthropic con herramientas entrega el output como lista de bloques y eso rompe la memoria
        raise AssertionError("el agente no debe invocar al modelo con streaming")


def _respuesta(texto: str) -> AIMessage:
    return AIMessage(content=texto, **USO)


def _pedir_calcular(expresion: str = "340 + 215") -> AIMessage:
    return pedir_herramienta("calcular", {"expresion": expresion}).model_copy(update=USO)


def _turno(id_turno: str = "t1", sesion: str = "s1", mensaje: str = "Pregunta ficticia.", esperadas: list[str] | None = None) -> dict:
    return {"id": id_turno, "sesion": sesion, "mensaje": mensaje, "herramientas_esperadas": esperadas or []}


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return ["--confirmar-api-real", "--casos", CASOS, "--salida", str(tmp_path / "salida.json"), *extra]


def _escribir_casos(tmp_path: Path, turnos: list[dict]) -> str:
    ruta = tmp_path / "casos.json"
    ruta.write_text(json.dumps({"turnos": turnos}), encoding="utf-8")
    return str(ruta)


def test_main_sin_confirmar_devuelve_2_sin_salida_ni_langchain(tmp_path, capsys):
    salida = tmp_path / "salida.json"
    assert ejecutar_agente.main(["--salida", str(salida)]) == 2
    assert "--confirmar-api-real" in capsys.readouterr().err
    assert not salida.exists()
    proceso = subprocess.run([sys.executable, "-c", CODIGO_SIN_FLAG, SCRIPT], capture_output=True, text=True, cwd=tmp_path, timeout=60)
    assert proceso.returncode == 0, proceso.stderr
    assert json.loads(proceso.stdout) == {"codigo": 2, "cargados": []}


@pytest.mark.parametrize(
    "argumentos",
    [["--tope-usd", "0.5"], ["--tope-usd", "0"], ["--max-tokens", "4096"], ["--max-tokens", "0"]],
    ids=["tope_alto", "tope_cero", "max_tokens_alto", "max_tokens_cero"],
)
def test_main_rechaza_topes_fuera_de_rango_sin_crear_modelo(monkeypatch, tmp_path, argumentos):
    falso = MagicMock()
    monkeypatch.setattr(modelo_langchain, "crear_modelo", falso)
    assert ejecutar_agente.main(_argv(tmp_path, *argumentos)) == 1
    falso.assert_not_called()


def test_main_sin_clave_devuelve_1_sin_crear_la_salida(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    monkeypatch.setattr(modelo_langchain, "load_dotenv", lambda: None)
    assert ejecutar_agente.main(_argv(tmp_path)) == 1
    assert capsys.readouterr().err.startswith("Error: Falta CLAUDE_API_KEY")
    assert not (tmp_path / "salida.json").exists()


@pytest.mark.parametrize(
    "casos",
    [
        {"otra": []},
        {"turnos": [{"id": "t", "sesion": "s", "herramientas_esperadas": []}]},
        {"turnos": [{"id": "t", "mensaje": "m", "herramientas_esperadas": []}]},
        {"turnos": [{"id": "t", "sesion": "s", "mensaje": "m", "herramientas_esperadas": ["inexistente"]}]},
    ],
    ids=["sin_turnos", "sin_mensaje", "sin_sesion", "herramienta_inexistente"],
)
def test_validar_casos_agente_devuelve_el_problema(casos):
    assert isinstance(ejecutar_agente.validar_casos_agente(casos), str)


def test_main_con_11_turnos_devuelve_1_sin_crear_modelo(monkeypatch, tmp_path):
    falso = MagicMock(side_effect=AssertionError("no debe crearse el modelo"))
    monkeypatch.setattr(modelo_langchain, "crear_modelo", falso)
    casos = _escribir_casos(tmp_path, [_turno(f"t{n}") for n in range(11)])
    assert ejecutar_agente.main(["--confirmar-api-real", "--casos", casos, "--salida", str(tmp_path / "salida.json")]) == 1
    falso.assert_not_called()


def test_archivo_real_de_casos_es_valido_y_separa_las_sesiones():
    casos = json.loads(Path(CASOS).read_text(encoding="utf-8"))
    assert ejecutar_agente.validar_casos_agente(casos) is None
    turnos = casos["turnos"]
    assert len(turnos) == 8
    assert turnos[5]["sesion"] == turnos[6]["sesion"] != turnos[7]["sesion"]


def test_turno_con_herramienta_registra_su_traza_su_uso_y_su_costo(tmp_path):
    modelo = ModeloDeGuion(guion=[_pedir_calcular(), _respuesta("Son 555.")])
    salida = ejecutar_agente.ejecutar_turnos({"turnos": [_turno(esperadas=["calcular"])]}, modelo, 0.05, str(tmp_path / "s.json"))
    [turno] = salida["resultados"]
    assert turno["respuesta"] == "Son 555."
    assert turno["herramientas_usadas"] == ["calcular"]
    assert turno["pasos"] == [{"herramienta": "calcular", "entrada": {"expresion": "340 + 215"}, "observacion": "555"}]
    assert turno["coincide_herramientas"] is True
    assert turno["llamadas"] == 2
    assert turno["mensajes_en_sesion"] == 2
    assert turno["tokens_entrada"] == 20
    assert turno["tokens_salida"] == 10
    assert turno["costo_estimado"] == pytest.approx(2 * COSTO_POR_LLAMADA)
    assert turno["hubo_fallo"] is False


def test_respuesta_directa_cuando_se_esperaba_herramienta_no_coincide(tmp_path):
    modelo = ModeloDeGuion(guion=[_respuesta("Sin herramienta.")])
    salida = ejecutar_agente.ejecutar_turnos({"turnos": [_turno(esperadas=["calcular"])]}, modelo, 0.05, str(tmp_path / "s.json"))
    [turno] = salida["resultados"]
    assert turno["herramientas_usadas"] == []
    assert turno["coincide_herramientas"] is False


def test_limite_de_llamadas_corta_el_turno_y_los_siguientes(tmp_path):
    modelo = ModeloDeGuion(guion=[_pedir_calcular()] * 20)
    casos = {"turnos": [_turno("t1"), _turno("t2")]}
    salida = ejecutar_agente.ejecutar_turnos(casos, modelo, 0.05, str(tmp_path / "s.json"), max_llamadas=3)
    assert [turno["id"] for turno in salida["resultados"]] == ["t1"]
    assert "error" in salida["resultados"][0]
    assert salida["cortado_por_llamadas"] is True
    assert salida["cortado_por_tope"] is False
    assert salida["llamadas_totales"] <= 3
    assert len(modelo.entradas) <= 3


def test_tope_de_gasto_corta_entre_turnos_y_dentro_de_un_turno(tmp_path):
    ruta = str(tmp_path / "s.json")
    casos = {"turnos": [_turno("t1"), _turno("t2")]}
    # entre turnos: la primera llamada ya alcanza el tope y el segundo turno no empieza
    entre_turnos = ejecutar_agente.ejecutar_turnos(casos, ModeloDeGuion(guion=[_respuesta("a"), _respuesta("b")]), COSTO_POR_LLAMADA / 2, ruta)
    assert [turno["id"] for turno in entre_turnos["resultados"]] == ["t1"]
    assert "error" not in entre_turnos["resultados"][0]
    assert entre_turnos["cortado_por_tope"] is True
    # dentro de un turno: la segunda llamada alcanza el tope y la tercera ya no se hace
    tope = COSTO_POR_LLAMADA * 1.5
    modelo = ModeloDeGuion(guion=[_pedir_calcular()] * 20)
    dentro = ejecutar_agente.ejecutar_turnos(casos, modelo, tope, ruta)
    assert [turno["id"] for turno in dentro["resultados"]] == ["t1"]
    assert "error" in dentro["resultados"][0]
    assert dentro["cortado_por_tope"] is True
    assert dentro["cortado_por_llamadas"] is False
    assert len(modelo.entradas) == 2
    assert dentro["costo_total_estimado"] <= tope + COSTO_POR_LLAMADA


def test_max_iteraciones_detiene_el_turno_y_marca_el_fallo(tmp_path):
    modelo = ModeloDeGuion(guion=[_pedir_calcular()] * 20)
    salida = ejecutar_agente.ejecutar_turnos(
        {"turnos": [_turno()]}, modelo, 0.05, str(tmp_path / "s.json"), max_llamadas=20, max_iteraciones=2
    )
    [turno] = salida["resultados"]
    assert turno["respuesta"] == "Agent stopped due to max iterations."
    assert turno["hubo_fallo"] is True
    assert "error" not in turno


def test_error_del_modelo_queda_en_su_turno_y_no_corta_el_siguiente(tmp_path):
    modelo = ModeloDeGuion(guion=[_respuesta("no se usa"), _respuesta("Respuesta dos.")], fallar_en=0)
    salida = ejecutar_agente.ejecutar_turnos({"turnos": [_turno("t1"), _turno("t2")]}, modelo, 0.05, str(tmp_path / "s.json"))
    primero, segundo = salida["resultados"]
    assert primero["error"] == "fallo simulado"
    assert primero["hubo_fallo"] is True
    assert segundo["respuesta"] == "Respuesta dos."
    assert salida["cortado_por_llamadas"] is False


def test_las_sesiones_del_script_quedan_aisladas(tmp_path):
    modelo = ModeloDeGuion(guion=[_respuesta("uno"), _respuesta("dos")])
    casos = {"turnos": [_turno("t1", "a", "dato-secreto-de-a"), _turno("t2", "b", "Pregunta de b.")]}
    salida = ejecutar_agente.ejecutar_turnos(casos, modelo, 0.05, str(tmp_path / "s.json"))
    assert "dato-secreto-de-a" in modelo.texto_recibido(0)
    assert "dato-secreto-de-a" not in modelo.texto_recibido(1)
    assert [turno["mensajes_en_sesion"] for turno in salida["resultados"]] == [2, 2]


def test_informar_muestra_diferencias_cortes_errores_y_la_nota(capsys):
    comunes = {"herramientas_esperadas": ["calcular"], "truncado": False, "hubo_fallo": False}
    salida = {
        "nota_costo": "Nota de costo ficticia.",
        "resultados": [
            {**comunes, "id": "t1", "respuesta": "uno", "herramientas_usadas": []},
            {**comunes, "id": "t2", "respuesta": "dos", "herramientas_usadas": ["calcular"], "truncado": True},
            {**comunes, "id": "t3", "error": "fallo simulado", "hubo_fallo": True},
        ],
        "llamadas_totales": 3,
        "costo_total_estimado": 0.00021,
        "cortado_por_tope": True,
        "cortado_por_llamadas": False,
    }
    ejecutar_agente.informar(salida)
    impreso = capsys.readouterr().out
    assert "[Turno 1] t1: uno" in impreso
    assert impreso.count("DIFERENCIA de herramientas") == 1
    assert impreso.count("AVISO: respuesta cortada por max_tokens.") == 1
    assert "Error: fallo simulado" in impreso
    assert "tope de gasto" in impreso
    assert "Costo total estimado: $0.00021" in impreso
    assert "Nota de costo ficticia." in impreso


def test_main_de_punta_a_punta_escribe_el_json_completo(monkeypatch, tmp_path):
    modelo = ModeloDeGuion(guion=[_pedir_calcular(), _respuesta("Son 555."), _respuesta("Directo.")])
    monkeypatch.setattr(modelo_langchain, "crear_modelo", lambda **kwargs: modelo)
    casos = _escribir_casos(tmp_path, [_turno("t1", "a", esperadas=["calcular"]), _turno("t2", "b")])
    salida = tmp_path / "salida.json"
    assert ejecutar_agente.main(["--confirmar-api-real", "--casos", casos, "--salida", str(salida)]) == 0
    escrito = json.loads(salida.read_text(encoding="utf-8"))
    assert escrito["proveedor"] == "claude"
    assert "estimación" in escrito["nota_costo"]
    assert escrito["llamadas_totales"] == 3
    assert escrito["costo_total_estimado"] == pytest.approx(3 * COSTO_POR_LLAMADA)
    assert len(escrito["resultados"]) == 2
    assert all(CAMPOS_DEL_TURNO <= set(turno) for turno in escrito["resultados"])


def test_truncado_se_calcula_por_turno_y_no_se_arrastra_al_siguiente(tmp_path):
    cortada = AIMessage(content="a medias", **{**USO, "response_metadata": {"model_name": "falso", "stop_reason": "max_tokens"}})
    modelo = ModeloDeGuion(guion=[cortada, _respuesta("completa")])
    casos = {"turnos": [_turno("t1"), _turno("t2")]}
    salida = ejecutar_agente.ejecutar_turnos(casos, modelo, 0.05, str(tmp_path / "s.json"))
    assert [turno["truncado"] for turno in salida["resultados"]] == [True, False]


def test_guarda_la_salida_despues_de_cada_turno_y_al_final(monkeypatch, tmp_path):
    # se anota cuántos resultados traía la salida en cada llamada, porque el dict se sigue modificando
    resultados_por_llamada = []
    monkeypatch.setattr(
        ejecutar_agente.ejecutar_cadenas, "guardar_resultados", lambda salida, ruta: resultados_por_llamada.append(len(salida["resultados"]))
    )
    modelo = ModeloDeGuion(guion=[_respuesta("uno"), _respuesta("dos")])
    casos = {"turnos": [_turno("t1"), _turno("t2")]}
    ejecutar_agente.ejecutar_turnos(casos, modelo, 0.05, str(tmp_path / "s.json"))
    assert resultados_por_llamada == [1, 2, 2]


def test_tope_igual_al_costo_exacto_de_un_turno_no_ejecuta_el_siguiente(tmp_path):
    # mismos precios y misma expresión que costo_de en el script
    costo_unitario = 10 * INPUT_PRICING + 5 * OUTPUT_PRICING
    modelo = ModeloDeGuion(guion=[_respuesta("uno"), _respuesta("dos")])
    casos = {"turnos": [_turno("t1"), _turno("t2")]}
    salida = ejecutar_agente.ejecutar_turnos(casos, modelo, costo_unitario, str(tmp_path / "s.json"))
    assert [turno["id"] for turno in salida["resultados"]] == ["t1"]
    assert salida["cortado_por_tope"] is True
    assert len(modelo.entradas) == 1


def test_turno_sin_datos_de_uso_corta_la_corrida(tmp_path):
    modelo = ModeloDeGuion(guion=[AIMessage(content="uno"), AIMessage(content="dos")])
    casos = {"turnos": [_turno("t1"), _turno("t2")]}
    salida = ejecutar_agente.ejecutar_turnos(casos, modelo, 0.05, str(tmp_path / "s.json"))
    assert salida["cortado_por_falta_de_uso"] is True
    assert [turno["id"] for turno in salida["resultados"]] == ["t1"]
    assert len(modelo.entradas) == 1


def test_informar_avisa_cuando_la_corrida_se_corto_por_falta_de_datos_de_uso(capsys):
    salida = {
        "nota_costo": "Nota de costo ficticia.",
        "resultados": [],
        "llamadas_totales": 1,
        "costo_total_estimado": 0.0,
        "cortado_por_tope": False,
        "cortado_por_llamadas": False,
        "cortado_por_falta_de_uso": True,
    }
    ejecutar_agente.informar(salida)
    assert "no devolvió datos de uso" in capsys.readouterr().out


@pytest.mark.parametrize("largo, esperado_texto", [(2001, True), (2000, False)], ids=["2001_caracteres", "2000_caracteres"])
def test_validar_casos_agente_limita_el_largo_del_mensaje(largo, esperado_texto):
    casos = {"turnos": [_turno(mensaje="x" * largo)]}
    problema = ejecutar_agente.validar_casos_agente(casos)
    if esperado_texto:
        assert "2000" in problema
    else:
        assert problema is None


def test_respuesta_en_bloques_se_guarda_como_texto(tmp_path):
    bloques = [{"type": "thinking", "thinking": "", "signature": "firma-opaca"}, {"type": "text", "text": "Su severidad era media."}]
    modelo = ModeloDeGuion(guion=[AIMessage(content=bloques, **USO)])
    salida = ejecutar_agente.ejecutar_turnos({"turnos": [_turno()]}, modelo, 0.05, str(tmp_path / "s.json"))
    [turno] = salida["resultados"]
    assert turno["respuesta"] == "Su severidad era media."
    assert turno["mensajes_en_sesion"] == 2


def test_main_rechaza_una_salida_que_es_un_enlace_duro_a_los_casos(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(modelo_langchain, "crear_modelo", MagicMock(side_effect=AssertionError("no debe crearse el modelo")))
    casos = Path(_escribir_casos(tmp_path, [_turno()]))
    original = casos.read_text(encoding="utf-8")
    enlace = tmp_path / "enlace.json"
    os.link(casos, enlace)
    assert ejecutar_agente.main(["--confirmar-api-real", "--casos", str(casos), "--salida", str(enlace)]) == 1
    assert "--salida" in capsys.readouterr().err
    assert casos.read_text(encoding="utf-8") == original
