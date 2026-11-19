import copy
import datetime
import importlib.util
import json
import socket
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import modelo_langchain
from api_claude_refactor import INPUT_PRICING, OUTPUT_PRICING
from enrutamiento import CATEGORIAS

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_SCRIPT = RAIZ / "scripts" / "medir_clasificador_router.py"
RUTA_DATASET = RAIZ / "JSON" / "dataset_tuning_clasificacion.json"
DATOS = json.loads(RUTA_DATASET.read_text(encoding="utf-8"))
ESPERADAS = [e["salida"]["categoria"] for e in DATOS["ejemplos"]]
HORA = datetime.datetime(2026, 11, 18, 21, 30, 0, tzinfo=datetime.timezone.utc)


def _cargar_script():
    spec = importlib.util.spec_from_file_location("medir_clasificador_router", RUTA_SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


medir = _cargar_script()


def _mensaje(contenido, entrada=100, salida=5):
    uso = {"input_tokens": entrada, "output_tokens": salida, "total_tokens": entrada + salida}
    return SimpleNamespace(content=contenido, usage_metadata=uso, response_metadata={})


class ModeloGuionado:
    """Modelo falso: invoke n llama a guion(n, prompt); por defecto responde la categoría correcta del ejemplo n."""

    def __init__(self, guion=None):
        self.guion = guion or (lambda n, prompt: _mensaje(ESPERADAS[(n - 1) % len(ESPERADAS)]))
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.guion(len(self.prompts), prompt)


def _preparar(tmp_path, monkeypatch, guion=None):
    modelo = ModeloGuionado(guion)
    fabrica = MagicMock(return_value=modelo)
    monkeypatch.setattr(modelo_langchain, "crear_modelo", fabrica)
    monkeypatch.setattr(medir, "_ahora", lambda: HORA)
    return SimpleNamespace(tmp=tmp_path, modelo=modelo, fabrica=fabrica, salida=tmp_path / "resultados.json")


def _argv(entorno, *extra, salida=None, dataset=None):
    return [
        "--confirmar-api-real", "--dataset", str(dataset or RUTA_DATASET),
        "--salida", str(entorno.salida if salida is None else salida), *extra,
    ]


def _codigo(argv) -> int:
    """Código de salida de main; un rechazo de argparse (SystemExit) también cuenta."""
    try:
        return medir.main(argv)
    except SystemExit as salida:
        return salida.code


def _json(entorno) -> dict:
    return json.loads(entorno.salida.read_text(encoding="utf-8"))


def _sin_llamadas(entorno) -> bool:
    return entorno.modelo.prompts == [] and entorno.fabrica.call_count == 0 and not entorno.salida.exists()


def _dataset(tmp_path, cambiar) -> str:
    datos = copy.deepcopy(DATOS)
    cambiar(datos)
    ruta = tmp_path / "dataset.json"
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    return str(ruta)


# K1
def test_k1_sin_confirmar_devuelve_2_sin_importar_nada_ni_leer_el_dataset(tmp_path, monkeypatch, capsys):
    for nombre in ("modelo_langchain", "medicion_clasificador", "dataset_tuning"):
        monkeypatch.setitem(sys.modules, nombre, None)
    assert medir.main(["--dataset", str(tmp_path / "no_existe.json")]) == 2
    captura = capsys.readouterr()
    assert "--confirmar-api-real" in captura.err and "no existe" not in captura.err
    assert list(tmp_path.iterdir()) == []


# K2
def test_k2_la_abreviatura_conf_no_activa_la_confirmacion(tmp_path, monkeypatch):
    entorno = _preparar(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as salida:
        medir.main(["--conf", "--salida", str(entorno.salida)])
    assert salida.value.code == 2 and _sin_llamadas(entorno)


# K3
@pytest.mark.parametrize(
    "malo",
    [["--repeticiones", "0"], ["--repeticiones", "4"], ["--max-tokens", "0"], ["--max-tokens", "257"], ["--tope-usd", "0"], ["--tope-usd", "0.06"]],
    ids=lambda malo: "_".join(malo),
)
def test_k3_rangos_invalidos(tmp_path, monkeypatch, malo):
    entorno = _preparar(tmp_path, monkeypatch)
    assert _codigo(_argv(entorno, *malo)) == 2
    assert _sin_llamadas(entorno)


# K4
def test_k4_tope_de_llamadas(tmp_path, monkeypatch, capsys):
    def agrandar(datos):
        for i, ejemplo in enumerate(datos["ejemplos"][:6]):
            datos["ejemplos"].append({**copy.deepcopy(ejemplo), "id": f"extra_{i}", "texto": f"{ejemplo['texto']} (variante {i})"})

    entorno = _preparar(tmp_path, monkeypatch)
    ruta = _dataset(tmp_path, agrandar)
    assert len(json.loads(Path(ruta).read_text(encoding="utf-8"))["ejemplos"]) == 30
    assert medir.main(_argv(entorno, "--repeticiones", "3", dataset=ruta)) == 1  # 90 > 75
    assert "Error:" in capsys.readouterr().err and _sin_llamadas(entorno)


# K5
def _categoria_mala(datos):
    datos["ejemplos"][3]["salida"]["categoria"] = "otra"


@pytest.mark.parametrize("caso", ["inexistente", "json_invalido", "categoria_no_permitida"])
def test_k5_dataset_invalido(tmp_path, monkeypatch, capsys, caso):
    entorno = _preparar(tmp_path, monkeypatch)
    ruta = {
        "inexistente": str(tmp_path / "no_existe.json"),
        "json_invalido": str(tmp_path / "roto.json"),
        "categoria_no_permitida": _dataset(tmp_path, _categoria_mala),
    }[caso]
    if caso == "json_invalido":
        Path(ruta).write_text("{roto", encoding="utf-8")
    assert medir.main(_argv(entorno, dataset=ruta)) == 1
    err = capsys.readouterr().err
    assert err.startswith("Error:") and "Traceback" not in err and _sin_llamadas(entorno)


# K6
def test_k6_flujo_feliz(tmp_path, monkeypatch):
    entorno = _preparar(tmp_path, monkeypatch)
    assert medir.main(_argv(entorno)) == 0
    datos = _json(entorno)
    assert len(entorno.modelo.prompts) == 24 and len(datos["resultados"]) == 24
    assert datos["incompleto"] is False and datos["error"] is None and datos["modelo"] == "claude-sonnet-5-5"
    assert datos["fecha"] == HORA.isoformat()
    assert datos["parametros"]["repeticiones"] == 1 and datos["parametros"]["max_tokens"] == 64 and datos["parametros"]["tope_usd"] == 0.03
    r = datos["resumen"]
    assert (r["exactas"], r["fallos_formato_estricto"], r["aciertos"]) == (24, 0, 24)
    assert datos["costo_total"] == pytest.approx(24 * (100 * INPUT_PRICING + 5 * OUTPUT_PRICING))
    primero = datos["resultados"][0]
    assert [primero["id"], primero["repeticion"], primero["categoria_esperada"], primero["prioridad_esperada"]] == ["ej_01", 1, "incidente", "alta"]
    assert primero["tokens"] == {"input_tokens": 100, "output_tokens": 5, "total_tokens": 105} and primero["truncado"] is False
    entorno.fabrica.assert_called_once_with(max_tokens=64)
    assert "Mensaje:\n" + DATOS["ejemplos"][0]["texto"] in entorno.modelo.prompts[0]


# K7
def _guion_mezclado(n, prompt):
    esperada = ESPERADAS[n - 1]
    otra = CATEGORIAS[(CATEGORIAS.index(esperada) + 1) % 3]
    return _mensaje({1: esperada.capitalize() + ".", 2: f"  {esperada}\n", 3: f"La categoria es {esperada}", 4: "no se", 5: otra}.get(n, esperada))


def test_k7_respuestas_mezcladas(tmp_path, monkeypatch, capsys):
    entorno = _preparar(tmp_path, monkeypatch, _guion_mezclado)
    assert medir.main(_argv(entorno)) == 0
    r = _json(entorno)["resumen"]
    assert (r["evaluadas"], r["exactas"], r["validas"], r["aciertos"]) == (24, 20, 22, 21)
    assert (r["fallos_formato_estricto"], r["fallos_formato_efectivo"]) == (4, 2)
    assert r["proporcion_estricta"] == 0.1667 and r["supera_umbral_estricto"] is True and r["supera_umbral_efectivo"] is False
    out = capsys.readouterr().out
    assert "Llamadas: 24 (con error: 0)" in out and "Respuestas evaluadas: 24" in out
    assert "Formato estricto (la respuesta es exactamente la categoria): 4 fallos de 24 (16.7 %)" in out
    assert "Formato efectivo (la categoria queda valida tras normalizar): 2 fallos de 24 (8.3 %)" in out
    assert "Umbral de la clase: 15 %. Supera el umbral (estricto): si" in out
    assert "Exactitud de la categoria: 21 de 24" in out
    for categoria, datos in r["por_categoria"].items():
        assert f"  {categoria}: {datos['aciertos']} de {datos['esperadas']}\n" in out
    assert "Costo total estimado de Claude: USD 0.00" in out


# K8
def test_k8_un_error_del_modelo_no_detiene_la_corrida(tmp_path, monkeypatch):
    def guion(n, prompt):
        if n == 2:
            raise RuntimeError("fallo")
        return _mensaje(ESPERADAS[n - 1])

    entorno = _preparar(tmp_path, monkeypatch, guion)
    assert medir.main(_argv(entorno)) == 0
    datos = _json(entorno)
    assert len(datos["resultados"]) == 24 and datos["resultados"][1]["error"] == "RuntimeError: fallo" and datos["resultados"][1]["tokens"] is None
    assert (datos["resumen"]["llamadas"], datos["resumen"]["con_error"], datos["resumen"]["evaluadas"]) == (24, 1, 23)


# K9
def test_k9_todas_las_llamadas_fallan(tmp_path, monkeypatch, capsys):
    def siempre(n, prompt):
        raise RuntimeError("fallo")

    entorno = _preparar(tmp_path, monkeypatch, siempre)
    assert medir.main(_argv(entorno)) == 1
    datos = _json(entorno)
    assert [r["error"] for r in datos["resultados"]] == ["RuntimeError: fallo"] * 24
    assert datos["incompleto"] is False and datos["error"] and "Error:" in capsys.readouterr().err


# K10
def test_k10_uso_en_cero_detiene_la_corrida(tmp_path, monkeypatch, capsys):
    entorno = _preparar(tmp_path, monkeypatch, lambda n, prompt: _mensaje(ESPERADAS[n - 1], *((0, 0) if n == 2 else (100, 5))))
    assert medir.main(_argv(entorno)) == 1
    datos = _json(entorno)
    assert len(datos["resultados"]) == 2 and datos["resultados"][1]["tokens"] is None and datos["incompleto"] is True
    assert f"Error: {datos['error']}\n" == capsys.readouterr().err and "sin uso no se controla el gasto" in datos["error"]


# K11
def test_k11_tope_de_gasto(tmp_path, monkeypatch, capsys):
    # cada llamada cuesta 1000 * 2e-6 + 50 * 1e-5 = USD 0.0025 > 0.001: tras la primera ya no cabe otra
    entorno = _preparar(tmp_path, monkeypatch, lambda n, prompt: _mensaje(ESPERADAS[n - 1], 1000, 50))
    assert medir.main(_argv(entorno, "--tope-usd", "0.001")) == 0
    datos = _json(entorno)
    assert len(entorno.modelo.prompts) == 1 and len(datos["resultados"]) == 1 and datos["incompleto"] is True
    assert "tope de gasto" in capsys.readouterr().out


# K12
def test_k12_ctrl_c_guarda_lo_hecho(tmp_path, monkeypatch, capsys):
    def guion(n, prompt):
        if n == 3:
            raise KeyboardInterrupt
        return _mensaje(ESPERADAS[n - 1])

    entorno = _preparar(tmp_path, monkeypatch, guion)
    assert medir.main(_argv(entorno)) == 130
    datos = _json(entorno)
    assert len(datos["resultados"]) == 2 and datos["incompleto"] is True
    assert "Interrumpido" in capsys.readouterr().err


# K13
def test_k13_respuesta_en_bloques_toma_solo_el_texto(tmp_path, monkeypatch):
    bloques = [{"type": "thinking", "thinking": "pienso", "signature": "firma"}, {"type": "text", "text": "incidente"}]
    entorno = _preparar(tmp_path, monkeypatch, lambda n, prompt: _mensaje(bloques if n == 1 else ESPERADAS[n - 1]))
    assert medir.main(_argv(entorno)) == 0
    primero = _json(entorno)["resultados"][0]
    assert primero["respuesta"] == "incidente" and primero["exacta"] is True and primero["correcta"] is True


# K14
def test_k14a_salida_igual_al_dataset(tmp_path, monkeypatch, capsys):
    entorno = _preparar(tmp_path, monkeypatch)
    ruta = _dataset(tmp_path, lambda datos: None)
    antes = Path(ruta).read_bytes()
    assert medir.main(_argv(entorno, salida=ruta, dataset=ruta)) == 1
    assert "Error:" in capsys.readouterr().err and Path(ruta).read_bytes() == antes
    assert entorno.modelo.prompts == [] and entorno.fabrica.call_count == 0


def test_k14b_salida_en_carpeta_inexistente(tmp_path, monkeypatch, capsys):
    entorno = _preparar(tmp_path, monkeypatch)
    destino = tmp_path / "no_hay" / "r.json"
    assert medir.main(_argv(entorno, salida=destino)) == 1
    assert "Error:" in capsys.readouterr().err and not destino.parent.exists()
    assert entorno.modelo.prompts == [] and entorno.fabrica.call_count == 0


# K15
def test_k15_texto_hostil(tmp_path, monkeypatch, capsys):
    entorno = _preparar(tmp_path, monkeypatch, lambda n, prompt: _mensaje("\x1b[31m" + "x" * 5000 if n == 1 else ESPERADAS[n - 1]))
    assert medir.main(_argv(entorno)) == 0
    captura = capsys.readouterr()
    assert chr(27) not in captura.out and chr(27) not in captura.err
    respuesta = _json(entorno)["resultados"][0]["respuesta"]
    assert len(respuesta) == 200 and respuesta.startswith(" [31mxxx") and chr(27) not in entorno.salida.read_text(encoding="utf-8")


# K16
def test_k16_hermetica(tmp_path, monkeypatch):
    def sin_red(*args, **kwargs):
        raise AssertionError("el script no debe abrir red")

    monkeypatch.setattr(socket.socket, "connect", sin_red)
    entorno = _preparar(tmp_path, monkeypatch)
    assert medir.main(_argv(entorno)) == 0


# K17
def test_k17_escritura_atomica(tmp_path, monkeypatch):
    entorno = _preparar(tmp_path, monkeypatch)
    entorno.salida.write_text("previo", encoding="utf-8")
    assert medir.main(_argv(entorno)) == 0
    assert len(_json(entorno)["resultados"]) == 24
    assert list(tmp_path.glob("**/*.tmp")) == []


# K18
def test_k18_la_salida_por_defecto():
    assert medir._crear_parser().parse_args([]).salida == str(RAIZ / "docs" / "resultados_clasificador_router.json")
