import fnmatch
import importlib.util
import json
import os
import stat
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from langchain_core.agents import AgentAction

from herramientas_agente import construir_herramientas
from supervision import RegistroAprobaciones
from trazas import leer_trazas, marcar_repeticiones_sospechosas, marcar_riesgo_alto_sin_aprobacion, registrar_traza

RAIZ = Path(__file__).parents[2]
_spec = importlib.util.spec_from_file_location("revisar_trazas", RAIZ / "scripts" / "revisar_trazas.py")
revisar_trazas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(revisar_trazas)


def _paso(herramienta: str, argumentos: dict, observacion) -> tuple:
    return (AgentAction(tool=herramienta, tool_input=argumentos, log=""), observacion)


def _resultado(*pasos, salida: str = "listo") -> dict:
    return {"output": salida, "intermediate_steps": list(pasos)}


def _registro(session_id: str, herramientas: list[str], aprobaciones: list[dict] | None = None) -> dict:
    return {
        "session_id": session_id,
        "acciones": [{"herramienta": h, "argumentos": {}, "observacion": ""} for h in herramientas],
        "aprobaciones": aprobaciones or [],
    }


def _lineas(ruta) -> list[str]:
    return ruta.read_text(encoding="utf-8").splitlines()


def test_registrar_traza_escribe_una_linea_json_con_las_claves_esperadas(tmp_path):
    ruta = tmp_path / "trazas.jsonl"
    registrar_traza("s1", "hola", _resultado(), str(ruta))
    [linea] = _lineas(ruta)
    traza = json.loads(linea)
    assert set(traza) == {"timestamp", "session_id", "entrada", "acciones", "respuesta_final", "aprobaciones"}
    assert datetime.fromisoformat(traza["timestamp"]).utcoffset() == timedelta(0)
    assert (traza["session_id"], traza["entrada"], traza["respuesta_final"]) == ("s1", "hola", "listo")


def test_registrar_traza_guarda_acciones_y_convierte_la_observacion_a_texto(tmp_path):
    ruta = tmp_path / "trazas.jsonl"
    resultado = _resultado(_paso("calcular", {"expresion": "1 + 1"}, "2"), _paso("calcular", {"expresion": "2 + 2"}, 4))
    registrar_traza("s1", "x", resultado, str(ruta))
    acciones = json.loads(_lineas(ruta)[0])["acciones"]
    assert acciones == [
        {"herramienta": "calcular", "argumentos": {"expresion": "1 + 1"}, "observacion": "2"},
        {"herramienta": "calcular", "argumentos": {"expresion": "2 + 2"}, "observacion": "4"},
    ]


def test_registrar_traza_escribe_las_aprobaciones_tal_cual(tmp_path):
    ruta = tmp_path / "trazas.jsonl"
    aprobaciones = [
        {
            "herramienta": "aislar_host_de_laboratorio",
            "argumentos": {"host": "host-lab-01.ejemplo.test"},
            "aprobado": False,
            "momento": "2026-11-11T00:00:00+00:00",
        }
    ]
    registrar_traza("s1", "x", _resultado(), str(ruta), aprobaciones=aprobaciones)
    assert json.loads(_lineas(ruta)[0])["aprobaciones"] == aprobaciones


def test_registrar_traza_sin_intermediate_steps_deja_acciones_vacias(tmp_path):
    ruta = tmp_path / "trazas.jsonl"
    registrar_traza("s1", "x", {"output": "listo"}, str(ruta))
    assert json.loads(_lineas(ruta)[0])["acciones"] == []


def test_registrar_traza_agrega_sin_tocar_la_linea_anterior(tmp_path):
    ruta = tmp_path / "trazas.jsonl"
    registrar_traza("s1", "uno", _resultado(), str(ruta))
    primera = _lineas(ruta)[0]
    registrar_traza("s2", "dos", _resultado(), str(ruta))
    lineas = _lineas(ruta)
    assert len(lineas) == 2
    assert lineas[0] == primera


def test_registrar_traza_crea_el_directorio_padre(tmp_path):
    ruta = tmp_path / "nuevo" / "trazas.jsonl"
    registrar_traza("s1", "x", _resultado(), str(ruta))
    assert ruta.exists()


def test_registrar_traza_escribe_los_acentos_en_utf8(tmp_path):
    ruta = tmp_path / "trazas.jsonl"
    registrar_traza("s1", "¿Qué pasó?", _resultado(salida="é"), str(ruta))
    contenido = ruta.read_bytes()
    assert "é".encode("utf-8") in contenido
    assert b"\\u00e9" not in contenido


def _linea(session_id: str) -> str:
    return json.dumps(_registro(session_id, []))


def test_leer_trazas_devuelve_los_registros_y_salta_lineas_en_blanco(tmp_path):
    ruta = tmp_path / "trazas.jsonl"
    ruta.write_text(f"{_linea('a')}\n\n   \n{_linea('b')}\n", encoding="utf-8")
    assert leer_trazas(str(ruta)) == [_registro("a", []), _registro("b", [])]


def test_leer_trazas_con_linea_corrupta_lanza_valueerror_con_su_numero(tmp_path):
    ruta = tmp_path / "trazas.jsonl"
    ruta.write_text(f"{_linea('a')}\nno es json\n", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        leer_trazas(str(ruta))
    # sin match="2": la ruta temporal puede contener un 2
    assert "La línea 2 de" in str(error.value)


def test_leer_trazas_de_un_archivo_inexistente_deja_pasar_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError):
        leer_trazas(str(tmp_path / "no_existe.jsonl"))


@pytest.mark.parametrize(
    "herramientas, esperadas",
    [
        (["calcular", "calcular"], ["Revisar: s1 usó 'calcular' dos veces seguidas"]),
        (["calcular", "buscar_en_documentacion", "calcular"], []),
    ],
    ids=["seguidas", "separadas"],
)
def test_marcar_repeticiones_solo_marca_las_consecutivas(herramientas, esperadas):
    assert marcar_repeticiones_sospechosas([_registro("s1", herramientas)]) == esperadas


@pytest.mark.parametrize(
    "herramienta, aprobaciones, cantidad",
    [
        ("aislar_host_de_laboratorio", [], 1),
        ("aislar_host_de_laboratorio", [{"herramienta": "aislar_host_de_laboratorio", "aprobado": False}], 0),
        ("calcular", [], 0),
    ],
    ids=["alto_sin_registro", "alto_con_negativa_registrada", "bajo_sin_registro"],
)
def test_marcar_riesgo_alto_exige_una_aprobacion_registrada_solo_en_riesgo_alto(herramienta, aprobaciones, cantidad):
    marcas = marcar_riesgo_alto_sin_aprobacion([_registro("s1", [herramienta], aprobaciones)])
    assert len(marcas) == cantidad
    if cantidad:
        assert marcas == [f"Revisar: s1 usó la herramienta de riesgo alto '{herramienta}' sin registro de aprobación"]


def test_un_intento_rechazado_por_host_no_permitido_no_se_marca_como_riesgo_alto_sin_aprobacion():
    registro = RegistroAprobaciones()
    herramienta = construir_herramientas(None, registro)[3]
    entrada = {"host": "servidor-prod.example.com", "motivo": "x"}
    assert herramienta.invoke(entrada).startswith("Host no permitido")
    traza = _registro("s1", ["aislar_host_de_laboratorio"], registro.registros)
    assert marcar_riesgo_alto_sin_aprobacion([traza]) == []


def test_revisar_trazas_imprime_las_marcas_y_devuelve_0(tmp_path, capsys):
    ruta = tmp_path / "trazas.jsonl"
    resultado = _resultado(_paso("calcular", {}, "1"), _paso("calcular", {}, "2"))
    registrar_traza("s1", "x", resultado, str(ruta))
    assert revisar_trazas.main(["--trazas", str(ruta)]) == 0
    assert "Revisar: s1 usó 'calcular' dos veces seguidas" in capsys.readouterr().out


def test_revisar_trazas_con_linea_corrupta_devuelve_1_y_menciona_la_linea(tmp_path, capsys):
    ruta = tmp_path / "trazas.jsonl"
    ruta.write_text(f"{_linea('a')}\nno es json\n", encoding="utf-8")
    assert revisar_trazas.main(["--trazas", str(ruta)]) == 1
    error = capsys.readouterr().err
    assert error.startswith("Error:") and "línea" in error


def test_revisar_trazas_con_archivo_inexistente_devuelve_1_y_avisa_por_stderr(tmp_path, capsys):
    assert revisar_trazas.main(["--trazas", str(tmp_path / "no_existe.jsonl")]) == 1
    assert capsys.readouterr().err.startswith("Error:")


def test_registrar_traza_crea_el_archivo_con_modo_0600_y_el_directorio_con_0700(tmp_path):
    anterior = os.umask(0o022)
    try:
        ruta = tmp_path / "nuevo" / "trazas.jsonl"
        registrar_traza("s1", "x", _resultado(), str(ruta))
    finally:
        os.umask(anterior)
    assert stat.S_IMODE(os.stat(ruta).st_mode) == 0o600
    assert stat.S_IMODE(os.stat(ruta.parent).st_mode) == 0o700


def test_registrar_traza_no_cambia_el_modo_de_un_archivo_existente(tmp_path):
    ruta = tmp_path / "trazas.jsonl"
    ruta.write_text("", encoding="utf-8")
    os.chmod(ruta, 0o640)
    registrar_traza("s1", "x", _resultado(), str(ruta))
    assert stat.S_IMODE(os.stat(ruta).st_mode) == 0o640


def test_leer_trazas_rechaza_los_registros_sin_la_forma_esperada_con_su_numero_de_linea(tmp_path):
    malos = [
        "[]", "5", "null", "{}",
        '{"acciones": null, "aprobaciones": []}',
        '{"acciones": [{"argumentos": {}}], "aprobaciones": []}',
    ]
    for mala in malos:
        ruta = tmp_path / "trazas.jsonl"
        ruta.write_text(f"{_linea('a')}\n{mala}\n", encoding="utf-8")
        with pytest.raises(ValueError) as error:
            leer_trazas(str(ruta))
        assert str(error.value).startswith("Línea 2: la traza no tiene la forma esperada ("), mala


def test_revisar_trazas_con_un_directorio_devuelve_1_sin_traceback_y_con_un_archivo_vacio_avisa(tmp_path, capsys):
    assert revisar_trazas.main(["--trazas", str(tmp_path)]) == 1
    error = capsys.readouterr().err
    assert error.startswith("Error:") and "Traceback" not in error
    vacio = tmp_path / "vacio.jsonl"
    vacio.write_text("", encoding="utf-8")
    assert revisar_trazas.main(["--trazas", str(vacio)]) == 0
    assert "Archivo sin registros." in capsys.readouterr().out


def test_revisar_trazas_no_imprime_crudos_los_controles_del_session_id(tmp_path, capsys):
    ruta = tmp_path / "trazas.jsonl"
    registro = _registro("a\x1b[2Kb", ["calcular", "calcular"])
    ruta.write_text(json.dumps(registro) + "\n", encoding="utf-8")
    assert revisar_trazas.main(["--trazas", str(ruta)]) == 0
    impreso = capsys.readouterr().out
    assert "\x1b" not in impreso and "usó 'calcular' dos veces seguidas" in impreso


def test_una_herramienta_ausente_de_riesgo_se_marca_como_sin_nivel_de_riesgo():
    [marca] = marcar_riesgo_alto_sin_aprobacion([_registro("s1", ["herramienta_nueva"])])
    assert "sin nivel de riesgo" in marca and "herramienta_nueva" in marca


def test_el_gitignore_deja_fuera_de_git_las_trazas_en_cualquier_ruta_razonable():
    reglas = (RAIZ / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert all(regla in reglas for regla in ("trazas/", "*trazas*.jsonl", "trazas*/"))
    assert all(fnmatch.fnmatch(nombre, "*trazas*.jsonl") for nombre in ("trazas_agente.jsonl", "mis_trazas.jsonl"))
    assert fnmatch.fnmatch("trazas2", "trazas*")
