import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

import decision_personalizacion as dp

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_SCRIPT = RAIZ / "scripts" / "decidir_personalizacion.py"

CASO_1 = dict(zip(dp.PREGUNTAS, [True, True, False, True, False]))
CASO_2 = dict(zip(dp.PREGUNTAS, [False, True, True, False, False]))
TODO_FALSE = {k: False for k in dp.PREGUNTAS}
PROBLEMA = "El clasificador de errores no responde con un formato consistente."


def _cargar_script():
    spec = importlib.util.spec_from_file_location("decidir_personalizacion", RUTA_SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


decidir = _cargar_script()


def _datos(respuestas=CASO_2, **cambios) -> dict:
    datos = {"problema": PROBLEMA, "respuestas": respuestas, "evidencia": {k: f"evidencia de {k}" for k in dp.PREGUNTAS}}
    datos.update(cambios)
    return datos


def _archivo(tmp_path, contenido, nombre="respuestas.json") -> str:
    ruta = tmp_path / nombre
    ruta.write_text(contenido if isinstance(contenido, str) else json.dumps(contenido), encoding="utf-8")
    return str(ruta)


# X1
def test_x1_flujo_feliz_caso_2(tmp_path, capsys):
    assert decidir.main(["--respuestas", _archivo(tmp_path, _datos())]) == 0
    esperado = [
        f"Problema: {PROBLEMA}",
        "Puntajes:",
        "  Fine-tuning completo: 3 puntos",
        "  PEFT: 3 puntos",
        "  RAG: 0 puntos",
        "Empate entre: Fine-tuning completo, PEFT",
        "Sensibilidad (una respuesta a la vez):",
        "  informacion_cambia_seguido -> True: RAG, Fine-tuning completo, PEFT  [CAMBIA]",
        "  problema_de_formato_o_tono -> False: Fine-tuning completo  [CAMBIA]",
        "  dataset_grande_y_de_calidad -> False: PEFT  [CAMBIA]",
        "  presupuesto_limitado_o_iterar_rapido -> True: PEFT  [CAMBIA]",
        "  necesita_multiples_variantes -> True: PEFT  [CAMBIA]",
        "Evidencia:",
    ] + [f"  {k}: {CASO_2[k]} -- evidencia de {k}" for k in dp.PREGUNTAS]
    assert capsys.readouterr().out.splitlines() == esperado


# X2
def test_x2_caso_1_recomienda_rag(tmp_path, capsys):
    assert decidir.main(["--respuestas", _archivo(tmp_path, _datos(CASO_1))]) == 0
    out = capsys.readouterr().out
    assert "Recomendacion: RAG\n" in out and "Empate" not in out


# X3
def test_x3_todo_false_sin_necesidad(tmp_path, capsys):
    assert decidir.main(["--respuestas", _archivo(tmp_path, _datos(TODO_FALSE))]) == 0
    out = capsys.readouterr().out
    assert "\nNinguna tecnica obtuvo puntos: ninguna respuesta justifica RAG, fine-tuning ni PEFT\n" in out
    assert "Recomendacion" not in out and "Empate" not in out


# X4
def _sin(clave):
    datos = _datos()
    del datos[clave]
    return datos


def _evidencia_sin(pregunta):
    datos = _datos()
    del datos["evidencia"][pregunta]
    return datos


@pytest.mark.parametrize(
    "contenido,mensaje",
    [
        (None, "el archivo de respuestas no existe o no es un archivo"),
        ("{roto", "el archivo de respuestas no es un JSON valido"),
        ("[1, 2]", "el JSON de nivel superior no es un objeto"),
        (_sin("respuestas"), "falta la clave respuestas"),
        (_datos({**CASO_2, "problema_de_formato_o_tono": "no"}), 'la respuesta "problema_de_formato_o_tono" no es True ni False'),
        (
            _datos(evidencia={**_datos()["evidencia"], "dataset_grande_y_de_calidad": "  "}),
            'la evidencia de "dataset_grande_y_de_calidad" esta vacia o no es texto',
        ),
        (_evidencia_sin("necesita_multiples_variantes"), 'falta la evidencia de "necesita_multiples_variantes"'),
    ],
    ids=["inexistente", "json_invalido", "lista", "sin_respuestas", "respuesta_no_bool", "evidencia_vacia", "falta_evidencia"],
)
def test_x4_archivo_invalido(tmp_path, capsys, contenido, mensaje):
    ruta = str(tmp_path / "no_existe.json") if contenido is None else _archivo(tmp_path, contenido)
    assert decidir.main(["--respuestas", ruta]) == 1
    captura = capsys.readouterr()
    assert f"Error: {mensaje}\n" in captura.err and "Traceback" not in captura.err and captura.out == ""


# X5
def test_x5_salida_escribe_el_esqueleto(tmp_path):
    salida = tmp_path / "decision.md"
    assert decidir.main(["--respuestas", _archivo(tmp_path, _datos()), "--salida", str(salida)]) == 0
    esperado = dp.generar_esqueleto_decision(PROBLEMA, dp.recomendar_estrategia(CASO_2)) + "\n"
    assert salida.read_text(encoding="utf-8") == esperado


def test_x5b_salida_existente_no_se_pisa(tmp_path, capsys):
    salida = tmp_path / "decision.md"
    salida.write_text("previo", encoding="utf-8")
    assert decidir.main(["--respuestas", _archivo(tmp_path, _datos()), "--salida", str(salida)]) == 1
    captura = capsys.readouterr()
    assert salida.read_text(encoding="utf-8") == "previo" and captura.out == "" and "Error:" in captura.err


# X6
def test_x6_salida_en_carpeta_inexistente(tmp_path, capsys):
    salida = tmp_path / "no_hay" / "decision.md"
    assert decidir.main(["--respuestas", _archivo(tmp_path, _datos()), "--salida", str(salida)]) == 1
    assert not salida.parent.exists() and "Error:" in capsys.readouterr().err


# X7
def test_x7a_costo_sin_revision(capsys):
    assert decidir.main(["--estimar-etiquetado", "200", "5", "25"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Etiquetado base: 16.67 horas, aprox $416.67",
        "Segunda revision: 0.00 horas, aprox $0.00",
        "Total: 16.67 horas, aprox $416.67",
        "Jornadas de 8 horas: 2.1",
    ]


def test_x7b_costo_con_revision(capsys):
    assert decidir.main(["--estimar-etiquetado", "500", "4", "20", "--revision", "0.2", "2"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "Etiquetado base: 33.33 horas, aprox $666.67",
        "Segunda revision: 3.33 horas, aprox $66.67",
        "Total: 36.67 horas, aprox $733.33",
        "Jornadas de 8 horas: 4.6",
    ]


def test_x7c_costo_invalido(capsys):
    assert decidir.main(["--estimar-etiquetado", "-5", "5", "25"]) == 1
    assert "Error: num_ejemplos debe ser un entero mayor o igual que 0" in capsys.readouterr().err


# X8
def test_x8a_sin_modo():
    with pytest.raises(SystemExit) as salida:
        decidir.main([])
    assert salida.value.code == 2


def test_x8b_los_dos_modos_a_la_vez(tmp_path):
    with pytest.raises(SystemExit) as salida:
        decidir.main(["--respuestas", _archivo(tmp_path, _datos()), "--estimar-etiquetado", "1", "1", "1"])
    assert salida.value.code == 2


# X9
def test_x9_texto_hostil_no_llega_crudo_a_la_consola(tmp_path, capsys):
    evidencia = {**_datos()["evidencia"], "informacion_cambia_seguido": "hola \x1b[31mrojo"}
    assert decidir.main(["--respuestas", _archivo(tmp_path, _datos(evidencia=evidencia))]) == 1
    captura = capsys.readouterr()
    assert chr(27) not in captura.out and chr(27) not in captura.err
    assert 'la evidencia de "informacion_cambia_seguido" tiene caracteres de control' in captura.err


# X10
def test_x10_hermetica(tmp_path, monkeypatch, capsys):
    for nombre in ("dotenv", "chromadb", "langchain_google_genai", "anthropic"):
        monkeypatch.setitem(sys.modules, nombre, None)

    def sin_red(*args, **kwargs):
        raise AssertionError("el script no debe abrir red")

    monkeypatch.setattr(socket.socket, "connect", sin_red)
    assert _cargar_script().main(["--respuestas", _archivo(tmp_path, _datos())]) == 0
    assert "Empate entre:" in capsys.readouterr().out


# X11
def test_x11_sin_salida_no_escribe_nada(tmp_path, monkeypatch):
    ruta = _archivo(tmp_path, _datos())
    antes = Path(ruta).read_bytes()
    vacio = tmp_path / "vacio"
    vacio.mkdir()
    monkeypatch.chdir(vacio)
    assert decidir.main(["--respuestas", ruta]) == 0
    assert list(vacio.iterdir()) == [] and Path(ruta).read_bytes() == antes


# J2
def _archivo_de_bytes(tmp_path, contenido: bytes) -> str:
    ruta = tmp_path / "respuestas.json"
    ruta.write_bytes(contenido)
    return str(ruta)


def _relleno(datos: dict, tamano: int) -> bytes:
    """JSON válido de los datos rellenado con espacios al final hasta pesar exactamente tamano bytes."""
    base = json.dumps(datos).encode("utf-8")
    assert len(base) <= tamano
    return base + b" " * (tamano - len(base))


def test_j2_h3_fila_sin_ganadoras_dice_ninguna(tmp_path, capsys):
    respuestas = dict(zip(dp.PREGUNTAS, [True, False, False, False, False]))
    assert decidir.main(["--respuestas", _archivo(tmp_path, _datos(respuestas))]) == 0
    assert "  informacion_cambia_seguido -> False: ninguna  [CAMBIA]" in capsys.readouterr().out.splitlines()


def test_j2_h4_evidencia_que_es_una_lista(tmp_path, capsys):
    assert decidir.main(["--respuestas", _archivo(tmp_path, _datos(evidencia=[]))]) == 1
    assert "Error: la clave evidencia no es un objeto\n" in capsys.readouterr().err


def test_j2_h5_cien_mil_corchetes_no_es_un_json_valido(tmp_path, capsys):
    ruta = _archivo_de_bytes(tmp_path, b"[" * 100000)
    assert Path(ruta).stat().st_size == 100000
    assert decidir.main(["--respuestas", ruta]) == 1
    err = capsys.readouterr().err
    assert "Error: el archivo de respuestas no es un JSON valido\n" in err and "Traceback" not in err


def test_j2_h6a_archivo_de_exactamente_el_maximo_de_bytes(tmp_path):
    ruta = _archivo_de_bytes(tmp_path, _relleno(_datos(), 100000))
    assert Path(ruta).stat().st_size == 100000
    assert decidir.main(["--respuestas", ruta]) == 0


def test_j2_h6b_un_byte_mas_que_el_maximo(tmp_path, capsys):
    ruta = _archivo_de_bytes(tmp_path, _relleno(_datos(), 100001))
    assert decidir.main(["--respuestas", ruta]) == 1
    assert "Error: el archivo de respuestas pesa mas de 100000 bytes\n" in capsys.readouterr().err


def test_j2_h7_utf8_real_con_tildes_y_enie(tmp_path, capsys):
    problema = "análisis de niños con tildes: análisis, cañón"
    ruta = tmp_path / "respuestas.json"
    ruta.write_text(json.dumps(_datos(problema=problema), ensure_ascii=False), encoding="utf-8")
    assert decidir.main(["--respuestas", str(ruta)]) == 0
    assert f"Problema: {problema}" in capsys.readouterr().out.splitlines()


def _con_evidencia_de(largo: int) -> dict:
    return _datos(evidencia={**_datos()["evidencia"], "informacion_cambia_seguido": "a" * largo})


def test_j2_h8a_evidencia_de_exactamente_500_caracteres(tmp_path):
    assert decidir.main(["--respuestas", _archivo(tmp_path, _con_evidencia_de(500))]) == 0


def test_j2_h8b_evidencia_de_501_caracteres(tmp_path, capsys):
    assert decidir.main(["--respuestas", _archivo(tmp_path, _con_evidencia_de(501))]) == 1
    assert 'Error: la evidencia de "informacion_cambia_seguido" supera los 500 caracteres\n' in capsys.readouterr().err


def test_j2_h9_no_acepta_abreviaturas(tmp_path):
    with pytest.raises(SystemExit) as salida:
        decidir.main(["--resp", _archivo(tmp_path, _datos())])
    assert salida.value.code == 2


def test_j2_h10a_salida_sin_respuestas(capsys):
    with pytest.raises(SystemExit) as salida:
        decidir.main(["--estimar-etiquetado", "10", "5", "25", "--salida", "x"])
    assert salida.value.code == 2 and "--salida solo se usa con --respuestas" in capsys.readouterr().err


def test_j2_h10b_revision_sin_estimar_etiquetado(tmp_path, capsys):
    with pytest.raises(SystemExit) as salida:
        decidir.main(["--respuestas", _archivo(tmp_path, _datos()), "--revision", "0.1", "2"])
    assert salida.value.code == 2 and "--revision solo se usa con --estimar-etiquetado" in capsys.readouterr().err


def test_j2_h11_separador_de_linea_no_llega_a_la_consola(tmp_path, capsys):
    evidencia = {**_datos()["evidencia"], "informacion_cambia_seguido": "a" + chr(0x2028) + "b"}  # categoría Zl, no es de control
    assert decidir.main(["--respuestas", _archivo(tmp_path, _datos(evidencia=evidencia))]) == 0
    out = capsys.readouterr().out
    assert chr(0x2028) not in out and "  informacion_cambia_seguido: False -- a b" in out.splitlines()
