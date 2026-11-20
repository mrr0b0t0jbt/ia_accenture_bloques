import copy
import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest

import dataset_tuning

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_SCRIPT = RAIZ / "scripts" / "preparar_dataset_tuning.py"
RUTA_DATASET = RAIZ / "JSON" / "dataset_tuning_clasificacion.json"


def _cargar_script():
    spec = importlib.util.spec_from_file_location("preparar_dataset_tuning", RUTA_SCRIPT)
    modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modulo)
    return modulo


preparar = _cargar_script()


def _escribir_dataset(tmp_path, cambiar) -> str:
    """Copia el dataset real, le aplica cambiar(datos) y lo escribe en tmp_path; devuelve la ruta."""
    datos = copy.deepcopy(json.loads(RUTA_DATASET.read_text(encoding="utf-8")))
    cambiar(datos)
    ruta = tmp_path / "dataset.json"
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    return str(ruta)


# P1
def test_p1_flujo_feliz_con_el_dataset_real(capsys):
    assert preparar.main([]) == 0
    salida = capsys.readouterr().out
    assert "24 ejemplos, 0 problemas encontrados" in salida
    assert "categoria:" in salida and "prioridad:" in salida
    assert salida.count("razon max/min: 1.0x") == 2
    assert "aviso" not in salida


# P2
def test_p2_un_problema_sale_con_1(tmp_path, capsys):
    def cambiar(datos):
        datos["ejemplos"][3]["salida"]["prioridad"] = "urgente"

    assert preparar.main(["--dataset", _escribir_dataset(tmp_path, cambiar)]) == 1
    salida = capsys.readouterr()
    assert ' - ejemplo 3: output tiene un valor no permitido para "prioridad"' in salida.out
    assert "Error: el dataset tiene 1 problemas; no esta listo para enviarse a entrenar." in salida.err


# P3
@pytest.mark.parametrize("contenido", [None, "{roto", "[1, 2]"])
def test_p3_dataset_ilegible_sale_con_1_sin_traceback(tmp_path, capsys, contenido):
    ruta = tmp_path / "d.json"
    if contenido is not None:
        ruta.write_text(contenido, encoding="utf-8")
    assert preparar.main(["--dataset", str(ruta)]) == 1
    error = capsys.readouterr().err
    assert error.startswith("Error:") and "Traceback" not in error


# P4
def test_p4_el_defecto_es_relativo_a_la_raiz_del_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert preparar.main([]) == 0


# P5
def test_p5_argumento_desconocido():
    with pytest.raises(SystemExit) as salida:
        preparar.main(["--no-existe"])
    assert salida.value.code == 2


# P6
def test_p6_demostrar_validador_detecta_los_cuatro_problemas(capsys):
    assert preparar.main(["--demostrar-validador"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "26 ejemplos, 4 problemas encontrados",
        " - ejemplo 24: text_input vacio",
        ' - ejemplo 24: output no tiene la clave "prioridad"',
        " - ejemplo 25: text_input duplicado del ejemplo 0 con otra salida",
        " - ejemplo 25: output no es JSON valido",
    ]


def test_p6_validador_ciego_sale_con_1(monkeypatch, capsys):
    monkeypatch.setattr(dataset_tuning, "validar_dataset", lambda *a, **k: [])
    assert preparar.main(["--demostrar-validador"]) == 1
    assert "Error: el validador no detecto lo esperado." in capsys.readouterr().err


# P7
def test_p7_texto_hostil_no_llega_en_crudo_a_la_consola(tmp_path, capsys):
    def cambiar(datos):
        datos["ejemplos"][0]["salida"]["categoria"] = "\x1b[31mX"

    assert preparar.main(["--dataset", _escribir_dataset(tmp_path, cambiar)]) == 1
    salida = capsys.readouterr()
    assert "[31mX" in salida.out
    assert chr(27) not in salida.out and chr(27) not in salida.err


# P8
def test_p8_hermetica(monkeypatch, capsys):
    for nombre in ("dotenv", "chromadb", "langchain_google_genai", "anthropic"):
        monkeypatch.setitem(sys.modules, nombre, None)

    def sin_red(*a, **k):
        raise AssertionError("el script no debe abrir red")

    monkeypatch.setattr(socket.socket, "connect", sin_red)
    assert _cargar_script().main([]) == 0
    assert "0 problemas encontrados" in capsys.readouterr().out


# P9
def test_p9_no_escribe_nada(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    antes = RUTA_DATASET.read_text(encoding="utf-8")
    assert preparar.main([]) == 0
    assert preparar.main(["--demostrar-validador"]) == 0
    assert list(tmp_path.iterdir()) == []
    assert RUTA_DATASET.read_text(encoding="utf-8") == antes


# I2
def _reasignar(datos, clave, asignacion):
    """Pone en cada ejemplo i el valor asignacion(i) de la clave de salida."""
    for i, ejemplo in enumerate(datos["ejemplos"]):
        ejemplo["salida"][clave] = asignacion(i)


def _sin_vulnerabilidad(datos):
    for ejemplo in datos["ejemplos"]:
        if ejemplo["salida"]["categoria"] == "vulnerabilidad":
            ejemplo["salida"]["categoria"] = "consulta"


def test_i2_g8_imprime_los_avisos(tmp_path, capsys):
    def cambiar(datos):  # 21 incidente, 2 consulta, 1 vulnerabilidad: razon 21/1
        _reasignar(datos, "categoria", lambda i: "vulnerabilidad" if i == 21 else "consulta" if i > 21 else "incidente")

    assert preparar.main(["--dataset", _escribir_dataset(tmp_path, cambiar)]) == 0
    salida = capsys.readouterr().out
    assert "  razon max/min: 21.0x" in salida
    assert "aviso: la razon max/min de categoria es 21.0x (mayor que 2.0)" in salida


def test_i2_g9_categoria_sin_ejemplos(tmp_path, capsys):
    preparar.main(["--dataset", _escribir_dataset(tmp_path, _sin_vulnerabilidad)])
    lineas = capsys.readouterr().out.splitlines()
    assert "  razon max/min: n/d" in lineas
    assert "aviso: categoria no tiene ejemplos de: vulnerabilidad" in lineas


def test_i2_c1d_surrogate_suelto_en_el_texto_sale_con_1(tmp_path, capsys):
    def cambiar(datos):  # json.dumps escapa el surrogate como \ud800
        datos["ejemplos"][5]["texto"] += chr(0xD800)

    assert preparar.main(["--dataset", _escribir_dataset(tmp_path, cambiar)]) == 1
    assert " - ejemplo 5: text_input no se puede codificar en UTF-8" in capsys.readouterr().out


def test_i2_c2a_categoria_sin_ejemplos_sale_con_1(tmp_path, capsys):
    assert preparar.main(["--dataset", _escribir_dataset(tmp_path, _sin_vulnerabilidad)]) == 1
    salida = capsys.readouterr()
    assert "Error: categoria no tiene ejemplos de: vulnerabilidad; no esta listo para enviarse a entrenar." in salida.err
    assert "aviso: categoria no tiene ejemplos de: vulnerabilidad" in salida.out


def test_i2_c2b_faltan_valores_en_las_dos_claves(tmp_path, capsys):
    def cambiar(datos):
        _sin_vulnerabilidad(datos)
        for ejemplo in datos["ejemplos"]:
            if ejemplo["salida"]["prioridad"] == "baja":
                ejemplo["salida"]["prioridad"] = "media"

    assert preparar.main(["--dataset", _escribir_dataset(tmp_path, cambiar)]) == 1
    error = capsys.readouterr().err
    de_categoria = "Error: categoria no tiene ejemplos de: vulnerabilidad; no esta listo para enviarse a entrenar."
    de_prioridad = "Error: prioridad no tiene ejemplos de: baja; no esta listo para enviarse a entrenar."
    assert de_categoria in error and de_prioridad in error
    assert error.index(de_categoria) < error.index(de_prioridad)


def test_i2_c2c_desbalance_sin_ausentes_solo_avisa(tmp_path, capsys):
    def cambiar(datos):  # 12 incidente, 8 consulta, 4 vulnerabilidad: razon 3.0
        _reasignar(datos, "categoria", lambda i: "incidente" if i < 12 else "consulta" if i < 20 else "vulnerabilidad")

    assert preparar.main(["--dataset", _escribir_dataset(tmp_path, cambiar)]) == 0
    salida = capsys.readouterr()
    assert "aviso: la razon max/min de categoria es 3.0x (mayor que 2.0)" in salida.out
    assert salida.err == ""


def test_i2_g10_no_acepta_abreviaturas():
    with pytest.raises(SystemExit) as salida:
        preparar.main(["--datas", "x"])
    assert salida.value.code == 2


def test_i2_g11_problemas_de_forma(tmp_path, capsys):
    assert preparar.main(["--dataset", _escribir_dataset(tmp_path, lambda d: d.update(claves=["categoria"]))]) == 1
    salida = capsys.readouterr()
    assert " - claves del archivo distintas de CLAVES" in salida.out
    assert "Error: el dataset tiene 1 problemas; no esta listo para enviarse a entrenar." in salida.err


# J2
def _diez_de_incidente(datos, prioridad_del_ultimo=None):
    """Deja 10 ejemplos válidos, todos de la categoría incidente, con las prioridades alta, media y baja repetidas."""
    prioridades = ("alta", "media", "baja")
    datos["ejemplos"] = [
        {"id": f"ej_{i:02d}", "texto": f"mensaje distinto numero {i}", "salida": {"categoria": "incidente", "prioridad": prioridades[i % 3]}}
        for i in range(10)
    ]
    if prioridad_del_ultimo:
        datos["ejemplos"][9]["texto"] = datos["ejemplos"][0]["texto"]
        datos["ejemplos"][9]["salida"]["prioridad"] = prioridad_del_ultimo


def test_j2_h17_categorias_ausentes_en_el_orden_de_valores_permitidos(tmp_path, capsys):
    assert preparar.main(["--dataset", _escribir_dataset(tmp_path, _diez_de_incidente)]) == 1
    lineas = capsys.readouterr().err.splitlines()
    assert lineas == ["Error: categoria no tiene ejemplos de: vulnerabilidad, consulta; no esta listo para enviarse a entrenar."]


def test_j2_h18_problemas_y_ausentes_se_reportan_los_dos_en_orden(tmp_path, capsys):
    def cambiar(datos):  # el ejemplo 9 repite el texto del 0 con otra salida: 1 problema
        _diez_de_incidente(datos, prioridad_del_ultimo="baja")

    assert preparar.main(["--dataset", _escribir_dataset(tmp_path, cambiar)]) == 1
    lineas = capsys.readouterr().err.splitlines()
    assert lineas == [
        "Error: el dataset tiene 1 problemas; no esta listo para enviarse a entrenar.",
        "Error: categoria no tiene ejemplos de: vulnerabilidad, consulta; no esta listo para enviarse a entrenar.",
    ]
