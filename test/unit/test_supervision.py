import io
import math
import os
import pty
import threading
import time
from datetime import datetime, timedelta

import pytest

import supervision
from supervision import RegistroAprobaciones, crear_aprobador_interactivo, denegar_siempre, esperar_aprobacion

TIEMPO_PRUEBA = 0.2


@pytest.fixture
def tuberia():
    """Entrega (descriptor de lectura, escritor) sobre un os.pipe real y cierra ambos extremos al terminar."""
    lectura, escritura = os.pipe()
    escritor = os.fdopen(escritura, "w")
    yield lectura, escritor
    for cierre in (lambda: os.close(lectura), escritor.close):
        try:
            cierre()
        except OSError:
            pass


@pytest.mark.parametrize("respuesta", ["s\n", "S\n", " s \n"])
def test_esperar_aprobacion_aprueba_solo_con_s(respuesta, tuberia):
    entrada, escritor = tuberia
    escritor.write(respuesta)
    escritor.flush()
    assert esperar_aprobacion("¿Aprobar?", TIEMPO_PRUEBA, entrada) is True


@pytest.mark.parametrize("respuesta", ["n\n", "\n", "si\n", "sí\n"])
def test_esperar_aprobacion_niega_con_cualquier_otra_respuesta(respuesta, tuberia):
    entrada, escritor = tuberia
    escritor.write(respuesta)
    escritor.flush()
    assert esperar_aprobacion("¿Aprobar?", TIEMPO_PRUEBA, entrada) is False


def test_esperar_aprobacion_sin_respuesta_niega_al_vencer_el_tiempo(tuberia):
    entrada, _ = tuberia
    inicio = time.monotonic()
    assert esperar_aprobacion("¿Aprobar?", TIEMPO_PRUEBA, entrada) is False
    assert time.monotonic() - inicio < 2


def test_esperar_aprobacion_con_eof_niega_de_inmediato(tuberia):
    entrada, escritor = tuberia
    escritor.close()
    inicio = time.monotonic()
    assert esperar_aprobacion("¿Aprobar?", 5, entrada) is False
    assert time.monotonic() - inicio < 2


def test_esperar_aprobacion_con_entrada_sin_fileno_niega_sin_lanzar(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("s\n"))
    assert esperar_aprobacion("¿Aprobar?", TIEMPO_PRUEBA) is False


def test_esperar_aprobacion_aprueba_si_la_respuesta_llega_antes_del_plazo(tuberia):
    lectura, escritor = tuberia
    escritor.write("s\n")
    escritor.flush()
    assert esperar_aprobacion("¿Aprobar?", TIEMPO_PRUEBA, lectura) is True


def test_esperar_aprobacion_no_pierde_la_segunda_linea_escrita_de_una_vez(tuberia):
    lectura, escritor = tuberia
    escritor.write("n\ns\n")
    escritor.flush()
    assert [esperar_aprobacion("¿Aprobar?", TIEMPO_PRUEBA, lectura) for _ in range(2)] == [False, True]


def test_esperar_aprobacion_con_linea_parcial_vence_el_plazo_aunque_el_escritor_siga_abierto(tuberia):
    lectura, escritor = tuberia
    escritor.write("s")
    escritor.flush()
    inicio = time.monotonic()
    assert esperar_aprobacion("¿Aprobar?", 0.05, lectura) is False
    assert time.monotonic() - inicio < 1


def test_esperar_aprobacion_con_el_salto_de_linea_despues_del_plazo_niega(tuberia):
    lectura, escritor = tuberia
    escritor.write("s")
    escritor.flush()
    llega_tarde = threading.Timer(0.15, lambda: (escritor.write("\n"), escritor.flush()))
    llega_tarde.start()
    try:
        assert esperar_aprobacion("¿Aprobar?", 0.05, lectura) is False
    finally:
        llega_tarde.join()


def test_esperar_aprobacion_con_eof_sin_salto_de_linea_niega(tuberia):
    lectura, escritor = tuberia
    escritor.write("s")
    escritor.close()
    assert esperar_aprobacion("¿Aprobar?", TIEMPO_PRUEBA, lectura) is False


def test_esperar_aprobacion_con_una_linea_de_mas_de_1024_bytes_niega(tuberia):
    lectura, escritor = tuberia
    escritor.write("s" * 1030 + "\n")
    escritor.flush()
    assert esperar_aprobacion("¿Aprobar?", TIEMPO_PRUEBA, lectura) is False


def test_esperar_aprobacion_con_tiempo_invalido_niega_sin_leer_ni_lanzar(tuberia):
    lectura, escritor = tuberia
    escritor.write("s\n")
    escritor.flush()
    for tiempo in (None, "5", math.inf, math.nan, 0, -1, 3601, 1e10, True):
        assert esperar_aprobacion("¿Aprobar?", tiempo, lectura) is False, tiempo
    # la línea sigue en la tubería: ninguna de las llamadas anteriores la leyó
    assert esperar_aprobacion("¿Aprobar?", TIEMPO_PRUEBA, lectura) is True


def test_esperar_aprobacion_con_descriptor_invalido_niega_sin_lanzar(monkeypatch):
    lectura, escritura = os.pipe()
    os.close(lectura)
    os.close(escritura)
    for fd in (lectura, -1, 5000):
        assert esperar_aprobacion("¿Aprobar?", TIEMPO_PRUEBA, fd) is False, fd
    monkeypatch.setattr("sys.stdin", None)
    assert esperar_aprobacion("¿Aprobar?", TIEMPO_PRUEBA) is False


def test_esperar_aprobacion_descarta_lo_tecleado_antes_en_una_terminal():
    maestro, esclavo = pty.openpty()
    try:
        os.write(maestro, b"s\n")
        assert esperar_aprobacion("¿Aprobar?", TIEMPO_PRUEBA, esclavo) is False
    finally:
        os.close(maestro)
        os.close(esclavo)


def test_esperar_aprobacion_deja_pasar_el_keyboardinterrupt(tuberia, monkeypatch):
    def interrumpir(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(supervision.select, "select", interrumpir)
    with pytest.raises(KeyboardInterrupt):
        esperar_aprobacion("¿Aprobar?", TIEMPO_PRUEBA, tuberia[0])


def test_esperar_aprobacion_imprime_el_mensaje(tuberia, capsys):
    entrada, _ = tuberia
    esperar_aprobacion("mensaje-para-el-supervisor", TIEMPO_PRUEBA, entrada)
    assert "mensaje-para-el-supervisor" in capsys.readouterr().out


def test_registrar_guarda_los_datos_con_copia_de_argumentos_y_momento_utc():
    registro = RegistroAprobaciones()
    argumentos = {"host": "a.ejemplo.test"}
    registro.registrar("aislar_host_de_laboratorio", argumentos, True)
    argumentos["host"] = "otro.ejemplo.test"
    [entrada] = registro.registros
    assert entrada["herramienta"] == "aislar_host_de_laboratorio"
    assert entrada["argumentos"] == {"host": "a.ejemplo.test"}
    assert entrada["aprobado"] is True
    assert datetime.fromisoformat(entrada["momento"]).utcoffset() == timedelta(0)


def test_registros_devuelve_una_copia():
    registro = RegistroAprobaciones()
    registro.registrar("herramienta", {}, False)
    registro.registros.clear()
    assert len(registro.registros) == 1


def test_registros_devuelve_una_copia_profunda():
    registro = RegistroAprobaciones()
    registro.registrar("herramienta", {"host": "a.ejemplo.test"}, True)
    registro.registros[0]["argumentos"]["host"] = "otro.ejemplo.test"
    assert registro.registros[0]["argumentos"]["host"] == "a.ejemplo.test"


def test_denegar_siempre_devuelve_false():
    assert denegar_siempre("cualquier mensaje") is False


def test_crear_aprobador_interactivo_reenvia_mensaje_y_tiempo(monkeypatch):
    llamadas = []

    def espia(mensaje, tiempo_limite_segundos):
        llamadas.append((mensaje, tiempo_limite_segundos))
        return True

    monkeypatch.setattr(supervision, "esperar_aprobacion", espia)
    assert crear_aprobador_interactivo(0.3)("¿Aprobar?") is True
    assert llamadas == [("¿Aprobar?", 0.3)]
