import io
import os
import time
from datetime import datetime, timedelta

import pytest

import supervision
from supervision import RegistroAprobaciones, crear_aprobador_interactivo, denegar_siempre, esperar_aprobacion

TIEMPO_PRUEBA = 0.2


@pytest.fixture
def tuberia():
    """Entrega (entrada, escritor) sobre un os.pipe real y cierra ambos extremos al terminar."""
    lectura, escritura = os.pipe()
    entrada, escritor = os.fdopen(lectura, "r"), os.fdopen(escritura, "w")
    yield entrada, escritor
    for extremo in (entrada, escritor):
        if not extremo.closed:
            extremo.close()


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


def test_esperar_aprobacion_con_entrada_sin_fileno_niega_sin_lanzar():
    assert esperar_aprobacion("¿Aprobar?", TIEMPO_PRUEBA, io.StringIO("s\n")) is False


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
