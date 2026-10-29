from collections import Counter

import pytest

from pipeline_texto import (
    coincidencia_por_palabras,
    limpiar_texto,
    procesar_texto,
    tokenizar,
    vectorizar,
)


# --- procesar_texto ---

def test_procesar_texto_normaliza_mayusculas_y_puntuacion():
    assert procesar_texto("El Cielo, es AZUL!") == ["el", "cielo", "es", "azul"]


def test_procesar_texto_solo_puntuacion_devuelve_lista_vacia():
    assert procesar_texto("¡¡¡...!!!") == []


# --- limpiar_texto ---

@pytest.mark.parametrize(
    "valor, tipo",
    [(None, "NoneType"), (42, "int"), (("texto", 1.0), "tuple")],
    ids=["none", "int", "tuple"],
)
def test_limpiar_texto_no_str_lanza_type_error(valor, tipo):
    with pytest.raises(TypeError, match=f"limpiar_texto espera str, no {tipo}"):
        limpiar_texto(valor)


def test_limpiar_texto_conserva_tildes_y_enie():
    assert limpiar_texto("Año pequeño: canción") == "año pequeño canción"


def test_limpiar_texto_conserva_numeros():
    assert limpiar_texto("Tiene 3 gatos") == "tiene 3 gatos"


def test_limpiar_texto_elimina_emojis():
    assert limpiar_texto("hola 😀") == "hola"


def test_limpiar_texto_colapsa_espacios_repetidos():
    assert limpiar_texto("  hola    mundo  ") == "hola mundo"


def test_limpiar_texto_conserva_guion_bajo():
    # comportamiento actual, no es un error
    assert limpiar_texto("snake_case") == "snake_case"


def test_limpiar_texto_une_decimales_sin_punto():
    # comportamiento actual, no es un error
    assert limpiar_texto("9.0 a 10.0") == "90 a 100"


# --- tokenizar y vectorizar ---

def test_tokenizar_texto_vacio_devuelve_lista_vacia():
    assert tokenizar("") == []


def test_vectorizar_cuenta_repeticiones():
    tokens = ["el", "gato", "y", "el", "perro"]
    assert vectorizar(tokens) == Counter({"el": 2, "gato": 1, "y": 1, "perro": 1})


# --- coincidencia_por_palabras ---

def test_coincidencia_cruda_vs_procesada_cielo_azul():
    referencia = "El cielo es azul."
    respuesta = "el cielo es azul"
    assert coincidencia_por_palabras(referencia.split(), respuesta.split()) == pytest.approx(0.5)
    assert coincidencia_por_palabras(procesar_texto(referencia), procesar_texto(respuesta)) == pytest.approx(1.0)


def test_coincidencia_ignora_el_orden():
    # límite documentado de la métrica: mismo conjunto de palabras, significado opuesto
    referencia = procesar_texto("El perro persiguió al gato.")
    respuesta = procesar_texto("El gato persiguió al perro.")
    assert coincidencia_por_palabras(referencia, respuesta) == pytest.approx(1.0)


def test_coincidencia_referencia_vacia_devuelve_cero():
    assert coincidencia_por_palabras([], ["hola"]) == 0.0


def test_coincidencia_no_distingue_repeticiones():
    assert coincidencia_por_palabras("el gato".split(), "el gato el gato".split()) == pytest.approx(1.0)
