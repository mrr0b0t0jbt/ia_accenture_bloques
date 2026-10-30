import pytest

from metricas_texto import (
    calcular_metricas,
    calcular_precision_cobertura_f1,
    contar_coincidencias,
    ordenar_peores_primero,
    resumir_metricas,
    seleccionar_muestra_revision,
)
from pipeline_texto import procesar_texto

GATO_REFERENCIA = ["el", "gato", "duerme", "en", "el", "sofa"]
GATO_RESPUESTA = ["el", "gato", "duerme", "en", "el", "sillon", "todo", "el", "dia"]
FUTBOL_REFERENCIA = ["el", "equipo", "gano", "el", "partido", "por", "3", "goles", "a", "1", "en", "su", "estadio", "local"]
FUTBOL_RESPUESTA = ["el", "equipo", "gano", "el", "partido", "en", "su", "estadio", "local"]
VACIOS = [([], ["a"]), (["a"], []), ([], [])]


def _resultado(f1: float) -> dict:
    return {"consulta": f"caso {f1}", "precision": f1, "cobertura": f1, "f1": f1}


# --- contar_coincidencias ---

def test_contar_coincidencias_las_repeticiones_de_mas_no_cuentan():
    assert contar_coincidencias(["el", "gato"], ["el", "gato", "el", "gato"]) == 2


def test_contar_coincidencias_ejemplo_del_gato():
    assert contar_coincidencias(GATO_REFERENCIA, GATO_RESPUESTA) == 5


def test_contar_coincidencias_sin_coincidencias_da_0():
    assert contar_coincidencias(["a", "b"], ["c", "d"]) == 0


@pytest.mark.parametrize("referencia, respuesta", VACIOS)
def test_contar_coincidencias_con_listas_vacias_da_0(referencia, respuesta):
    assert contar_coincidencias(referencia, respuesta) == 0


@pytest.mark.parametrize("referencia, respuesta", [("el gato", ["el", "gato"]), (["el", "gato"], "el gato")])
def test_contar_coincidencias_con_str_lanza_type_error(referencia, respuesta):
    with pytest.raises(TypeError):
        contar_coincidencias(referencia, respuesta)


# --- calcular_precision_cobertura_f1 ---

def test_precision_cobertura_f1_ejemplo_del_gato():
    metricas = calcular_precision_cobertura_f1(GATO_REFERENCIA, GATO_RESPUESTA)
    assert metricas["precision"] == pytest.approx(5 / 9)
    assert metricas["cobertura"] == pytest.approx(5 / 6)
    assert metricas["f1"] == pytest.approx(0.6667, abs=1e-4)


def test_precision_cobertura_f1_sinonimo():
    metricas = calcular_precision_cobertura_f1(["el", "auto", "es", "veloz"], ["el", "vehiculo", "es", "rapido"])
    assert metricas == pytest.approx({"precision": 0.5, "cobertura": 0.5, "f1": 0.5})


def test_precision_cobertura_f1_dato_equivocado():
    metricas = calcular_precision_cobertura_f1(
        ["python", "fue", "creado", "en", "1991"], ["python", "fue", "creado", "en", "1989"]
    )
    assert metricas == pytest.approx({"precision": 0.8, "cobertura": 0.8, "f1": 0.8})


def test_precision_cobertura_f1_negacion_ausente():
    metricas = calcular_precision_cobertura_f1(
        ["python", "no", "es", "un", "lenguaje", "compilado"], ["python", "es", "un", "lenguaje", "compilado"]
    )
    assert metricas["precision"] == pytest.approx(1.0)
    assert metricas["cobertura"] == pytest.approx(5 / 6)
    assert metricas["f1"] == pytest.approx(0.9091, abs=1e-4)


def test_precision_cobertura_f1_futbol():
    metricas = calcular_precision_cobertura_f1(FUTBOL_REFERENCIA, FUTBOL_RESPUESTA)
    assert metricas["precision"] == pytest.approx(1.0)
    assert metricas["cobertura"] == pytest.approx(9 / 14)
    assert metricas["f1"] == pytest.approx(0.783, abs=1e-3)


@pytest.mark.parametrize("referencia, respuesta", VACIOS)
def test_precision_cobertura_f1_con_listas_vacias_da_ceros(referencia, respuesta):
    assert calcular_precision_cobertura_f1(referencia, respuesta) == {"precision": 0.0, "cobertura": 0.0, "f1": 0.0}


@pytest.mark.parametrize("referencia, respuesta", [("", ["a"]), (["a"], "")])
def test_precision_cobertura_f1_con_str_vacio_lanza_type_error(referencia, respuesta):
    # el tipo se valida antes que las listas vacías
    with pytest.raises(TypeError):
        calcular_precision_cobertura_f1(referencia, respuesta)


def test_precision_cobertura_f1_sin_coincidencias_da_ceros():
    assert calcular_precision_cobertura_f1(["a"], ["b"]) == {"precision": 0.0, "cobertura": 0.0, "f1": 0.0}


# --- calcular_metricas ---

def test_calcular_metricas_normaliza_puntuacion_y_mayusculas():
    metricas = calcular_metricas("El auto es veloz.", "El vehículo es rápido.")
    assert metricas == pytest.approx({"precision": 0.5, "cobertura": 0.5, "f1": 0.5})


def test_calcular_metricas_con_none_lanza_type_error():
    with pytest.raises(TypeError):
        calcular_metricas(None, "texto")


def test_calcular_metricas_futbol_igual_que_sobre_tokens_procesados():
    referencia = "El equipo ganó el partido por 3 goles a 1 en su estadio local."
    respuesta = "El equipo ganó el partido en su estadio local."
    esperado = calcular_precision_cobertura_f1(procesar_texto(referencia), procesar_texto(respuesta))
    assert calcular_metricas(referencia, respuesta) == pytest.approx(esperado)


# --- resumir_metricas ---

def test_resumir_metricas_promedia_las_tres_metricas():
    resultados = [
        {"precision": 0.83, "cobertura": 0.71, "f1": 0.77},
        {"precision": 0.40, "cobertura": 0.50, "f1": 0.44},
        {"precision": 0.62, "cobertura": 0.62, "f1": 0.62},
    ]
    resumen = resumir_metricas(resultados)
    assert resumen["f1_promedio"] == pytest.approx(0.61, abs=0.005)
    assert resumen["precision_promedio"] == pytest.approx((0.83 + 0.40 + 0.62) / 3)
    assert resumen["cobertura_promedio"] == pytest.approx((0.71 + 0.50 + 0.62) / 3)


def test_resumir_metricas_lista_vacia_da_ceros():
    assert resumir_metricas([]) == {"precision_promedio": 0.0, "cobertura_promedio": 0.0, "f1_promedio": 0.0}


# --- muestra para revision manual ---

def test_ordenar_peores_primero_ordena_por_f1_sin_modificar_la_original():
    resultados = [_resultado(0.5), _resultado(0.1), _resultado(0.9)]
    copia = list(resultados)
    ordenados = ordenar_peores_primero(resultados)
    assert [r["f1"] for r in ordenados] == [0.1, 0.5, 0.9]
    assert resultados == copia


def test_seleccionar_muestra_revision_cinco_peores_y_cinco_mejores():
    f1s = [0.12, 0.93, 0.45, 0.20, 0.88, 0.33, 0.51, 0.67, 0.75, 0.28, 0.81, 0.60]
    muestra = seleccionar_muestra_revision([_resultado(f1) for f1 in f1s])
    assert [r["f1"] for r in muestra] == [0.12, 0.20, 0.28, 0.33, 0.45, 0.67, 0.75, 0.81, 0.88, 0.93]
    assert len({r["consulta"] for r in muestra}) == 10


def test_seleccionar_muestra_revision_con_pocos_casos_no_duplica():
    resultados = [_resultado(f1) for f1 in [0.6, 0.1, 0.4, 0.9, 0.2, 0.7]]
    muestra = seleccionar_muestra_revision(resultados)
    assert [r["f1"] for r in muestra] == [0.1, 0.2, 0.4, 0.6, 0.7, 0.9]


def test_seleccionar_muestra_revision_lista_vacia():
    assert seleccionar_muestra_revision([]) == []
