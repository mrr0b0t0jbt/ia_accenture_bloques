import json
from pathlib import Path

import pytest

import indexacion
from metricas_recuperacion import (
    K_MAXIMO,
    MAX_CASOS,
    evaluar_caso,
    evaluar_recuperacion,
    exhaustividad_en_k,
    f1,
    precision_en_k,
    promediar,
    validar_casos_evaluacion,
)

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_EVALUACION = RAIZ / "JSON" / "evaluacion_recuperacion.json"
RUTA_CASOS = RAIZ / "JSON" / "casos_recuperacion.json"
RUTA_CORPUS = RAIZ / "JSON" / "corpus_miniapp.json"
CASO_VALIDO = {"id": "c1", "pregunta": "¿Plazo?", "ids_relevantes": ["doc0_0"], "respondible": True}
CASO_SIN_RESPUESTA = {"id": "c2", "pregunta": "¿Presupuesto?", "ids_relevantes": [], "respondible": False}
RANKING_CLASE = ["doc2", "doc4", "doc0", "doc1", "doc3"]


def _casos_reales() -> list[dict]:
    return json.loads(RUTA_EVALUACION.read_text(encoding="utf-8"))


def _funcion_falsa(rankings: dict[str, list[str]], llamadas: list | None = None):
    """Devuelve recuperar_ids que responde con el ranking fijo de cada pregunta (distancias 0.1, 0.2...)."""

    def recuperar_ids(pregunta, k_max):
        if llamadas is not None:
            llamadas.append((pregunta, k_max))
        return [(id_chunk, 0.1 * (n + 1)) for n, id_chunk in enumerate(rankings[pregunta][:k_max])]

    return recuperar_ids


def _caso(id_caso, relevantes, respondible=True):
    return {"id": id_caso, "pregunta": f"pregunta {id_caso}", "ids_relevantes": relevantes, "respondible": respondible}


def test_01_ejemplo_de_la_clase():
    assert precision_en_k(["doc2", "doc0", "doc4"], {"doc0"}) == pytest.approx(1 / 3)
    assert exhaustividad_en_k(["doc2", "doc0", "doc4"], {"doc0"}) == 1.0


@pytest.mark.parametrize(
    "k, precision, exhaustividad, valor_f1",
    [(1, 0, 0, 0), (2, 0, 0, 0), (3, 1 / 3, 1, 0.5), (4, 1 / 4, 1, 0.4), (5, 1 / 5, 1, 1 / 3)],
)
def test_02_trade_off_de_la_clase(k, precision, exhaustividad, valor_f1):
    p = precision_en_k(RANKING_CLASE[:k], {"doc0"})
    r = exhaustividad_en_k(RANKING_CLASE[:k], {"doc0"})
    assert p == pytest.approx(precision) and r == pytest.approx(exhaustividad)
    assert f1(p, r) == pytest.approx(valor_f1)


def test_03_lista_vacia_de_recuperados_no_divide_por_cero():
    assert precision_en_k([], {"doc0"}) == 0.0
    assert exhaustividad_en_k([], {"doc0"}) == 0.0
    assert f1(0.0, 0.0) == 0.0


def test_04_sin_relevantes_la_exhaustividad_y_el_f1_no_estan_definidos():
    assert exhaustividad_en_k(["doc1"], frozenset()) is None
    assert f1(precision_en_k(["doc1"], frozenset()), None) is None
    assert precision_en_k(["doc1"], frozenset()) == 0.0


def test_05_ids_relevantes_debe_ser_set_o_frozenset():
    with pytest.raises(TypeError):
        precision_en_k(["doc0"], "doc0")
    with pytest.raises(TypeError):
        exhaustividad_en_k(["doc0"], ["doc0"])
    assert precision_en_k(["doc0"], frozenset({"doc0"})) == 1.0


def test_06_ids_recuperados_debe_ser_lista_de_textos_sin_repetidos():
    with pytest.raises(ValueError):
        precision_en_k(["doc0", "doc0"], {"doc0"})
    with pytest.raises(TypeError):
        precision_en_k([0], {"doc0"})
    for no_lista in (("doc0",), "doc0"):
        with pytest.raises(TypeError):
            exhaustividad_en_k(no_lista, {"doc0"})


def test_07_la_comparacion_es_exacta():
    assert precision_en_k(["Doc0"], {"doc0"}) == 0.0
    assert precision_en_k(["doc0 "], {"doc0"}) == 0.0


def test_08_promediar_ignora_none():
    assert promediar([1.0, None, 0.0]) == 0.5
    assert promediar([]) == 0.0
    assert promediar([None, None]) == 0.0


def test_09_evaluar_caso_devuelve_las_claves_de_k_y_respeta_ks():
    resultado = evaluar_caso(RANKING_CLASE, {"doc0"})
    assert list(resultado) == [1, 2, 3, 4, 5]
    assert all(set(metricas) == {"precision", "exhaustividad", "f1"} for metricas in resultado.values())
    assert resultado[3]["f1"] == pytest.approx(0.5)
    assert list(evaluar_caso(RANKING_CLASE, {"doc0"}, ks=[3])) == [3]


@pytest.mark.parametrize("ks", [[0], [K_MAXIMO + 1], [True], ["3"], []])
def test_10_evaluar_caso_rechaza_k_invalido(ks):
    with pytest.raises(ValueError):
        evaluar_caso(RANKING_CLASE, {"doc0"}, ks=ks)


def test_11_ejemplo_de_la_clase_con_cuatro_casos():
    top3 = {
        "doc0": ["doc0", "doc4", "doc2"],
        "doc1": ["doc2", "doc4", "doc3"],
        "doc2": ["doc2", "doc0", "doc4"],
        "doc3": ["doc4", "doc3", "doc1"],
    }
    todos = ["doc0", "doc1", "doc2", "doc3", "doc4"]
    rankings = {f"pregunta {d}": ids + [x for x in todos if x not in ids] for d, ids in top3.items()}
    casos = [_caso(d, [d]) for d in top3]
    resultado = evaluar_recuperacion(_funcion_falsa(rankings), casos, k=3)
    promedios = resultado["promedios_respondibles"][3]
    assert promedios["precision"] == pytest.approx(0.25)
    assert promedios["exhaustividad"] == pytest.approx(0.75)
    assert promedios["exhaustividad"] >= promedios["precision"]
    assert resultado["sin_respuesta"] == []


def test_12_el_caso_sin_respuesta_no_entra_en_los_promedios():
    rankings = {"pregunta a": ["a_0", "x_0", "y_0"], "pregunta b": ["x_0", "b_0", "y_0"], "pregunta s": ["x_0", "y_0", "a_0"]}
    casos = [_caso("a", ["a_0"]), _caso("b", ["b_0"]), _caso("s", [], respondible=False)]
    resultado = evaluar_recuperacion(_funcion_falsa(rankings), casos, k=2)
    promedios = resultado["promedios_respondibles"][2]
    assert promedios["precision"] == pytest.approx(0.5) and promedios["exhaustividad"] == 1.0
    assert resultado["sin_respuesta"] == [{"id": "s", "ids_recuperados": ["x_0", "y_0"], "distancia_minima": pytest.approx(0.1)}]
    por_caso = {c["id"]: c for c in resultado["por_caso"]}
    assert por_caso["s"]["metricas"] is None and por_caso["a"]["metricas"][2]["precision"] == 0.5
    assert por_caso["a"]["recuperados"][0] == {"rango": 1, "id": "a_0", "distancia": pytest.approx(0.1)}
    assert por_caso["s"]["ids_relevantes"] == [] and por_caso["s"]["respondible"] is False


def test_13_sin_respuesta_que_no_recupera_nada():
    resultado = evaluar_recuperacion(lambda pregunta, k_max: [], [_caso("s", [], respondible=False)], k=3)
    assert resultado["sin_respuesta"] == [{"id": "s", "ids_recuperados": [], "distancia_minima": None}]
    assert resultado["promedios_respondibles"][3] == {"precision": 0.0, "exhaustividad": 0.0, "f1": 0.0}


def test_14_recuperar_ids_se_llama_una_vez_por_caso_con_k_maximo():
    llamadas = []
    rankings = {"pregunta a": ["a_0", "x_0", "y_0", "z_0", "w_0"], "pregunta b": ["x_0", "y_0", "z_0", "w_0", "b_0"]}
    resultado = evaluar_recuperacion(_funcion_falsa(rankings, llamadas), [_caso("a", ["a_0"]), _caso("b", ["b_0"])], k=3)
    assert llamadas == [("pregunta a", K_MAXIMO), ("pregunta b", K_MAXIMO)]
    assert sorted(resultado["promedios_respondibles"]) == [1, 2, 3, 4, 5]
    assert resultado["promedios_respondibles"][5]["exhaustividad"] == 1.0
    assert resultado["promedios_respondibles"][1]["exhaustividad"] == 0.5


@pytest.mark.parametrize("k", [0, K_MAXIMO + 1, True])
def test_15_k_invalido_falla_sin_llamar_a_recuperar_ids(k):
    llamadas = []
    with pytest.raises(ValueError):
        evaluar_recuperacion(_funcion_falsa({}, llamadas), [CASO_VALIDO], k=k)
    assert llamadas == []


def test_16_el_json_real_es_valido():
    assert validar_casos_evaluacion(_casos_reales()) is None


@pytest.mark.parametrize(
    "casos",
    [
        [],
        [{**CASO_VALIDO, "id": f"c{n}"} for n in range(MAX_CASOS + 1)],
        [CASO_VALIDO, "no soy un objeto"],
        [{k: v for k, v in CASO_VALIDO.items() if k != "pregunta"}],
        [CASO_VALIDO, dict(CASO_VALIDO)],
        [{**CASO_VALIDO, "ids_relevantes": "doc0_0"}],
        [{**CASO_VALIDO, "ids_relevantes": ["doc0_0", "doc0_0"]}],
        [{**CASO_VALIDO, "ids_relevantes": []}],
        [{**CASO_SIN_RESPUESTA, "ids_relevantes": ["doc0_0"]}],
        [{**CASO_VALIDO, "respondible": "si"}],
    ],
    ids=["vacia", "mas_de_20", "no_objeto", "sin_pregunta", "id_repetido", "relevantes_no_lista",
         "relevante_repetido", "respondible_sin_ids", "no_respondible_con_ids", "respondible_no_bool"],
)
def test_17_estructuras_invalidas_devuelven_texto(casos):
    problema = validar_casos_evaluacion(casos)
    assert isinstance(problema, str) and problema


def test_18_ids_validos_detecta_el_id_desconocido():
    problema = validar_casos_evaluacion([CASO_VALIDO], ids_validos={"otro_0"})
    assert "doc0_0" in problema and "c1" in problema
    assert validar_casos_evaluacion([CASO_VALIDO, CASO_SIN_RESPUESTA], ids_validos={"doc0_0", "otro_0"}) is None


def test_19_el_json_real_tiene_14_casos_y_sus_ids_existen_en_el_corpus():
    casos = _casos_reales()
    assert len(casos) == 14
    assert sum(c["respondible"] for c in casos) == 11 and sum(not c["respondible"] for c in casos) == 3
    _, ids = indexacion.dividir_corpus(
        indexacion.cargar_corpus(str(RUTA_CORPUS)), indexacion.CHUNK_SIZE_POR_DEFECTO, indexacion.CHUNK_OVERLAP_POR_DEFECTO
    )
    assert validar_casos_evaluacion(casos, ids_validos=set(ids)) is None
    assert {i for c in casos for i in c["ids_relevantes"]} <= set(ids)


def test_20_q1_a_q8_coinciden_con_casos_recuperacion():
    previos = {c["id"]: c for c in json.loads(RUTA_CASOS.read_text(encoding="utf-8"))}
    reales = [c for c in _casos_reales() if c["id"] in previos]
    assert len(reales) == len(previos) == 8
    for caso in reales:
        previo = previos[caso["id"]]
        assert caso["pregunta"] == previo["pregunta"]
        documentos = {id_chunk.rsplit("_", 1)[0] for id_chunk in caso["ids_relevantes"]}
        assert documentos <= set(previo["fuentes_esperadas"])
        assert documentos == set(previo["fuentes_esperadas"])
    assert {d for c in reales if c["id"].startswith("q1_") for d in (i.rsplit("_", 1)[0] for i in c["ids_relevantes"])} == {
        "aviso_cve_2099_10002",
        "politica_parcheo",
    }
