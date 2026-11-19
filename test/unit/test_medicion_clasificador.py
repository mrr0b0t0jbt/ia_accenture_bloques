import pytest

import medicion_clasificador as mc
from api_claude_refactor import INPUT_PRICING, OUTPUT_PRICING
from enrutamiento import CATEGORIAS

TOKENS = {"input_tokens": 100, "output_tokens": 5, "total_tokens": 105}


# M1
@pytest.mark.parametrize(
    "cruda,esperada,exacta,normalizada,valida,correcta",
    [
        ("incidente", "incidente", True, "incidente", True, True),
        ("Incidente.", "incidente", False, "incidente", True, True),
        (" vulnerabilidad\n", "vulnerabilidad", False, "vulnerabilidad", True, True),
        ("consulta", "incidente", True, "consulta", True, False),
        ("La categoria es incidente", "incidente", False, "la categoria es incidente", False, False),
        ("", "consulta", False, "", False, False),
        ("incidente o vulnerabilidad", "incidente", False, "incidente o vulnerabilidad", False, False),
    ],
    ids=["exacta", "mayuscula_y_punto", "espacios_y_salto", "valida_incorrecta", "verbosa", "vacia", "dos_categorias"],
)
def test_m1_clasificar_respuesta(cruda, esperada, exacta, normalizada, valida, correcta):
    r = mc.clasificar_respuesta(cruda, esperada)
    assert (r["exacta"], r["normalizada"], r["valida"], r["correcta"]) == (exacta, normalizada, valida, correcta)
    assert set(r) == {"respuesta", "exacta", "normalizada", "valida", "correcta"}


def test_m1_cruda_no_texto():
    with pytest.raises(ValueError) as error:
        mc.clasificar_respuesta(None, "incidente")
    assert str(error.value) == "la respuesta cruda no es texto"


# M2
def test_m2_saneado_sin_controles():
    r = mc.clasificar_respuesta("a\x1b[31m\nb\r\nc", "consulta")
    assert r["respuesta"] == "a [31m b  c" and all(c.isprintable() for c in r["respuesta"])


def test_m2_recorte_a_200():
    assert mc.clasificar_respuesta("x" * 200, "consulta")["respuesta"] == "x" * 200
    assert mc.clasificar_respuesta("x" * 201, "consulta")["respuesta"] == "x" * 200


# M3
def _res(esperada, normalizada, exacta=True, error=None, tokens=TOKENS):
    return {
        "categoria_esperada": esperada, "normalizada": normalizada, "exacta": exacta, "valida": normalizada in CATEGORIAS,
        "correcta": normalizada == esperada, "error": error, "tokens": tokens,
    }


def _con_fallos(n_evaluadas: int, fallos: int) -> list[dict]:
    """n_evaluadas resultados correctos; los primeros `fallos` no son exactos (pero sí válidos)."""
    return [_res("incidente", "incidente", exacta=i >= fallos) for i in range(n_evaluadas)]


def test_m3a_diez_resultados_a_mano():
    resultados = (
        [_res(c, c) for c in CATEGORIAS for _ in range(2)]
        + [_res("incidente", "incidente", exacta=False), _res("consulta", "consulta", exacta=False)]
        + [_res("vulnerabilidad", "consulta")]
        + [_res("incidente", "la categoria es incidente", exacta=False)]
    )
    r = mc.resumir(resultados)
    assert (r["llamadas"], r["con_error"], r["evaluadas"], r["exactas"], r["validas"]) == (10, 0, 10, 7, 9)
    assert (r["fallos_formato_estricto"], r["fallos_formato_efectivo"], r["aciertos"], r["exactitud"]) == (3, 1, 8, 0.8)
    assert r["por_categoria"] == {
        "incidente": {"esperadas": 4, "aciertos": 3, "predichas": {"incidente": 3, "vulnerabilidad": 0, "consulta": 0, "otra": 1}},
        "vulnerabilidad": {"esperadas": 3, "aciertos": 2, "predichas": {"incidente": 0, "vulnerabilidad": 2, "consulta": 1, "otra": 0}},
        "consulta": {"esperadas": 3, "aciertos": 3, "predichas": {"incidente": 0, "vulnerabilidad": 0, "consulta": 3, "otra": 0}},
    }
    assert list(r["por_categoria"]) == list(CATEGORIAS) and r["umbral"] == 0.15


@pytest.mark.parametrize("fallos,proporcion,supera", [(3, 0.15, True), (2, 0.1, False)], ids=["justo_en_el_umbral", "por_debajo"])
def test_m3b_umbral_con_20_evaluadas(fallos, proporcion, supera):
    r = mc.resumir(_con_fallos(20, fallos))
    assert r["proporcion_estricta"] == proporcion and r["supera_umbral_estricto"] is supera
    assert r["fallos_formato_efectivo"] == 0 and r["supera_umbral_efectivo"] is False


def test_m3c_error_o_sin_tokens_no_se_evaluan():
    resultados = [_res("incidente", "incidente"), _res("incidente", "", error="RuntimeError: x", tokens=None), _res("incidente", "incidente", tokens=None)]
    r = mc.resumir(resultados)
    assert (r["llamadas"], r["con_error"], r["evaluadas"], r["aciertos"]) == (3, 1, 1, 1)


def test_m3d_sin_evaluadas():
    for resultados in ([], [_res("incidente", "", error="x", tokens=None)]):
        r = mc.resumir(resultados)
        assert r["evaluadas"] == 0 and r["exactitud"] is None
        assert r["proporcion_estricta"] is None and r["proporcion_efectiva"] is None
        assert r["supera_umbral_estricto"] is None and r["supera_umbral_efectivo"] is None


def test_m3e_invalida_cuenta_en_otra_y_en_las_esperadas():
    r = mc.resumir([_res("consulta", "no se", exacta=False)])
    assert r["por_categoria"]["consulta"] == {"esperadas": 1, "aciertos": 0, "predichas": {"incidente": 0, "vulnerabilidad": 0, "consulta": 0, "otra": 1}}


@pytest.mark.parametrize("fallos,proporcion,supera", [(4, 0.1667, True), (3, 0.125, False)], ids=["4_de_24", "3_de_24"])
def test_m3f_umbral_con_24_evaluadas(fallos, proporcion, supera):
    r = mc.resumir(_con_fallos(24, fallos))
    assert r["proporcion_estricta"] == proporcion and r["supera_umbral_estricto"] is supera


# M4
def test_m4_costo_con_tokens_a_mano():
    assert mc.costo_de_llamada(445, 138) == pytest.approx(445 * INPUT_PRICING + 138 * OUTPUT_PRICING)


def test_m4_costo_cero():
    assert mc.costo_de_llamada(0, 0) == 0.0
