import copy

import pytest

import decision_personalizacion as dp

CASO_1 = {
    "informacion_cambia_seguido": True,
    "problema_de_formato_o_tono": True,
    "dataset_grande_y_de_calidad": False,
    "presupuesto_limitado_o_iterar_rapido": True,
    "necesita_multiples_variantes": False,
}
CASO_2 = {
    "informacion_cambia_seguido": False,
    "problema_de_formato_o_tono": True,
    "dataset_grande_y_de_calidad": True,
    "presupuesto_limitado_o_iterar_rapido": False,
    "necesita_multiples_variantes": False,
}
TODO_FALSE = {k: False for k in dp.PREGUNTAS}
TODO_TRUE = {k: True for k in dp.PREGUNTAS}
PROBLEMA = "El clasificador de errores no responde con un formato consistente."


# R1
def test_r1_constantes():
    assert dp.TECNICAS == ("RAG", "Fine-tuning completo", "PEFT")
    assert dp.PREGUNTAS == (
        "informacion_cambia_seguido", "problema_de_formato_o_tono", "dataset_grande_y_de_calidad",
        "presupuesto_limitado_o_iterar_rapido", "necesita_multiples_variantes",
    )
    assert dp.PESOS == {
        "informacion_cambia_seguido": {"RAG": 3},
        "problema_de_formato_o_tono": {"PEFT": 2, "Fine-tuning completo": 1},
        "dataset_grande_y_de_calidad": {"Fine-tuning completo": 2, "PEFT": 1},
        "presupuesto_limitado_o_iterar_rapido": {"RAG": 2, "PEFT": 1},
        "necesita_multiples_variantes": {"PEFT": 2},
    }
    assert dp.JORNADA_HORAS == 8 and dp.MAX_CARACTERES_PROBLEMA == 1000


# R2
def test_r2a_caso_2_empata_fine_tuning_y_peft():
    r = dp.recomendar_estrategia(CASO_2)
    assert r == {
        "puntos": {"RAG": 0, "Fine-tuning completo": 3, "PEFT": 3}, "maximo": 3,
        "ganadoras": ["Fine-tuning completo", "PEFT"], "empate": True, "sin_necesidad": False,
    }
    assert list(r["puntos"]) == list(dp.TECNICAS)


def test_r2b_caso_1_gana_rag():
    r = dp.recomendar_estrategia(CASO_1)
    assert r["puntos"] == {"RAG": 5, "Fine-tuning completo": 1, "PEFT": 3}
    assert r["ganadoras"] == ["RAG"] and r["empate"] is False and r["sin_necesidad"] is False


def test_r2c_todo_false_no_justifica_ninguna_tecnica():
    r = dp.recomendar_estrategia(TODO_FALSE)
    assert r["puntos"] == {"RAG": 0, "Fine-tuning completo": 0, "PEFT": 0} and r["maximo"] == 0
    assert r["ganadoras"] == [] and r["empate"] is False and r["sin_necesidad"] is True


def test_r2d_todo_true_gana_peft():
    r = dp.recomendar_estrategia(TODO_TRUE)
    assert r["puntos"] == {"RAG": 5, "Fine-tuning completo": 3, "PEFT": 6}
    assert r["ganadoras"] == ["PEFT"]


# V1
@pytest.mark.parametrize("malo", [[], None, "texto"], ids=["lista", "none", "texto"])
def test_v1_no_objeto(malo):
    assert dp.validar_respuestas(malo) == ["respuestas no es un objeto"]


def test_v1_valida_completa():
    assert dp.validar_respuestas(CASO_1) == []


def test_v1_falta_una_clave():
    r = {k: v for k, v in CASO_1.items() if k != "necesita_multiples_variantes"}
    assert dp.validar_respuestas(r) == ['falta la respuesta "necesita_multiples_variantes"']


def test_v1_clave_de_mas():
    assert dp.validar_respuestas({**CASO_1, "nota": True}) == ["respuestas tiene claves de mas: 'nota'"]


@pytest.mark.parametrize("valor", ["no", 1, None, []], ids=["texto", "entero", "none", "lista"])
def test_v1_valor_no_booleano(valor):
    r = {**CASO_1, "problema_de_formato_o_tono": valor}
    assert dp.validar_respuestas(r) == ['la respuesta "problema_de_formato_o_tono" no es True ni False']


# R4
@pytest.mark.parametrize(
    "malo,mensaje",
    [
        ({k: v for k, v in CASO_1.items() if k != "informacion_cambia_seguido"}, 'falta la respuesta "informacion_cambia_seguido"'),
        ({**CASO_1, "necesita_multiples_variantes": "no"}, 'la respuesta "necesita_multiples_variantes" no es True ni False'),
        ([], "respuestas no es un objeto"),
    ],
    ids=["clave_faltante", "valor_no_bool", "lista"],
)
def test_r4_recomendar_levanta_con_el_primer_problema(malo, mensaje):
    with pytest.raises(ValueError) as error:
        dp.recomendar_estrategia(malo)
    assert str(error.value) == mensaje


# S1
def test_s1a_caso_2_filas_exactas():
    todas = list(dp.TECNICAS)
    assert dp.analizar_sensibilidad(CASO_2) == [
        {"pregunta": "informacion_cambia_seguido", "nuevo_valor": True, "ganadoras": todas, "cambia": True},
        {"pregunta": "problema_de_formato_o_tono", "nuevo_valor": False, "ganadoras": ["Fine-tuning completo"], "cambia": True},
        {"pregunta": "dataset_grande_y_de_calidad", "nuevo_valor": False, "ganadoras": ["PEFT"], "cambia": True},
        {"pregunta": "presupuesto_limitado_o_iterar_rapido", "nuevo_valor": True, "ganadoras": ["PEFT"], "cambia": True},
        {"pregunta": "necesita_multiples_variantes", "nuevo_valor": True, "ganadoras": ["PEFT"], "cambia": True},
    ]


def test_s1b_todo_false_cada_fila_cambia():
    filas = dp.analizar_sensibilidad(TODO_FALSE)
    assert all(f["cambia"] is True and f["nuevo_valor"] is True for f in filas)
    assert filas[0]["ganadoras"] == ["RAG"]


def test_s1c_no_muta_la_entrada():
    entrada = copy.deepcopy(CASO_2)
    dp.analizar_sensibilidad(entrada)
    dp.recomendar_estrategia(entrada)
    assert entrada == CASO_2


def test_s1d_respuestas_invalidas():
    with pytest.raises(ValueError):
        dp.analizar_sensibilidad({**CASO_1, "necesita_multiples_variantes": 1})


# C1
@pytest.mark.parametrize(
    "n,minutos,tarifa,horas,costo",
    [(8, 5, 25, 40 / 60, 40 / 60 * 25), (200, 5, 25, 1000 / 60, 1000 / 60 * 25), (0, 5, 25, 0.0, 0.0)],
    ids=["8", "200", "cero"],
)
def test_c1a_estimar_costo_de_etiquetar(n, minutos, tarifa, horas, costo):
    h, c = dp.estimar_costo_de_etiquetar(n, minutos, tarifa)
    assert h == pytest.approx(horas) and c == pytest.approx(costo)
    assert isinstance(h, float) and isinstance(c, float)


@pytest.mark.parametrize(
    "args,mensaje",
    [
        ((-1, 5, 25), "num_ejemplos debe ser un entero mayor o igual que 0"),
        ((2.5, 5, 25), "num_ejemplos debe ser un entero mayor o igual que 0"),
        ((True, 5, 25), "num_ejemplos debe ser un entero mayor o igual que 0"),
        ((8, -1, 25), "minutos_por_ejemplo debe ser un numero finito mayor o igual que 0"),
        ((8, float("nan"), 25), "minutos_por_ejemplo debe ser un numero finito mayor o igual que 0"),
        ((8, 5, float("inf")), "costo_por_hora debe ser un numero finito mayor o igual que 0"),
        ((8, 5, -1), "costo_por_hora debe ser un numero finito mayor o igual que 0"),
        ((8, 5, "25"), "costo_por_hora debe ser un numero finito mayor o igual que 0"),
    ],
)
def test_c1b_invalidos(args, mensaje):
    with pytest.raises(ValueError) as error:
        dp.estimar_costo_de_etiquetar(*args)
    assert str(error.value) == mensaje


# C2
def test_c2a_con_revision():
    r = dp.estimar_costo_con_revision(500, 4, 20, 0.20, 2)
    esperado = {
        "horas_base": 33.3333333, "costo_base": 666.666667, "ejemplos_revisados": 100.0, "horas_revision": 3.3333333,
        "costo_revision": 66.666667, "horas_total": 36.666667, "costo_total": 733.333333, "jornadas": 4.5833333,
    }
    assert list(r) == list(esperado)
    for clave, valor in esperado.items():
        assert r[clave] == pytest.approx(valor)


def test_c2b_fraccion_cero():
    r = dp.estimar_costo_con_revision(500, 4, 20, 0, 2)
    assert r["horas_revision"] == 0 and r["costo_revision"] == 0
    assert r["horas_total"] == r["horas_base"] and r["costo_total"] == r["costo_base"]


def test_c2c_fraccion_uno():
    assert dp.estimar_costo_con_revision(500, 4, 20, 1, 2)["ejemplos_revisados"] == 500


@pytest.mark.parametrize("fraccion", [-0.1, 1.1, float("nan"), "x"], ids=["negativa", "mayor_a_1", "nan", "texto"])
def test_c2d_fraccion_invalida(fraccion):
    with pytest.raises(ValueError) as error:
        dp.estimar_costo_con_revision(500, 4, 20, fraccion, 2)
    assert str(error.value) == "fraccion_revision debe ser un numero entre 0 y 1"


def test_c2e_minutos_revision_invalidos():
    with pytest.raises(ValueError) as error:
        dp.estimar_costo_con_revision(500, 4, 20, 0.2, -1)
    assert str(error.value) == "minutos_revision debe ser un numero finito mayor o igual que 0"


# E1
ESQUELETO_EMPATE = "\n".join([
    "# Decision de personalizacion",
    "",
    "## 1. El problema",
    PROBLEMA,
    "",
    "## 2. Resultado del metodo de decision",
    "- Fine-tuning completo: 3 puntos",
    "- PEFT: 3 puntos",
    "- RAG: 0 puntos",
    "",
    "**Empate entre: Fine-tuning completo, PEFT**",
    "",
    "## 3. Tecnica elegida y por que",
    "[COMPLETAR: el puntaje no resolvio el empate solo -- justifica aqui con criterio propio]",
    "",
    "## 4. Que necesitarias para ejecutarla",
    "[COMPLETAR: corpus faltante, o cantidad de ejemplos etiquetados que faltan]",
])


def _esqueleto(problema, respuestas=CASO_2):
    return dp.generar_esqueleto_decision(problema, dp.recomendar_estrategia(respuestas))


def test_e1a_empate_texto_exacto():
    texto = _esqueleto(PROBLEMA)
    assert texto == ESQUELETO_EMPATE and not texto.endswith("\n")


def test_e1b_ganadora_unica():
    texto = _esqueleto(PROBLEMA, CASO_1)
    assert "[COMPLETAR: por que RAG responde mejor a este caso]" in texto
    assert "Empate" not in texto and "Ninguna tecnica" not in texto
    assert "- RAG: 5 puntos\n- PEFT: 3 puntos\n- Fine-tuning completo: 1 puntos\n" in texto


def test_e1c_sin_necesidad():
    texto = _esqueleto(PROBLEMA, TODO_FALSE)
    assert "\n\n**Ninguna tecnica obtuvo puntos**\n\n## 3. Tecnica elegida y por que\n" in texto
    assert (
        "[COMPLETAR: ninguna respuesta justifica RAG, fine-tuning ni PEFT; explica por que el prompt actual basta o que evidencia falta]"
        in texto
    )
    assert "Empate" not in texto


@pytest.mark.parametrize("problema", ["", "   "], ids=["vacio", "espacios"])
def test_e1d_problema_vacio(problema):
    with pytest.raises(ValueError, match="^problema vacio$"):
        _esqueleto(problema)


@pytest.mark.parametrize("problema", [None, 5], ids=["none", "entero"])
def test_e1e_problema_no_es_texto(problema):
    with pytest.raises(ValueError, match="^problema no es texto$"):
        _esqueleto(problema)


def test_e1f_caracteres_de_control():
    with pytest.raises(ValueError, match="^problema tiene caracteres de control$"):
        _esqueleto("hola " + chr(27) + "[31mrojo")


def test_e1g_limite_de_1000_caracteres_valido():
    assert "a" * 1000 in _esqueleto("a" * 1000)


def test_e1g_1001_caracteres_levanta():
    with pytest.raises(ValueError, match="^problema supera los 1000 caracteres$"):
        _esqueleto("a" * 1001)


def test_e1h_normaliza_a_una_linea_y_rechaza_encabezados():
    assert "## 1. El problema\nhola mundo\n\n## 2." in _esqueleto("  hola\n\n  mundo   ")
    for falso in ("# titulo", "\n## 3. algo"):
        with pytest.raises(ValueError, match="^problema no puede empezar con #$"):
            _esqueleto(falso)


def test_e1i_resultado_invalido():
    for resultado in ("texto", {"ganadoras": [], "empate": False, "sin_necesidad": True}):
        with pytest.raises(ValueError, match="^resultado no es el que devuelve recomendar_estrategia$"):
            dp.generar_esqueleto_decision(PROBLEMA, resultado)


# J2
def test_j2_h1_claves_de_mas_salen_ordenadas_con_repr():
    respuestas = {**CASO_1, "zeta": True, "alfa": True}  # zeta se inserta antes que alfa
    esperado = "respuestas tiene claves de mas: " + ", ".join(repr(k) for k in ("alfa", "zeta"))
    assert dp.validar_respuestas(respuestas) == [esperado]


def test_j2_h2_caso_1_cambia_y_ganadoras_por_fila():
    filas = dp.analizar_sensibilidad(CASO_1)
    assert [f["pregunta"] for f in filas] == list(dp.PREGUNTAS)
    assert [f["cambia"] for f in filas] == [True, False, False, False, True]
    assert [f["ganadoras"] for f in filas] == [["PEFT"], ["RAG"], ["RAG"], ["RAG"], ["RAG", "PEFT"]]
