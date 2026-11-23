import importlib.util
import json
import os
from pathlib import Path

import pytest

import matriz_riesgos

RAIZ = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location("script_matriz_riesgos", RAIZ / "scripts" / "matriz_riesgos.py")
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

CODIGO = "CONSTANTE = 1\nANOTADA: int = 2\n\n\ndef funcion():\n    def interna():\n        return 1\n    return interna\n\n\nclass Clase:\n    def metodo(self):\n        return 1\n"


def fila_ok(**cambios):
    fila = {
        "riesgo": "Riesgo de prueba",
        "probabilidad": 3,
        "impacto": 4,
        "mitigacion": "Una mitigacion concreta.",
        "evidencias": ["src/codigo.py::funcion"],
        "supervision": None,
        "nota": "",
    }
    fila.update(cambios)
    return fila


@pytest.fixture
def raiz(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "codigo.py").write_text(CODIGO, encoding="utf-8")
    return tmp_path


def escribir(ruta, filas):
    Path(ruta).write_text(json.dumps(filas), encoding="utf-8")
    return str(ruta)


@pytest.mark.parametrize("p,i,esperado", [(1, 1, 1), (3, 4, 12), (5, 5, 25), (2, 5, 10)])
def test_l3_a_el_puntaje_es_probabilidad_por_impacto(p, i, esperado):
    assert matriz_riesgos.puntaje_riesgo(p, i) == esperado


@pytest.mark.parametrize("p,i", [(0, 3), (3, 0), (6, 3), (3, 6), (True, 3), (3, True), (2.0, 3), (3, "3"), (None, 3), (-1, 3)])
def test_l3_b_el_puntaje_rechaza_valores_fuera_de_escala_o_que_no_son_enteros(p, i):
    with pytest.raises(ValueError):
        matriz_riesgos.puntaje_riesgo(p, i)


@pytest.mark.parametrize(
    "puntaje,esperada",
    [(1, "BAJO"), (3, "BAJO"), (4, "MEDIO"), (8, "MEDIO"), (9, "ALTO"), (14, "ALTO"), (15, "CRITICO"), (25, "CRITICO")],
)
def test_l3_c_la_categoria_cambia_exactamente_en_los_cortes_4_9_y_15(puntaje, esperada):
    assert matriz_riesgos.categoria_riesgo(puntaje) == esperada


@pytest.mark.parametrize("puntaje", [0, 26, -3, True, 9.0, "9", None])
def test_l3_d_la_categoria_rechaza_puntajes_invalidos(puntaje):
    with pytest.raises(ValueError):
        matriz_riesgos.categoria_riesgo(puntaje)


@pytest.mark.parametrize(
    "p,i,esperado",
    [(1, 1, False), (5, 4, False), (5, 5, False), (4, 4, True), (3, 3, True), (1, 5, True), (1, 2, True), (5, 3, True), (1, 4, True), (4, 1, True)],
)
def test_l3_e_al_limite_solo_si_un_punto_cambia_la_categoria_sin_salir_de_1_a_5(p, i, esperado):
    assert matriz_riesgos.esta_en_el_limite(p, i) is esperado


def test_l3_e2_al_limite_coincide_con_un_calculo_independiente_en_las_25_combinaciones():
    def categoria(puntaje):
        return "CRITICO" if puntaje >= 15 else "ALTO" if puntaje >= 9 else "MEDIO" if puntaje >= 4 else "BAJO"

    for p in range(1, 6):
        for i in range(1, 6):
            vecinos = [(p + dp, i + di) for dp, di in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 1 <= p + dp <= 5 and 1 <= i + di <= 5]
            esperado = any(categoria(a * b) != categoria(p * i) for a, b in vecinos)
            assert matriz_riesgos.esta_en_el_limite(p, i) is esperado, (p, i)


def test_l3_f_una_fila_correcta_no_tiene_problemas():
    assert matriz_riesgos.problemas_de_fila(fila_ok(), 1) == []
    assert matriz_riesgos.problemas_de_fila(fila_ok(supervision="esperar_aprobacion", nota="algo"), 1) == []


@pytest.mark.parametrize(
    "cambios",
    [
        {"riesgo": "   "},
        {"riesgo": 5},
        {"riesgo": "x" * 501},
        {"probabilidad": True},
        {"probabilidad": 0},
        {"probabilidad": 3.0},
        {"impacto": 6},
        {"impacto": "4"},
        {"mitigacion": ""},
        {"mitigacion": None},
        {"evidencias": "src/codigo.py::funcion"},
        {"evidencias": [5]},
        {"evidencias": ["a.py::b"] * 11},
        {"supervision": ""},
        {"supervision": 5},
        {"nota": 5},
        {"nota": "x" * 501},
    ],
    ids=lambda c: next(iter(c)) + "=" + repr(next(iter(c.values())))[:12],
)
def test_l3_g_una_fila_con_un_campo_invalido_da_un_problema_que_cita_el_numero_de_fila(cambios):
    problemas = matriz_riesgos.problemas_de_fila(fila_ok(**cambios), 7)
    assert len(problemas) == 1
    assert problemas[0].startswith("fila 7: ")


def test_l3_h_las_claves_deben_ser_exactas_y_estar_en_orden():
    extra = {**fila_ok(), "otra": 1}
    faltante = {k: v for k, v in fila_ok().items() if k != "nota"}
    desordenada = dict(reversed(list(fila_ok().items())))
    for fila in (extra, faltante, desordenada):
        assert matriz_riesgos.problemas_de_fila(fila, 2) == ["fila 2: las claves no son las esperadas o no estan en orden"]
    assert matriz_riesgos.problemas_de_fila(["no es objeto"], 2) == ["fila 2: no es un objeto"]


def test_l3_i_los_mensajes_nunca_citan_el_contenido_de_la_fila():
    fila = fila_ok(riesgo="SECRETO-ABC", mitigacion="SECRETO-DEF", impacto=9, evidencias=["SECRETO-GHI"], nota=["SECRETO-JKL"])
    texto = " ".join(matriz_riesgos.problemas_de_fila(fila, 1))
    assert "SECRETO" not in texto
    assert texto


@pytest.mark.parametrize("matriz", [[], {}, "texto", None, [fila_ok()] * 0, [fila_ok(riesgo=str(n)) for n in range(101)]])
def test_l3_j_la_matriz_debe_ser_una_lista_de_1_a_100_filas(matriz):
    assert matriz_riesgos.problemas_de_matriz(matriz) == ["la matriz debe ser una lista de 1 a 100 filas"]


def test_l3_k_un_riesgo_repetido_se_detecta_sin_importar_mayusculas_ni_espacios():
    filas = [fila_ok(riesgo="Inyeccion de prompt"), fila_ok(riesgo="otro"), fila_ok(riesgo="  INYECCION   de   prompt ")]
    assert matriz_riesgos.problemas_de_matriz(filas) == ["fila 3: riesgo repetido (igual que la fila 1)"]
    assert matriz_riesgos.problemas_de_matriz(filas[:2]) == []


def test_l3_l_leer_matriz_devuelve_las_filas_validas(tmp_path):
    filas = [fila_ok(), fila_ok(riesgo="otro riesgo")]
    assert matriz_riesgos.leer_matriz(escribir(tmp_path / "m.json", filas)) == filas


def test_l3_m_leer_matriz_no_cita_la_ruta_ni_el_contenido_en_sus_errores(tmp_path):
    inexistente = tmp_path / "secreto_ruta.json"
    with pytest.raises(ValueError) as info:
        matriz_riesgos.leer_matriz(str(inexistente))
    assert str(info.value) == "no se pudo leer la matriz (FileNotFoundError)"
    with pytest.raises(ValueError) as info:
        matriz_riesgos.leer_matriz(str(tmp_path))
    assert "IsADirectoryError" in str(info.value) and str(tmp_path) not in str(info.value)
    malo = tmp_path / "malo.json"
    malo.write_text('[\n{"riesgo": SECRETO}]', encoding="utf-8")
    with pytest.raises(ValueError) as info:
        matriz_riesgos.leer_matriz(str(malo))
    assert str(info.value) == "la matriz no es JSON valido (linea 2)"
    binario = tmp_path / "bin.json"
    binario.write_bytes(b"\xff\xfe[]")
    with pytest.raises(ValueError) as info:
        matriz_riesgos.leer_matriz(str(binario))
    assert str(info.value) == "la matriz no es UTF-8 valido"


def test_l3_n_leer_matriz_rechaza_filas_invalidas_con_todos_los_problemas(tmp_path):
    ruta = escribir(tmp_path / "m.json", [fila_ok(impacto=9), fila_ok(riesgo="otro", mitigacion="")])
    with pytest.raises(ValueError) as info:
        matriz_riesgos.leer_matriz(ruta)
    assert str(info.value) == "fila 1: impacto fuera de 1 a 5 o no es entero; fila 2: mitigacion vacia, muy larga o no es texto"


@pytest.mark.parametrize("simbolo", ["funcion", "Clase", "CONSTANTE", "ANOTADA"])
def test_l3_o_una_evidencia_existente_no_da_problema(raiz, simbolo):
    assert matriz_riesgos.problema_de_evidencia(str(raiz), f"src/codigo.py::{simbolo}") is None


@pytest.mark.parametrize(
    "evidencia,motivo",
    [
        ("src/codigo.py::interna", "el simbolo no existe en el archivo"),
        ("src/codigo.py::metodo", "el simbolo no existe en el archivo"),
        ("src/codigo.py::NoExiste", "el simbolo no existe en el archivo"),
        ("src/otro.py::funcion", "el archivo no existe"),
        ("src/codigo.py", "formato invalido (se espera ruta.py::simbolo)"),
        ("src/codigo.py::", "formato invalido (se espera ruta.py::simbolo)"),
        ("src/codigo.py::a b", "formato invalido (se espera ruta.py::simbolo)"),
        ("src/codigo.txt::funcion", "formato invalido (se espera ruta.py::simbolo)"),
        ("::funcion", "formato invalido (se espera ruta.py::simbolo)"),
        ("../fuera.py::funcion", "la ruta debe ser relativa y no salir de la raiz"),
        ("src/../../fuera.py::funcion", "la ruta debe ser relativa y no salir de la raiz"),
        ("/etc/passwd.py::funcion", "la ruta debe ser relativa y no salir de la raiz"),
    ],
)
def test_l3_p_una_evidencia_invalida_da_el_motivo_sin_citar_la_ruta(raiz, evidencia, motivo):
    assert matriz_riesgos.problema_de_evidencia(str(raiz), evidencia) == motivo


def test_l3_q_un_enlace_simbolico_que_sale_de_la_raiz_se_rechaza(raiz, tmp_path_factory):
    fuera = tmp_path_factory.mktemp("fuera")
    (fuera / "ajeno.py").write_text("def funcion():\n    return 1\n", encoding="utf-8")
    os.symlink(fuera / "ajeno.py", raiz / "src" / "enlace.py")
    assert matriz_riesgos.problema_de_evidencia(str(raiz), "src/enlace.py::funcion") == "la ruta sale de la raiz"


def test_l3_r_un_archivo_con_error_de_sintaxis_no_se_puede_analizar(raiz):
    (raiz / "src" / "roto.py").write_text("def (:\n", encoding="utf-8")
    assert matriz_riesgos.problema_de_evidencia(str(raiz), "src/roto.py::x") == "el archivo no se pudo analizar"


def test_l3_s_la_evidencia_se_lee_con_ast_y_nunca_ejecuta_el_archivo(raiz, tmp_path):
    marca = tmp_path / "ejecutado.txt"
    (raiz / "src" / "efecto.py").write_text(f"open({str(marca)!r}, 'w').write('x')\n\n\ndef funcion():\n    return 1\n", encoding="utf-8")
    assert matriz_riesgos.problema_de_evidencia(str(raiz), "src/efecto.py::funcion") is None
    assert not marca.exists()


def test_l3_t_las_evidencias_de_la_matriz_citan_fila_y_posicion_no_el_contenido(raiz):
    filas = [fila_ok(evidencias=["src/codigo.py::funcion", "src/codigo.py::SECRETO"]), fila_ok(riesgo="otro", evidencias=[])]
    assert matriz_riesgos.problemas_de_evidencias(filas, str(raiz)) == [
        "fila 1, evidencia 2: el simbolo no existe en el archivo",
        "fila 2: sin evidencias",
    ]
    assert matriz_riesgos.problemas_de_evidencias([fila_ok()], str(raiz)) == []


def test_l3_u_las_filas_calculadas_ordenan_por_puntaje_con_empates_estables_y_no_mutan_la_entrada():
    filas = [fila_ok(riesgo="a", probabilidad=2, impacto=3), fila_ok(riesgo="b", probabilidad=4, impacto=4), fila_ok(riesgo="c", probabilidad=3, impacto=2)]
    copia = json.loads(json.dumps(filas))
    calculadas = matriz_riesgos.filas_calculadas(filas)
    assert [f["riesgo"] for f in calculadas] == ["b", "a", "c"]
    assert [(f["puntaje"], f["categoria"], f["al_limite"]) for f in calculadas] == [(16, "CRITICO", True), (6, "MEDIO", True), (6, "MEDIO", True)]
    assert filas == copia


def test_l3_v_la_tabla_tiene_el_formato_de_la_clase_con_la_marca_al_limite():
    filas = [fila_ok(riesgo="Inyeccion", probabilidad=4, impacto=4), fila_ok(riesgo="Contraste", probabilidad=1, impacto=1)]
    assert matriz_riesgos.tabla(filas) == ["CRITICO  (16) - Inyeccion [AL LIMITE]", "BAJO     ( 1) - Contraste"]


def test_l3_w_la_matriz_del_repo_es_valida_y_todas_sus_evidencias_existen_en_el_codigo():
    filas = matriz_riesgos.leer_matriz(str(RAIZ / matriz_riesgos.RUTA_MATRIZ_POR_DEFECTO))
    assert len(filas) >= 5
    assert matriz_riesgos.problemas_de_evidencias(filas, str(RAIZ)) == []
    assert {f["categoria"] for f in matriz_riesgos.filas_calculadas(filas)} <= {"CRITICO", "ALTO", "MEDIO", "BAJO"}


def test_l3_x_el_script_usa_por_defecto_la_matriz_y_la_raiz_del_proyecto():
    argumentos = script._argumentos([])
    assert argumentos.matriz == str(RAIZ / "docs" / "matriz_riesgos.json")
    assert argumentos.raiz == str(RAIZ)


def test_l3_y_el_script_con_la_matriz_del_repo_sale_con_0_e_imprime_la_tabla(capsys):
    assert script.main([]) == 0
    salida = capsys.readouterr()
    assert salida.err == ""
    assert "todas las evidencias existen en el codigo." in salida.out
    assert salida.out.splitlines()[0].startswith(("CRITICO", "ALTO", "MEDIO", "BAJO"))


def test_l3_z_el_script_sale_con_1_y_sin_traceback_si_la_matriz_no_se_puede_leer(tmp_path, capsys):
    assert script.main(["--matriz", str(tmp_path / "no_existe.json")]) == 1
    salida = capsys.readouterr()
    assert salida.out == ""
    assert salida.err == "ERROR: no se pudo leer la matriz (FileNotFoundError)\n"


def test_l3_za_el_script_sale_con_1_si_una_evidencia_no_existe_pero_imprime_la_tabla(raiz, tmp_path, capsys):
    ruta = escribir(tmp_path / "m.json", [fila_ok(evidencias=["src/codigo.py::NoExiste"])])
    assert script.main(["--matriz", ruta, "--raiz", str(raiz)]) == 1
    salida = capsys.readouterr()
    assert salida.out.startswith("ALTO     (12) - Riesgo de prueba")
    assert salida.err == "ERROR: fila 1, evidencia 1: el simbolo no existe en el archivo\n"


def test_l3_zb_el_script_no_modifica_la_matriz_ni_el_codigo(raiz, tmp_path):
    ruta = escribir(tmp_path / "m.json", [fila_ok()])
    antes = (os.stat(ruta).st_mtime_ns, os.stat(raiz / "src" / "codigo.py").st_mtime_ns, sorted(os.listdir(tmp_path)))
    assert script.main(["--matriz", ruta, "--raiz", str(raiz)]) == 0
    assert antes == (os.stat(ruta).st_mtime_ns, os.stat(raiz / "src" / "codigo.py").st_mtime_ns, sorted(os.listdir(tmp_path)))
