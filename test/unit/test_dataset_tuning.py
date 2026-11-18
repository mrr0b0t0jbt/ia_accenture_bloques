import copy
import json
import socket
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
from google import genai
from google.genai import types

import dataset_tuning
import enrutamiento
from dataset_tuning import (
    CLAVES,
    MAX_BYTES_ARCHIVO,
    MIN_EJEMPLOS,
    VALORES_PERMITIDOS,
    avisos_de_balance,
    cargar_dataset,
    construir_ejemplos,
    contar_por_clave,
    normalizar_texto,
    resumir_balance,
    validar_datos,
    validar_dataset,
)

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_DATASET = RAIZ / "JSON" / "dataset_tuning_clasificacion.json"
_POR_DEFECTO = object()
MENSAJE_MINIMO = "el dataset tiene {n} ejemplos; el minimo es 10"
CANONICO = "ejemplo 9: output no esta en el formato canonico (claves en orden y separadores por defecto)"


def _salida(categoria="incidente", prioridad="alta") -> str:
    return json.dumps({"categoria": categoria, "prioridad": prioridad}, ensure_ascii=False)


def _ej(texto, output=_POR_DEFECTO):
    return SimpleNamespace(text_input=texto, output=_salida() if output is _POR_DEFECTO else output)


def _validos(n: int) -> list:
    return [_ej(f"mensaje valido numero {i}") for i in range(n)]


def _problemas_del_ultimo(ejemplo) -> list[str]:
    """Problemas de un dataset de 9 ejemplos válidos más el ejemplo dado (índice 9)."""
    return validar_dataset(_validos(MIN_EJEMPLOS - 1) + [ejemplo])


def _dataset_real() -> dict:
    return cargar_dataset(RUTA_DATASET)


def _datos(ejemplos=None) -> dict:
    return {
        "claves": list(CLAVES),
        "valores_permitidos": {k: list(v) for k, v in VALORES_PERMITIDOS.items()},
        "ejemplos": ejemplos if ejemplos is not None else [{"id": "ej_01", "texto": "hola", "salida": {"categoria": "incidente", "prioridad": "alta"}}],
    }


# D1
def test_d1_constantes():
    assert CLAVES == ("categoria", "prioridad")
    assert VALORES_PERMITIDOS == {"categoria": ("incidente", "vulnerabilidad", "consulta"), "prioridad": ("alta", "media", "baja")}
    assert VALORES_PERMITIDOS["categoria"] == enrutamiento.CATEGORIAS


# D2
def test_d2_dataset_valido():
    assert validar_dataset(_validos(10)) == []


# D3
@pytest.mark.parametrize(
    "texto, esperado",
    [("", "text_input vacio"), ("   \n", "text_input vacio"), (None, "text_input no es texto"), (5, "text_input no es texto")],
)
def test_d3_text_input(texto, esperado):
    assert _problemas_del_ultimo(_ej(texto)) == [f"ejemplo 9: {esperado}"]


# D4
@pytest.mark.parametrize("largo, esperado", [(1000, []), (1001, ["ejemplo 9: text_input supera los 1000 caracteres"])])
def test_d4_largo(largo, esperado):
    assert _problemas_del_ultimo(_ej("a" * largo)) == esperado


# D5
@pytest.mark.parametrize(
    "texto, esperado",
    [
        ("a\x1bb", ["ejemplo 9: text_input tiene caracteres de control"]),
        ("a\x00b", ["ejemplo 9: text_input tiene caracteres de control"]),
        ("a\rb", ["ejemplo 9: text_input tiene caracteres de control"]),
        ("linea1\nlinea2\tx", []),
    ],
)
def test_d5_caracteres_de_control(texto, esperado):
    assert _problemas_del_ultimo(_ej(texto)) == esperado


# D6
@pytest.mark.parametrize(
    "output, esperado",
    [("", "output vacio"), ("  ", "output vacio"), (None, "output no es texto"), (5, "output no es texto")],
)
def test_d6_output_vacio_o_sin_texto(output, esperado):
    assert _problemas_del_ultimo(_ej("texto nuevo", output)) == [f"ejemplo 9: {esperado}"]


# D7
@pytest.mark.parametrize(
    "output",
    [
        "no es json",
        "{" + chr(39) + "categoria" + chr(39) + ": " + chr(39) + "incidente" + chr(39) + "}",
        '{"categoria": "incidente",}',
    ],
)
def test_d7_no_es_json_valido(output):
    assert _problemas_del_ultimo(_ej("texto nuevo", output)) == ["ejemplo 9: output no es JSON valido"]


# D8
@pytest.mark.parametrize("output", ["5", "[1]", '"texto"', "null", "true"])
def test_d8_no_es_un_objeto(output):
    assert _problemas_del_ultimo(_ej("texto nuevo", output)) == ["ejemplo 9: output no es un objeto JSON"]


# D9
@pytest.mark.parametrize(
    "output, esperado",
    [
        ('{"categoria": "incidente"}', ['ejemplo 9: output no tiene la clave "prioridad"']),
        (
            '{"categoria": "incidente", "prioridad": "alta", "nota": "x"}',
            [f"ejemplo 9: output tiene claves de mas: {repr('nota')}"],
        ),
        (
            '{"categoria": "incidente", "nota": "x"}',
            ['ejemplo 9: output no tiene la clave "prioridad"', f"ejemplo 9: output tiene claves de mas: {repr('nota')}"],
        ),
    ],
)
def test_d9_claves_faltantes_y_sobrantes(output, esperado):
    assert _problemas_del_ultimo(_ej("texto nuevo", output)) == esperado


# D10
def test_d10_clave_repetida():
    output = '{"categoria": "consulta", "categoria": "incidente", "prioridad": "alta"}'
    assert _problemas_del_ultimo(_ej("texto nuevo", output)) == [f"ejemplo 9: output repite la clave {repr('categoria')}"]


# D11
@pytest.mark.parametrize(
    "clave, valor",
    [("prioridad", "urgente"), ("categoria", "Incidente"), ("prioridad", 5), ("prioridad", ["alta"])],
)
def test_d11_valor_no_permitido(clave, valor):
    salida = {"categoria": "incidente", "prioridad": "alta", clave: valor}
    problemas = _problemas_del_ultimo(_ej("texto nuevo", json.dumps(salida)))
    assert problemas == [f'ejemplo 9: output tiene un valor no permitido para "{clave}"']
    assert str(valor) not in problemas[0].replace(clave, "")


# D12
@pytest.mark.parametrize(
    "output",
    [
        '{"prioridad": "alta", "categoria": "incidente"}',
        '{"categoria":"incidente","prioridad":"alta"}',
        '{"categoria" : "incidente", "prioridad": "alta"}',
    ],
)
def test_d12_formato_no_canonico(output):
    assert _problemas_del_ultimo(_ej("texto nuevo", output)) == [CANONICO]


# D13
@pytest.mark.parametrize(
    "variante, salida_distinta",
    [("  Hola   MUNDO", False), ("hola\nmundo", False), ("hola\tmundo", False), ("hola mundo", True)],
)
def test_d13_duplicados(variante, salida_distinta):
    otra = _salida("consulta", "baja") if salida_distinta else _salida()
    ejemplos = [_ej("hola mundo"), _ej(variante, otra)] + _validos(8)
    sufijo = " con otra salida" if salida_distinta else ""
    assert validar_dataset(ejemplos) == [f"ejemplo 1: text_input duplicado del ejemplo 0{sufijo}"]
    # un texto vacío nunca es duplicado de otro vacío
    vacios = validar_dataset([_ej(""), _ej("")] + _validos(8))
    assert vacios == ["ejemplo 0: text_input vacio", "ejemplo 1: text_input vacio"]


# D14
def test_d14_varios_problemas_en_un_ejemplo():
    problemas = _problemas_del_ultimo(_ej("", '{"prioridad": "urgente"}'))
    assert problemas == [
        "ejemplo 9: text_input vacio",
        'ejemplo 9: output no tiene la clave "categoria"',
        'ejemplo 9: output tiene un valor no permitido para "prioridad"',
    ]


def test_d14_orden_global_luego_indice():
    ejemplos = [_ej(None), _ej("segundo texto"), _ej("tercer texto", "no es json")]
    assert validar_dataset(ejemplos) == [
        MENSAJE_MINIMO.format(n=3),
        "ejemplo 0: text_input no es texto",
        "ejemplo 2: output no es JSON valido",
    ]


# D15
@pytest.mark.parametrize(
    "n, esperado",
    [
        (0, [MENSAJE_MINIMO.format(n=0)]),
        (9, [MENSAJE_MINIMO.format(n=9)]),
        (10, []),
        (501, ["el dataset tiene 501 ejemplos; el maximo es 500"]),
    ],
)
def test_d15_cantidad(n, esperado):
    assert validar_dataset(_validos(n)) == esperado


# D16
def test_d16_conteo_del_dataset_real():
    ejemplos = construir_ejemplos(_dataset_real())
    for clave in CLAVES:
        assert sorted(contar_por_clave(ejemplos, clave).values()) == [8, 8, 8]


def test_d16_conteo_ignora_outputs_rotos():
    ejemplos = [
        _ej("a"),
        _ej("b", _salida("consulta", "alta")),
        _ej("c", None),
        _ej("d", ""),
        _ej("e", "no es json"),
        _ej("f", "[1]"),
        _ej("g", '{"categoria": "consulta"}'),
        _ej("h", '{"prioridad": 5}'),
    ]
    assert contar_por_clave(ejemplos, "prioridad") == Counter({"alta": 2})
    assert contar_por_clave(ejemplos, "categoria") == Counter({"incidente": 1, "consulta": 2})


def test_d16_resumen_con_razon():
    resumen = resumir_balance(Counter({"a": 3, "b": 2, "c": 1, "d": 1, "e": 1}))
    assert resumen["razon"] == 3.0 and resumen["maximo"] == 3 and resumen["minimo"] == 1 and resumen["ausentes"] == []
    assert list(resumen["conteo"]) == ["a", "b", "c", "d", "e"]


def test_d16_valor_ausente():
    resumen = resumir_balance(Counter(incidente=3, consulta=1), VALORES_PERMITIDOS["categoria"])
    assert resumen["razon"] is None and resumen["ausentes"] == ["vulnerabilidad"] and resumen["minimo"] == 0
    assert resumen["conteo"] == {"incidente": 3, "consulta": 1, "vulnerabilidad": 0}
    assert avisos_de_balance({"categoria": resumen}) == ["aviso: categoria no tiene ejemplos de: vulnerabilidad"]


def test_d16_conteo_vacio():
    resumen = resumir_balance(Counter())
    assert resumen["razon"] is None and resumen["maximo"] == 0 and resumen["minimo"] == 0 and resumen["conteo"] == {}


def test_d16_razon_dos_no_avisa():
    resumen = resumir_balance(Counter(a=4, b=2))
    assert resumen["razon"] == 2.0
    assert avisos_de_balance({"k": resumen}) == []


def test_d16_razon_mayor_avisa():
    resumen = resumir_balance(Counter(a=5, b=2))
    assert avisos_de_balance({"k": resumen}) == [
        "aviso: la razon max/min de k es 2.5x (mayor que 2.0): conviene agregar ejemplos a los valores menos representados"
    ]


# D17
def test_d17_construye_tuning_examples_con_acentos():
    salida = {"categoria": "acción", "prioridad": "alta"}
    resultado = construir_ejemplos(_datos([{"id": "ej_01", "texto": "¿Qué pasó?", "salida": salida}]))
    assert len(resultado) == 1 and isinstance(resultado[0], types.TuningExample)
    assert resultado[0].text_input == "¿Qué pasó?"
    assert resultado[0].output == json.dumps(salida, ensure_ascii=False) and "acción" in resultado[0].output


@pytest.mark.parametrize("datos", [[], {"claves": list(CLAVES)}, _datos([{"id": "ej_01", "salida": {}}])])
def test_d17_datos_invalidos_levantan_valueerror(datos):
    with pytest.raises(ValueError):
        construir_ejemplos(datos)


def test_d17_no_modifica_los_datos():
    datos = _dataset_real()
    copia = copy.deepcopy(datos)
    construir_ejemplos(datos)
    assert datos == copia


# D18
def test_d18_archivo_inexistente(tmp_path):
    with pytest.raises(ValueError, match="no existe"):
        cargar_dataset(tmp_path / "no_existe.json")


@pytest.mark.parametrize("contenido", ["{roto", "[1, 2]"])
def test_d18_json_invalido_o_que_no_es_objeto(tmp_path, contenido):
    ruta = tmp_path / "d.json"
    ruta.write_text(contenido, encoding="utf-8")
    with pytest.raises(ValueError):
        cargar_dataset(ruta)


def test_d18_archivo_demasiado_grande_no_se_lee(tmp_path, monkeypatch):
    ruta = tmp_path / "grande.json"
    ruta.write_text(" " * (MAX_BYTES_ARCHIVO + 1), encoding="utf-8")

    def prohibido(*a, **k):
        raise AssertionError("no debe leer el archivo")

    monkeypatch.setattr(Path, "read_bytes", prohibido)
    monkeypatch.setattr(Path, "read_text", prohibido)
    with pytest.raises(ValueError, match="pesa"):
        cargar_dataset(ruta)


def test_d18_archivo_valido(tmp_path):
    ruta = tmp_path / "d.json"
    ruta.write_text('{"a": 1}', encoding="utf-8")
    assert cargar_dataset(ruta) == {"a": 1}


# D19
@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    "campo, config",
    [
        ("adapter_size", {"adapter_size": types.AdapterSize.ADAPTER_SIZE_FOUR}),
        ("tuning_mode", {"tuning_mode": types.TuningMode.TUNING_MODE_PEFT_ADAPTER}),
    ],
)
def test_d19_la_api_de_desarrolladores_rechaza_peft(monkeypatch, campo, config):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    def sin_red(*a, **k):
        raise AssertionError("la prueba no debe abrir red")

    monkeypatch.setattr(socket.socket, "connect", sin_red)
    cliente = genai.Client(api_key="clave-falsa", http_options=types.HttpOptions(base_url="http://127.0.0.1:9"))
    with pytest.raises(ValueError, match=campo):
        cliente.tunings.tune(
            base_model="models/gemini-2.0-flash",
            training_dataset=types.TuningDataset(examples=[types.TuningExample(text_input="a", output="b")]),
            config=types.CreateTuningJobConfig(**config),
        )


# D20
def _textos(nodo):
    if isinstance(nodo, str):
        yield nodo
    elif isinstance(nodo, dict):
        for valor in nodo.values():
            yield from _textos(valor)
    elif isinstance(nodo, list):
        for valor in nodo:
            yield from _textos(valor)


def test_d20_dataset_real_valido_y_balanceado():
    datos = _dataset_real()
    assert validar_datos(datos) == []
    ejemplos = construir_ejemplos(datos)
    assert len(ejemplos) == 24 and validar_dataset(ejemplos) == []
    for clave in CLAVES:
        assert dict(contar_por_clave(ejemplos, clave)) == {v: 8 for v in VALORES_PERMITIDOS[clave]}


def test_d20_sin_solapamiento_con_los_casos_del_repo():
    archivos = [
        "casos_cadenas", "casos_agente", "casos_metricas", "casos_pipeline",
        "casos_recuperacion", "evaluacion_recuperacion", "casos_latencia_costo",
    ]
    otros = set()
    for nombre in archivos:
        contenido = json.loads((RAIZ / "JSON" / f"{nombre}.json").read_text(encoding="utf-8"))
        otros |= {normalizar_texto(t) for t in _textos(contenido) if len(t) > 25}
    assert otros
    for ejemplo in _dataset_real()["ejemplos"]:
        assert normalizar_texto(ejemplo["texto"]) not in otros


def test_d20_ids_y_largos():
    ejemplos = _dataset_real()["ejemplos"]
    assert [e["id"] for e in ejemplos] == [f"ej_{i:02d}" for i in range(1, 25)]
    assert all(20 <= len(e["texto"]) <= 300 for e in ejemplos)


# D21
_LECCION = [
    ("TimeoutError al llamar a la API", "red"),
    ("ValueError por clave invalida", "autenticacion"),
    ("Error 429 por exceso de peticiones", "cuota"),
    ("Falta la API key en el entorno", "autenticacion"),
    ("sqlite3 no abre la base de datos", "entorno"),
    ("ModuleNotFoundError al importar un modulo", "entorno"),
    ("PermissionError al escribir el archivo", "entorno"),
    ("InvalidArgumentError por dimension distinta", "consistencia_indice"),
]
_CLAVES_LECCION = ("categoria", "accion_sugerida")


def _leccion() -> list:
    return [_ej(t, json.dumps({"categoria": c, "accion_sugerida": "x"})) for t, c in _LECCION]


def test_d21_leccion_solo_falta_el_minimo():
    ejemplos = _leccion()
    assert validar_dataset(ejemplos, claves=_CLAVES_LECCION, valores_permitidos=None) == [MENSAJE_MINIMO.format(n=8)]
    resumen = resumir_balance(contar_por_clave(ejemplos, "categoria"))
    assert resumen["conteo"]["entorno"] == 3 and resumen["conteo"]["autenticacion"] == 2 and resumen["razon"] == 3.0


def test_d21_leccion_con_dos_ejemplos_rotos():
    ejemplos = _leccion() + [_ej("", '{"categoria": "red"}'), _ej(_LECCION[0][0], "no es json")]
    assert validar_dataset(ejemplos, claves=_CLAVES_LECCION, valores_permitidos=None) == [
        "ejemplo 8: text_input vacio",
        'ejemplo 8: output no tiene la clave "accion_sugerida"',
        "ejemplo 9: text_input duplicado del ejemplo 0 con otra salida",
        "ejemplo 9: output no es JSON valido",
    ]


# D22
def test_d22_ids_repetidos():
    ejemplo = {"id": "ej_01", "texto": "hola", "salida": {}}
    assert validar_datos(_datos([ejemplo, dict(ejemplo)])) == [f"ejemplos[1] repite el id {repr('ej_01')}"]


def test_d22_claves_del_archivo_distintas():
    datos = _datos()
    datos["claves"] = ["categoria"]
    assert validar_datos(datos) == ["claves del archivo distintas de CLAVES"]


def test_d22_valores_permitidos_del_archivo_distintos():
    datos = _datos()
    datos["valores_permitidos"]["prioridad"] = ["alta"]
    assert validar_datos(datos) == ["valores_permitidos del archivo distintos de VALORES_PERMITIDOS"]


def test_d22_ejemplo_que_no_es_objeto():
    assert validar_datos(_datos(["texto suelto"])) == ["ejemplos[0] no es un objeto"]
