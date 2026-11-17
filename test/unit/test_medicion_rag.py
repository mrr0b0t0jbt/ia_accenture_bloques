import datetime
import gc
import itertools
import json
from types import SimpleNamespace

import pytest
from chromadb.api.shared_system_client import SharedSystemClient

import indexacion
import medicion_rag
import recuperacion
from embeddings_falsos import EmbeddingsPorTema

TEMAS = {
    "parcheo": ["parche", "parchear", "parcheo", "plazo", "vulnerabilidad", "cvss"],
    "respaldos": ["respaldo", "respaldos", "copia", "restauracion", "rpo"],
    "phishing": ["phishing", "correo", "enlace", "clave", "remitente"],
    "registros": ["log", "logs", "auditoria", "registro", "siem"],
}
# cada documento cabe en un solo chunk (200 caracteres), así que los ids son <id>_0
CORPUS = [
    {"id": "parcheo", "fuente": "parcheo.md", "texto": "Politica de parcheo: el parche critico se aplica en cuatro dias por la vulnerabilidad con cvss alto."},
    {"id": "respaldos", "fuente": "respaldos.md", "texto": "Guia de respaldos: copia de respaldo diaria, prueba de restauracion mensual y rpo de 24 horas."},
    {"id": "phishing", "fuente": "phishing.md", "texto": "Playbook phishing: reportar el correo, no abrir el enlace y cambiar la clave si la escribio el remitente."},
    {"id": "registros", "fuente": "registros.md", "texto": "Registro de auditoria: los logs se envian al siem y se conservan noventa dias para consulta."},
    {"id": "accesos", "fuente": "accesos.md", "texto": "Control de accesos: la clave se rota cada noventa dias y se revisa el registro de cuentas privilegiadas."},
    {"id": "redes", "fuente": "redes.md", "texto": "Segmentacion de redes: el trafico entre zonas pasa por el firewall y se registra en los logs."},
    {"id": "incidentes", "fuente": "incidentes.md", "texto": "Gestion de incidentes: un incidente S1 se atiende de inmediato y se escala al responsable de turno."},
    {"id": "continuidad", "fuente": "continuidad.md", "texto": "Continuidad: el plan de recuperacion usa la copia de respaldo y el rpo acordado con cada area."},
]
PREGUNTA = "plazo del parche para una vulnerabilidad con cvss alto"
PRECIOS = (2e-6, 1e-5)
FECHA = datetime.datetime(2026, 11, 16, 19, 30, 0, tzinfo=datetime.timezone.utc)
CLAVES_REGISTRO = [
    "timestamp", "corrida", "id_caso", "pregunta", "k", "max_tokens", "tiempos_ms", "tiempo_total_ms", "reintentos_embed",
    "ids_recuperados", "caracteres_prompt", "tokens_prompt_aprox", "tokens", "costo_estimado", "truncado", "sin_datos_de_uso",
    "admite_falta_de_contexto", "respuesta", "error",
]


@pytest.fixture(autouse=True)
def _libera_chroma():
    """Cierra los clientes de Chroma al terminar: sin esto los descriptores abiertos superan 1024 y rompen select() en test_supervision."""
    yield
    gc.collect()
    SharedSystemClient.clear_system_cache()


@pytest.fixture
def embeddings():
    return EmbeddingsPorTema(TEMAS)


@pytest.fixture
def vectorstore(tmp_path, embeddings):
    indexacion.indexar(CORPUS, embeddings, str(tmp_path / "indice"))
    almacen = indexacion.abrir_vectorstore(embeddings, str(tmp_path / "indice"), indexacion.COLECCION_POR_DEFECTO)
    embeddings.llamadas_documentos = embeddings.llamadas_consulta = 0
    return almacen


def _reloj(valores):
    """Reloj falso que devuelve los valores dados en orden."""
    iterador = iter(valores)
    return lambda: next(iterador)


def _reloj_lineal():
    """Reloj falso que avanza 10 ms por lectura."""
    contador = itertools.count()
    return lambda: next(contador) * 0.01


class ModeloFalso:
    """Modelo falso: invoke devuelve el mensaje (o lo calcula desde el prompt) o lanza la excepción dada."""

    def __init__(self, respuesta=None, error=None):
        self.respuesta, self.error, self.prompts = respuesta, error, []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return self.respuesta(prompt) if callable(self.respuesta) else self.respuesta


def _mensaje(contenido="Respuesta [1].", uso=None, metadata=None, con_uso=True, con_metadata=True):
    campos = {"content": contenido}
    if con_uso:
        campos["usage_metadata"] = {"input_tokens": 523, "output_tokens": 129, "total_tokens": 652} if uso is None else uso
    if con_metadata:
        campos["response_metadata"] = metadata or {}
    return SimpleNamespace(**campos)


def _registrar(vectorstore, embeddings, modelo, pregunta=PREGUNTA, k=3, contador=None, max_consultas=10, reloj=None, max_tokens=512):
    return medicion_rag.registrar_consulta(
        "caso1", pregunta, k, vectorstore, embeddings, modelo, {"consultas": 0} if contador is None else contador,
        max_consultas, PRECIOS, "20261116T193000Z", max_tokens, reloj=reloj or _reloj_lineal(), ahora=lambda: FECHA, dormir=lambda s: None,
    )


class EmbeddingGuionado:
    """Embedding falso: embed_query n devuelve o lanza lo que diga guion(n) (n desde 1)."""

    def __init__(self, guion):
        self.guion, self.peticiones = guion, 0

    def embed_query(self, texto):
        self.peticiones += 1
        return self.guion(self.peticiones)


# M1
def test_m1_medir_guarda_los_segundos_solo_si_el_bloque_termina():
    tiempos = {}
    with medicion_rag.medir(tiempos, "x", _reloj([0.0, 0.25])):
        pass
    assert tiempos == {"x": 0.25}
    vacios = {}
    with pytest.raises(RuntimeError):
        with medicion_rag.medir(vacios, "x", _reloj([0.0, 0.25])):
            raise RuntimeError("fallo")
    assert vacios == {}


# M2
def test_m2_calcular_costo():
    assert medicion_rag.calcular_costo(523, 129, 2e-6, 1e-5) == pytest.approx(0.002336)
    assert medicion_rag.calcular_costo(0, 0, 2e-6, 1e-5) == 0


# M3
@pytest.mark.parametrize("largo, esperado", [(43, 10), (567, 141), (0, 0)])
def test_m3_aprox_tokens(largo, esperado):
    assert medicion_rag.aprox_tokens("a" * largo) == esperado


# M4
def test_m4_embeber_a_la_primera():
    embedding, contador = EmbeddingGuionado(lambda n: [0.1, 0.2]), {"consultas": 0}
    vector, segundos, reintentos = medicion_rag.embeber_con_reintentos(embedding, "p", contador, 5, reloj=_reloj([3.0, 3.4]))
    assert vector == [0.1, 0.2] and segundos == pytest.approx(0.4) and reintentos == 0
    assert contador == {"consultas": 1} and embedding.peticiones == 1


# M5
def test_m5_dos_429_y_luego_bien_el_tiempo_es_solo_el_del_intento_exitoso():
    def guion(n):
        if n <= 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return [0.5]

    embedding, contador, esperas = EmbeddingGuionado(guion), {"consultas": 0}, []
    # lecturas reales: los dos intentos fallidos leen solo la entrada (0 y 1) y el exitoso lee entrada y salida (10 y 11.5)
    vector, segundos, reintentos = medicion_rag.embeber_con_reintentos(
        embedding, "p", contador, 10, dormir=esperas.append, reloj=_reloj([0, 1, 10, 11.5])
    )
    assert vector == [0.5] and reintentos == 2 and contador["consultas"] == 3
    assert esperas == [5, 10] and segundos == pytest.approx(1.5)


# M6
def test_m6_el_tope_se_fuerza_en_ejecucion_sin_hacer_la_peticion_de_mas():
    def siempre_429(n):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    embedding, contador, esperas = EmbeddingGuionado(siempre_429), {"consultas": 0}, []
    with pytest.raises(medicion_rag.TopeDePeticiones) as info:
        medicion_rag.embeber_con_reintentos(embedding, "p", contador, 3, dormir=esperas.append)
    assert embedding.peticiones == 3 and contador["consultas"] == 3 and len(esperas) == 3
    assert "429" not in str(info.value) and "RESOURCE_EXHAUSTED" not in str(info.value)


# M7
def test_m7_un_error_que_no_es_de_limite_se_relanza_sin_reintentar():
    def clave_rechazada(n):
        raise RuntimeError("API key not valid")

    embedding, contador, esperas = EmbeddingGuionado(clave_rechazada), {"consultas": 0}, []
    with pytest.raises(RuntimeError, match="API key"):
        medicion_rag.embeber_con_reintentos(embedding, "p", contador, 10, dormir=esperas.append)
    assert embedding.peticiones == 1 and contador["consultas"] == 1 and esperas == []


# M8
def test_m8_registro_feliz_con_k3(vectorstore, embeddings):
    modelo = ModeloFalso(_mensaje())
    # lecturas: embed (0, 0.1), busqueda (1, 1.05), generacion (2, 2.5)
    registro = _registrar(vectorstore, embeddings, modelo, reloj=_reloj([0, 0.1, 1, 1.05, 2, 2.5]))
    assert list(registro) == CLAVES_REGISTRO
    assert registro["tiempos_ms"] == {"embed_query": 100.0, "busqueda_vectorial": 50.0, "generacion": 500.0}
    assert registro["tiempo_total_ms"] == pytest.approx(650.0)
    esperados = recuperacion.recuperar(vectorstore, PREGUNTA, 3)
    assert registro["ids_recuperados"] == [f"{d.metadata['documento']}_{d.metadata['fragmento']}" for d, _ in esperados]
    por_vector = vectorstore.similarity_search_by_vector_with_relevance_scores(embeddings.embed_query(PREGUNTA), k=3)
    assert [d for _, d in esperados] == pytest.approx([d for _, d in por_vector])
    assert registro["tokens"] == {"input_tokens": 523, "output_tokens": 129, "total_tokens": 652}
    assert registro["costo_estimado"] == pytest.approx(523 * 2e-6 + 129 * 1e-5)
    assert registro["truncado"] is False and registro["error"] is None and registro["sin_datos_de_uso"] is False
    assert registro["tokens_prompt_aprox"] == len(modelo.prompts[0]) // 4 and registro["caracteres_prompt"] == len(modelo.prompts[0])
    assert registro["timestamp"] == FECHA.isoformat() and registro["corrida"] == "20261116T193000Z"
    assert (registro["id_caso"], registro["k"], registro["max_tokens"], registro["reintentos_embed"]) == ("caso1", 3, 512, 0)


# M9
def test_m8b_el_registro_guarda_el_max_tokens_recibido(vectorstore, embeddings):
    # mutante: max_tokens fijo en 512
    assert _registrar(vectorstore, embeddings, ModeloFalso(_mensaje()), max_tokens=256)["max_tokens"] == 256


def test_m8c_reintentos_embed_cuenta_los_reintentos(vectorstore, embeddings):
    # mutante: reintentos_embed siempre 0
    def una_falla(n):
        if n == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return embeddings.embed_query(PREGUNTA)

    con_falla = _registrar(vectorstore, EmbeddingGuionado(una_falla), ModeloFalso(_mensaje()))
    sin_falla = _registrar(vectorstore, EmbeddingGuionado(lambda n: embeddings.embed_query(PREGUNTA)), ModeloFalso(_mensaje()))
    assert con_falla["reintentos_embed"] == 1 and sin_falla["reintentos_embed"] == 0


def test_m9_el_prompt_es_igual_al_de_construir_prompt(vectorstore, embeddings):
    modelo = ModeloFalso(_mensaje())
    _registrar(vectorstore, embeddings, modelo, k=4)
    assert modelo.prompts == [recuperacion.construir_prompt(PREGUNTA, vectorstore, 4)[0]]


# M10
def test_m10_una_sola_peticion_de_embedding_y_sin_recuperar(vectorstore, embeddings, monkeypatch):
    def prohibido(*args, **kwargs):
        raise AssertionError("no debe llamarse")

    monkeypatch.setattr(recuperacion, "recuperar", prohibido)
    _registrar(vectorstore, embeddings, ModeloFalso(_mensaje()))
    assert embeddings.llamadas_consulta == 1 and embeddings.llamadas_documentos == 0


# M11
def test_m11_mas_k_mas_entrada(vectorstore, embeddings):
    modelo = ModeloFalso(lambda prompt: _mensaje(uso={"input_tokens": len(prompt) // 4, "output_tokens": 10}))
    registros = [_registrar(vectorstore, embeddings, modelo, k=k) for k in (1, 3, 6)]
    caracteres = [r["caracteres_prompt"] for r in registros]
    entradas = [r["tokens"]["input_tokens"] for r in registros]
    assert caracteres == sorted(set(caracteres)) and entradas == sorted(set(entradas))
    assert [len(r["ids_recuperados"]) for r in registros] == [1, 3, 6]


# M12
def test_m12_un_error_del_modelo_queda_en_el_registro(vectorstore, embeddings):
    registro = _registrar(vectorstore, embeddings, ModeloFalso(error=RuntimeError("fallo")))
    assert registro["error"] == "RuntimeError: fallo"
    assert registro["tokens"] is None and registro["costo_estimado"] is None
    assert registro["tiempos_ms"]["generacion"] is None
    assert registro["tiempos_ms"]["embed_query"] is not None and registro["tiempos_ms"]["busqueda_vectorial"] is not None
    assert registro["tiempo_total_ms"] == pytest.approx(registro["tiempos_ms"]["embed_query"] + registro["tiempos_ms"]["busqueda_vectorial"])
    assert registro["sin_datos_de_uso"] is False and registro["respuesta"] == ""


def test_m12b_el_error_del_modelo_se_recorta_a_300_caracteres(vectorstore, embeddings):
    # mutante: sin recorte
    registro = _registrar(vectorstore, embeddings, ModeloFalso(error=RuntimeError("x" * 500)))
    prefijo = "RuntimeError: "
    assert registro["error"].startswith(prefijo) and len(registro["error"]) == len(prefijo) + 300


# M13
@pytest.mark.parametrize(
    "mensaje",
    [
        _mensaje(uso={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}),
        _mensaje(uso={"input_tokens": None, "output_tokens": 5}),
        _mensaje(uso={"input_tokens": True, "output_tokens": 5}),
        _mensaje(uso={"input_tokens": "5", "output_tokens": 5}),
        _mensaje(con_uso=False),
    ],
    ids=["ceros", "none", "bool", "texto", "ausente"],
)
def test_m13_uso_faltante(vectorstore, embeddings, mensaje):
    registro = _registrar(vectorstore, embeddings, ModeloFalso(mensaje))
    assert registro["sin_datos_de_uso"] is True and registro["tokens"] is None and registro["costo_estimado"] is None
    assert registro["error"] is None


def test_m13b_sin_output_tokens_cuenta_cero_salida(vectorstore, embeddings):
    registro = _registrar(vectorstore, embeddings, ModeloFalso(_mensaje(uso={"input_tokens": 100})))
    assert registro["tokens"] == {"input_tokens": 100, "output_tokens": 0, "total_tokens": 100}
    assert registro["costo_estimado"] == pytest.approx(100 * 2e-6) and registro["sin_datos_de_uso"] is False


# M14
@pytest.mark.parametrize(
    "mensaje, esperado",
    [
        (_mensaje(metadata={"stop_reason": "max_tokens"}), True),
        (_mensaje(metadata={"stop_reason": "end_turn"}), False),
        (_mensaje(con_metadata=False), False),
    ],
    ids=["max_tokens", "end_turn", "sin_metadata"],
)
def test_m14_truncado(vectorstore, embeddings, mensaje, esperado):
    assert _registrar(vectorstore, embeddings, ModeloFalso(mensaje))["truncado"] is esperado


# M15
def test_m15_respuesta_en_bloques_y_falta_de_contexto(vectorstore, embeddings):
    bloques = [{"type": "thinking", "thinking": "secreto"}, {"type": "text", "text": "Hola [1]"}]
    registro = _registrar(vectorstore, embeddings, ModeloFalso(_mensaje(bloques)))
    assert registro["respuesta"] == "Hola [1]" and registro["admite_falta_de_contexto"] is False
    registro = _registrar(vectorstore, embeddings, ModeloFalso(_mensaje(recuperacion.FRASE_SIN_CONTEXTO)))
    assert registro["admite_falta_de_contexto"] is True


# M16
def test_m16_la_respuesta_se_recorta(vectorstore, embeddings):
    registro = _registrar(vectorstore, embeddings, ModeloFalso(_mensaje("a" * 5000)))
    assert len(registro["respuesta"]) == medicion_rag.MAX_CARACTERES_RESPUESTA


# M17
@pytest.mark.parametrize("k", [0, 7, True, "3"], ids=["cero", "siete", "bool", "texto"])
def test_m17_k_invalido(vectorstore, embeddings, k):
    modelo = ModeloFalso(_mensaje())
    with pytest.raises(ValueError):
        _registrar(vectorstore, embeddings, modelo, k=k)
    assert embeddings.llamadas_consulta == 0 and modelo.prompts == []


# M18
@pytest.mark.parametrize("pregunta", ["", "   ", "a" * 501], ids=["vacia", "espacios", "larga"])
def test_m18_pregunta_invalida(vectorstore, embeddings, pregunta):
    modelo = ModeloFalso(_mensaje())
    with pytest.raises(ValueError):
        _registrar(vectorstore, embeddings, modelo, pregunta=pregunta)
    assert embeddings.llamadas_consulta == 0 and modelo.prompts == []


def test_m18b_una_pregunta_de_exactamente_500_caracteres_pasa(vectorstore, embeddings):
    assert _registrar(vectorstore, embeddings, ModeloFalso(_mensaje()), pregunta="a" * 500)["error"] is None


def _reg(id_caso, k, tiempos, entrada, salida, costo, error=None, con_tokens=True):
    embed, busqueda, generacion = tiempos
    return {
        "id_caso": id_caso, "k": k, "error": error,
        "tiempos_ms": {"embed_query": embed, "busqueda_vectorial": busqueda, "generacion": generacion},
        "tiempo_total_ms": sum(t for t in tiempos if t is not None),
        "tokens": {"input_tokens": entrada, "output_tokens": salida, "total_tokens": entrada + salida} if con_tokens else None,
        "costo_estimado": costo if con_tokens else None,
    }


REGISTROS_A_MANO = [
    _reg("A", 3, (100, 10, 1000), 500, 100, 0.002),
    _reg("A", 6, (120, 12, 1200), 900, 120, 0.003),
    _reg("B", 3, (200, 20, 2000), 600, 200, 0.004),
    _reg("B", 6, (300, 30, 100), 550, 50, 0.001),  # su entrada no crece con k y su generación es menor que el embed
]


# M19
def test_m19_resumir_a_mano():
    resumen = medicion_rag.resumir(REGISTROS_A_MANO)
    assert resumen["llamadas"] == 4 and resumen["con_error"] == 0 and list(resumen["por_k"]) == [3, 6]
    k3, k6 = resumen["por_k"][3], resumen["por_k"][6]
    assert k3["registros"] == 2 and k3["promedio_ms"] == {"embed_query": 150.0, "busqueda_vectorial": 15.0, "generacion": 1500.0, "total": 1665.0}
    assert (k3["promedio_input_tokens"], k3["promedio_output_tokens"]) == (550.0, 150.0)
    assert k3["costo_total"] == pytest.approx(0.006)
    assert k6["registros"] == 2 and k6["promedio_ms"] == {"embed_query": 210.0, "busqueda_vectorial": 21.0, "generacion": 650.0, "total": 881.0}
    assert (k6["promedio_input_tokens"], k6["promedio_output_tokens"]) == (725.0, 85.0)
    assert k6["costo_total"] == pytest.approx(0.004)
    assert resumen["tokens_crecen_con_k"] == {"si": False, "casos_que_no_crecen": ["B"]}
    assert resumen["generacion_mas_lenta"] == {"si": 3, "total": 4}


# M20
def test_m20_resumir_ignora_errores_y_registros_sin_tokens():
    registros = [
        *REGISTROS_A_MANO[:2],
        _reg("C", 3, (1, 1, None), 0, 0, 0, error="RuntimeError: fallo", con_tokens=False),
        _reg("D", 3, (5000, 5000, 5000), 0, 0, 0, con_tokens=False),
    ]
    resumen = medicion_rag.resumir(registros)
    assert resumen["llamadas"] == 4 and resumen["con_error"] == 1
    assert list(resumen["por_k"]) == [3, 6] and resumen["por_k"][3]["registros"] == 1
    assert resumen["por_k"][3]["promedio_ms"]["embed_query"] == 100.0
    assert resumen["tokens_crecen_con_k"] == {"si": True, "casos_que_no_crecen": []}
    assert resumen["generacion_mas_lenta"] == {"si": 2, "total": 2}


def test_m20c_un_empate_de_input_tokens_entre_dos_k_no_cuenta_como_crecimiento():
    # mutante: >= en vez de >
    empate = [_reg("E", 3, (1, 1, 1), 700, 10, 0.001), _reg("E", 6, (1, 1, 1), 700, 10, 0.001)]
    assert medicion_rag.resumir(empate)["tokens_crecen_con_k"] == {"si": False, "casos_que_no_crecen": ["E"]}


def test_m20d_los_promedios_se_redondean_a_2_decimales():
    # mutante: round a 3 decimales
    registros = [_reg(f"F{n}", 3, (t, 1, 1), 500, 100, 0.001) for n, t in enumerate((1.0, 1.0, 2.0))]
    assert medicion_rag.resumir(registros)["por_k"][3]["promedio_ms"]["embed_query"] == 1.33


def test_m20b_lista_vacia_y_un_solo_k():
    assert medicion_rag.resumir([]) == {
        "llamadas": 0, "con_error": 0, "por_k": {},
        "tokens_crecen_con_k": {"si": None, "casos_que_no_crecen": []}, "generacion_mas_lenta": {"si": 0, "total": 0},
    }
    assert medicion_rag.resumir([REGISTROS_A_MANO[0], REGISTROS_A_MANO[2]])["tokens_crecen_con_k"]["si"] is None


# M21
def test_m21_anexar_registro(tmp_path):
    ruta = tmp_path / "registros.jsonl"
    medicion_rag.anexar_registro(str(ruta), {"a": 1, "texto": "información útil"})
    primera = ruta.read_bytes()
    assert primera.endswith(b"\n") and "información útil".encode("utf-8") in primera
    medicion_rag.anexar_registro(str(ruta), {"b": 2})
    contenido = ruta.read_bytes()
    assert contenido.startswith(primera) and contenido.count(b"\n") == 2
    assert [json.loads(linea) for linea in contenido.decode("utf-8").splitlines()] == [{"a": 1, "texto": "información útil"}, {"b": 2}]


# M22
@pytest.mark.parametrize(
    "linea_mala",
    ['{"timestamp": 1, "tiempos_ms": {}', "[1, 2]", '{"timestamp": 1}'],
    ids=["corrupta", "no_objeto", "sin_tiempos"],
)
def test_m22_leer_registros_rechaza_la_linea_mala_con_su_numero(tmp_path, linea_mala):
    ruta = tmp_path / "r.jsonl"
    bueno = json.dumps({"timestamp": "t", "tiempos_ms": {}})
    ruta.write_text(f"{bueno}\n\n{linea_mala}\n", encoding="utf-8")  # la línea mala es la 3
    with pytest.raises(ValueError, match="línea 3"):
        medicion_rag.leer_registros(str(ruta))


def test_m22b_leer_registros_valido_ignora_lineas_en_blanco(tmp_path):
    ruta = tmp_path / "r.jsonl"
    bueno = json.dumps({"timestamp": "t", "tiempos_ms": {}})
    ruta.write_text(f"{bueno}\n\n   \n{bueno}\n", encoding="utf-8")
    assert len(medicion_rag.leer_registros(str(ruta))) == 2
