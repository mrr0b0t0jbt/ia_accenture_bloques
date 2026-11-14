import datetime
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from chromadb.api.shared_system_client import SharedSystemClient
from langchain_core.documents import Document

import embeddings_gemini
import indexacion
import metricas_recuperacion
import recuperacion
from embeddings_falsos import EmbeddingsPorTema

RAIZ = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location("evaluar_recuperacion", RAIZ / "scripts" / "evaluar_recuperacion.py")
evaluar = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluar)

TEMAS = {
    "parcheo": ["parche", "parchear", "parcheo", "plazo", "vulnerabilidad", "cvss"],
    "respaldos": ["respaldo", "respaldos", "copia", "restauracion", "rpo"],
    "phishing": ["phishing", "correo", "enlace", "clave", "remitente"],
}
# cada documento cabe en un solo chunk (200 caracteres), así que los ids son parcheo_0, respaldos_0 y phishing_0
CORPUS = [
    {"id": "parcheo", "fuente": "parcheo.md", "texto": "Politica de parcheo: el parche critico se aplica en cuatro dias por la vulnerabilidad con cvss alto."},
    {"id": "respaldos", "fuente": "respaldos.md", "texto": "Guia de respaldos: copia de respaldo diaria, prueba de restauracion mensual y rpo de 24 horas."},
    {"id": "phishing", "fuente": "phishing.md", "texto": "Playbook phishing: reportar el correo, no abrir el enlace y cambiar la clave si la escribio el remitente."},
]
CASOS = [
    {"id": "c1_parcheo", "pregunta": "plazo parche vulnerabilidad cvss", "ids_relevantes": ["parcheo_0"], "respondible": True},
    {"id": "c2_respaldos", "pregunta": "respaldo copia restauracion rpo", "ids_relevantes": ["respaldos_0"], "respondible": True},
    {"id": "c3_sin_respuesta", "pregunta": "presupuesto del correo", "ids_relevantes": [], "respondible": False},
    {"id": "c4_falla_en_k1", "pregunta": "phishing enlace correo clave", "ids_relevantes": ["respaldos_0"], "respondible": True},
]
CLAVES_SALIDA = [
    "nota_costo", "fecha", "configuracion", "consultas_de_embeddings", "interrumpido", "incompleto", "error", "determinista",
    "max_dif_distancia", "nota_metodo", "promedios_respondibles", "sin_respuesta", "por_caso",
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


def _escribir(ruta: Path, datos) -> Path:
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    return ruta


def _preparar(tmp_path, monkeypatch, embeddings, casos=CASOS, indexar=True) -> Path:
    """Escribe corpus y casos en tmp_path, indexa el corpus de prueba y parchea la fábrica de embeddings."""
    _escribir(tmp_path / "corpus.json", CORPUS)
    _escribir(tmp_path / "casos.json", casos)
    directorio = tmp_path / "indice"
    if indexar:
        indexacion.indexar(CORPUS, embeddings, str(directorio))
    embeddings.llamadas_consulta = 0
    monkeypatch.setattr(embeddings_gemini, "crear_embeddings", lambda: embeddings)
    return directorio


def _argv(tmp_path, directorio, *extra, salida=None) -> list[str]:
    return [
        "--confirmar-api-real", "--casos", str(tmp_path / "casos.json"), "--corpus", str(tmp_path / "corpus.json"),
        "--directorio", str(directorio), "--salida", str(salida or tmp_path / "salida.json"), *extra,
    ]


def _leer(tmp_path) -> dict:
    return json.loads((tmp_path / "salida.json").read_text(encoding="utf-8"))


def _codigo(argv) -> int:
    """Código de salida de main; un rechazo de argparse (SystemExit) también cuenta."""
    try:
        return evaluar.main(argv)
    except SystemExit as salida:
        return salida.code


def test_01_sin_confirmar_devuelve_2_y_no_importa_el_cliente(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "embeddings_gemini", None)
    monkeypatch.setitem(sys.modules, "langchain_google_genai", None)
    assert evaluar.main(["--casos", str(tmp_path / "no_existe.json")]) == 2
    assert "--confirmar-api-real" in capsys.readouterr().err
    assert evaluar.main([]) == 2
    assert list(tmp_path.iterdir()) == []


def test_02_sin_confirmar_no_lee_casos_ni_corpus(tmp_path, capsys):
    assert evaluar.main(["--casos", str(tmp_path / "x.json"), "--corpus", str(tmp_path / "y.json")]) == 2
    err = capsys.readouterr().err
    assert "no se pudieron leer" not in err and "Corpus" not in err


@pytest.mark.parametrize("malo", [["--k", "0"], ["--k", "6"], ["--k", "x"], ["--repeticiones", "0"], ["--repeticiones", "3"]])
def test_03_argumentos_fuera_de_rango_devuelven_2(tmp_path, embeddings, monkeypatch, malo):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    assert _codigo(_argv(tmp_path, directorio, *malo)) == 2
    assert embeddings.llamadas_consulta == 0 and not (tmp_path / "salida.json").exists()


def test_03b_la_abreviatura_conf_no_activa_la_confirmacion(tmp_path, embeddings, monkeypatch):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    assert _codigo(["--conf", "--directorio", str(directorio), "--salida", str(tmp_path / "salida.json")]) == 2
    assert embeddings.llamadas_consulta == 0 and not (tmp_path / "salida.json").exists()


def test_04_id_relevante_que_no_existe_en_el_corpus_no_gasta(tmp_path, monkeypatch, capsys, embeddings):
    casos = [{**CASOS[0], "ids_relevantes": ["parcheo_9"]}]
    directorio = _preparar(tmp_path, monkeypatch, embeddings, casos)
    assert evaluar.main(_argv(tmp_path, directorio)) == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "parcheo_9" in err
    assert embeddings.llamadas_consulta == 0 and not (tmp_path / "salida.json").exists()


def test_05_mas_casos_que_el_maximo_no_gasta(tmp_path, monkeypatch, capsys, embeddings):
    casos = [{**CASOS[0], "id": f"c{n}"} for n in range(evaluar.MAX_CASOS + 1)]
    directorio = _preparar(tmp_path, monkeypatch, embeddings, casos)
    assert evaluar.main(_argv(tmp_path, directorio)) == 1
    assert "Error:" in capsys.readouterr().err and embeddings.llamadas_consulta == 0


def test_05b_casos_por_repeticiones_sobre_el_tope_de_consultas_no_gasta(tmp_path, monkeypatch, capsys, embeddings):
    monkeypatch.setattr(evaluar, "MAX_CASOS", 30)
    monkeypatch.setattr(metricas_recuperacion, "MAX_CASOS", 30)
    casos = [{**CASOS[0], "id": f"c{n}"} for n in range(21)]  # 21 x 2 = 42 > 40
    directorio = _preparar(tmp_path, monkeypatch, embeddings, casos)
    assert evaluar.main(_argv(tmp_path, directorio)) == 1
    assert "consultas de embedding" in capsys.readouterr().err
    assert embeddings.llamadas_consulta == 0 and not (tmp_path / "salida.json").exists()
    # con 1 repetición son 21 consultas: pasa el tope
    assert evaluar.main(_argv(tmp_path, directorio, "--repeticiones", "1")) == 0
    assert embeddings.llamadas_consulta == 21


@pytest.mark.parametrize("estado", ["inexistente", "vacio", "conteo"])
def test_06_indice_inexistente_vacio_o_de_otro_corpus_no_gasta(tmp_path, monkeypatch, capsys, embeddings, estado):
    directorio = _preparar(tmp_path, monkeypatch, embeddings, indexar=False)
    if estado == "vacio":
        indexacion.abrir_vectorstore(embeddings, str(directorio), indexacion.COLECCION_POR_DEFECTO)
    elif estado == "conteo":
        indexacion.indexar(CORPUS[:2], embeddings, str(directorio))
    embeddings.llamadas_consulta = 0
    assert evaluar.main(_argv(tmp_path, directorio)) == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "reindexa" in err
    assert embeddings.llamadas_consulta == 0 and not (tmp_path / "salida.json").exists()
    assert directorio.exists() is (estado != "inexistente")


@pytest.mark.parametrize("caso", ["padre_inexistente", "igual_casos", "igual_corpus", "dentro_indice", "ajeno", "no_escribible"])
def test_07_salida_insegura_se_rechaza_antes_de_gastar(tmp_path, monkeypatch, capsys, embeddings, caso):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    ajeno = _escribir(tmp_path / "ajeno.json", {"a": 1})
    carpeta = tmp_path / "solo_lectura"
    carpeta.mkdir()
    destino = {
        "padre_inexistente": tmp_path / "no_hay" / "s.json",
        "igual_casos": tmp_path / "casos.json",
        "igual_corpus": tmp_path / "corpus.json",
        "dentro_indice": directorio / "s.json",
        "ajeno": ajeno,
        "no_escribible": carpeta / "s.json",
    }[caso]
    if caso == "no_escribible":
        if os.geteuid() == 0:
            pytest.skip("root ignora los permisos")
        carpeta.chmod(0o555)
    intactos = {ruta: ruta.read_text(encoding="utf-8") for ruta in (tmp_path / "casos.json", tmp_path / "corpus.json", ajeno)}
    try:
        assert evaluar.main(_argv(tmp_path, directorio, salida=destino)) == 1
    finally:
        carpeta.chmod(0o755)
    assert "Error:" in capsys.readouterr().err
    assert {ruta: ruta.read_text(encoding="utf-8") for ruta in intactos} == intactos
    assert embeddings.llamadas_consulta == 0 and not (directorio / "s.json").exists()


def test_07b_salida_por_enlace_simbolico_o_duro_a_casos_se_rechaza(tmp_path, monkeypatch, capsys, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    enlace, duro = tmp_path / "enlace.json", tmp_path / "duro.json"
    enlace.symlink_to(tmp_path / "casos.json")
    os.link(tmp_path / "casos.json", duro)
    original = (tmp_path / "casos.json").read_text(encoding="utf-8")
    for destino in (enlace, duro):
        assert evaluar.main(_argv(tmp_path, directorio, salida=destino)) == 1
        assert "--salida" in capsys.readouterr().err
    assert (tmp_path / "casos.json").read_text(encoding="utf-8") == original and embeddings.llamadas_consulta == 0


def test_08_flujo_feliz_con_promedios_solo_de_los_respondibles(tmp_path, monkeypatch, capsys, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    assert evaluar.main(_argv(tmp_path, directorio)) == 0
    assert embeddings.llamadas_consulta == len(CASOS) * 2
    salida = _leer(tmp_path)
    assert list(salida) == CLAVES_SALIDA
    assert salida["consultas_de_embeddings"] == 8 and salida["interrumpido"] is False
    assert salida["determinista"] is True and salida["max_dif_distancia"] == 0
    assert salida["configuracion"] == {
        "k": 3, "k_maximo": 5, "repeticiones": 2, "chunk_size": 200, "chunk_overlap": 30,
        "coleccion": "corpus_miniapp", "modelo_embeddings": "gemini-embedding-001", "casos": 4, "chunks_en_el_indice": 3,
    }
    promedios = salida["promedios_respondibles"]
    assert list(promedios) == ["1", "2", "3", "4", "5"]
    # k=1: c1 y c2 aciertan y c4 falla (3 respondibles; c3 no cuenta); k>=3: los 3 chunks, P=1/3 y R=1
    assert promedios["1"] == {"precision": pytest.approx(2 / 3), "exhaustividad": pytest.approx(2 / 3), "f1": pytest.approx(2 / 3)}
    assert promedios["3"] == {"precision": pytest.approx(1 / 3), "exhaustividad": 1.0, "f1": pytest.approx(0.5)}
    assert promedios["5"] == promedios["3"]
    assert [c["id"] for c in salida["por_caso"]] == [c["id"] for c in CASOS]
    assert salida["por_caso"][2]["metricas"] is None
    sin_respuesta = salida["sin_respuesta"]
    assert len(sin_respuesta) == 1 and sin_respuesta[0]["id"] == "c3_sin_respuesta"
    assert len(sin_respuesta[0]["ids_recuperados"]) == 3 and sin_respuesta[0]["ids_recuperados"][0] == "phishing_0"
    assert sin_respuesta[0]["distancia_minima"] == salida["por_caso"][2]["recuperados"][0]["distancia"]
    assert salida["por_caso"][0]["recuperados"][0]["id"] == "parcheo_0"
    assert list(tmp_path.glob("*.tmp")) == []
    out = capsys.readouterr().out
    assert "P=0.33 R=1.00 F1=0.50  c1_parcheo" in out and "sin respuesta: recupero [phishing_0" in out
    assert "Consultas de embeddings: 8" in out and "Determinismo: sí" in out


def _doc(id_documento="parcheo", fragmento=0, id_propio="otro_id"):
    return Document(id=id_propio, page_content="texto", metadata={"documento": id_documento, "fragmento": fragmento, "fuente": "x.md"})


def test_09_el_id_del_chunk_sale_de_la_metadata_y_no_de_document_id(tmp_path, monkeypatch, capsys, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings, [CASOS[0]])
    monkeypatch.setattr(recuperacion, "recuperar", lambda vectorstore, pregunta, k=3: [(_doc(), 0.1)])
    assert evaluar.main(_argv(tmp_path, directorio, "--repeticiones", "1")) == 0
    caso = _leer(tmp_path)["por_caso"][0]
    assert [r["id"] for r in caso["recuperados"]] == ["parcheo_0"]
    assert caso["metricas"]["1"]["precision"] == 1.0


@pytest.mark.parametrize("metadata", [{}, {"documento": "parcheo"}, {"documento": "parcheo", "fragmento": "0"}, {"documento": "", "fragmento": 0}])
def test_09b_metadata_incompleta_da_error_claro(tmp_path, monkeypatch, capsys, embeddings, metadata):
    directorio = _preparar(tmp_path, monkeypatch, embeddings, [CASOS[0]])
    monkeypatch.setattr(recuperacion, "recuperar", lambda vectorstore, pregunta, k=3: [(Document(id="x", page_content="t", metadata=metadata), 0.1)])
    assert evaluar.main(_argv(tmp_path, directorio)) == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "metadata" in err and "Traceback" not in err


def test_10_una_segunda_pasada_con_otro_orden_marca_determinista_false(tmp_path, monkeypatch, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings, CASOS[:2])
    original = recuperacion.recuperar
    llamadas = []

    def alterna(vectorstore, pregunta, k=3):
        pares = original(vectorstore, pregunta, k)
        llamadas.append(pregunta)
        if len(llamadas) > 2:  # segunda pasada: orden invertido y distancias corridas en 0.25
            pares = [(documento, distancia + 0.25) for documento, distancia in reversed(pares)]
        return pares

    monkeypatch.setattr(recuperacion, "recuperar", alterna)
    assert evaluar.main(_argv(tmp_path, directorio)) == 0
    salida = _leer(tmp_path)
    assert salida["determinista"] is False
    esperado = 0.0
    for caso in salida["por_caso"]:
        d = [r["distancia"] for r in caso["recuperados"]]
        esperado = max([esperado] + [abs(a - (b + 0.25)) for a, b in zip(d, reversed(d))])
    assert salida["max_dif_distancia"] == pytest.approx(esperado, abs=1e-5) and esperado > 0.2


def test_11_con_una_repeticion_el_determinismo_es_null(tmp_path, monkeypatch, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    assert evaluar.main(_argv(tmp_path, directorio, "--repeticiones", "1")) == 0
    salida = _leer(tmp_path)
    assert salida["determinista"] is None and salida["max_dif_distancia"] is None
    assert salida["consultas_de_embeddings"] == len(CASOS) == embeddings.llamadas_consulta


def test_12_ctrl_c_guarda_lo_hecho_y_devuelve_130_sin_traceback(tmp_path, monkeypatch, capsys, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    original, consultas = embeddings.embed_query, []

    def embed_query(texto):
        consultas.append(texto)
        if len(consultas) == 3:
            raise KeyboardInterrupt
        return original(texto)

    monkeypatch.setattr(embeddings, "embed_query", embed_query)
    assert evaluar.main(_argv(tmp_path, directorio)) == 130
    captura = capsys.readouterr()
    assert "Interrumpido por la persona" in captura.err and "Traceback" not in captura.err
    salida = _leer(tmp_path)
    assert salida["interrumpido"] is True and salida["determinista"] is None
    assert salida["incompleto"] is True and salida["error"] is None
    assert [c["id"] for c in salida["por_caso"]] == ["c1_parcheo", "c2_respaldos"]
    assert not (tmp_path / "salida.json.tmp").exists()


def test_12b_ctrl_c_durante_la_segunda_pasada_deja_el_determinismo_en_null(tmp_path, monkeypatch, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings, CASOS[:2])
    original, consultas = embeddings.embed_query, []

    def embed_query(texto):
        consultas.append(texto)
        if len(consultas) == 4:
            raise KeyboardInterrupt
        return original(texto)

    monkeypatch.setattr(embeddings, "embed_query", embed_query)
    assert evaluar.main(_argv(tmp_path, directorio)) == 130
    salida = _leer(tmp_path)
    assert salida["interrumpido"] is True and salida["determinista"] is None and len(salida["por_caso"]) == 2
    assert salida["incompleto"] is True and salida["error"] is None


def test_13_el_archivo_ya_existe_y_es_json_valido_tras_el_caso_2(tmp_path, monkeypatch, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    original, consultas, instantaneas = embeddings.embed_query, [], []

    def embed_query(texto):
        consultas.append(texto)
        if len(consultas) == 3:
            instantaneas.append(_leer(tmp_path))
        return original(texto)

    monkeypatch.setattr(embeddings, "embed_query", embed_query)
    assert evaluar.main(_argv(tmp_path, directorio)) == 0
    assert len(instantaneas) == 1
    assert [c["id"] for c in instantaneas[0]["por_caso"]] == ["c1_parcheo", "c2_respaldos"]
    assert instantaneas[0]["interrumpido"] is False and instantaneas[0]["consultas_de_embeddings"] == 2
    assert instantaneas[0]["incompleto"] is True
    assert _leer(tmp_path)["incompleto"] is False


@pytest.mark.parametrize(
    "error, esperado, esperas",
    [
        (RuntimeError("API key not valid"), "rechazó la clave", []),
        (RuntimeError("403 PERMISSION_DENIED"), "permiso", []),
        (RuntimeError("429 RESOURCE_EXHAUSTED"), "límite", [5, 10, 20]),
        (RuntimeError("fallo raro"), "No se pudo completar la operación con la API (RuntimeError): fallo raro", []),
    ],
    ids=["clave", "permiso", "limite_429", "desconocido"],
)
def test_14_errores_del_cliente_de_embeddings_dan_mensaje_claro(tmp_path, monkeypatch, capsys, embeddings, error, esperado, esperas):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    intentos, dormidas = [], []

    def falla(texto):
        intentos.append(texto)
        raise error

    monkeypatch.setattr(embeddings, "embed_query", falla)
    monkeypatch.setattr(indexacion.time, "sleep", dormidas.append)
    assert evaluar.main(_argv(tmp_path, directorio)) == 1
    err = capsys.readouterr().err
    assert "Error:" in err and esperado in err and "Traceback" not in err
    assert dormidas == esperas and len(intentos) == len(esperas) + 1


def test_14b_clave_faltante_al_crear_los_embeddings(tmp_path, monkeypatch, capsys, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)

    def sin_clave():
        raise ValueError("Falta GEMINI_API_KEY en .env.")

    monkeypatch.setattr(embeddings_gemini, "crear_embeddings", sin_clave)
    assert evaluar.main(_argv(tmp_path, directorio)) == 1
    err = capsys.readouterr().err
    assert "Error: Falta GEMINI_API_KEY" in err and "Traceback" not in err and not (tmp_path / "salida.json").exists()


def test_15_la_consola_limpia_los_controles_y_saltos_de_los_nombres_de_caso(tmp_path, monkeypatch, capsys, embeddings):
    casos = [
        {**CASOS[0], "id": "uno\x1b[31mrojo\nsegunda"},
        {**CASOS[2], "id": "dos\x1b[0m\nlinea"},
    ]
    directorio = _preparar(tmp_path, monkeypatch, embeddings, casos)
    assert evaluar.main(_argv(tmp_path, directorio)) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out
    assert "  uno [31mrojo segunda\n" in out and "  dos [0m linea\n" in out


TEXTO_LIMITE = embeddings_gemini.describir_error_api(RuntimeError("429 RESOURCE_EXHAUSTED"))[1]


def _falla_desde(monkeypatch, embeddings, desde: int, error: Exception) -> tuple[list, list]:
    """Hace que embed_query falle desde la petición número `desde` (1 = la primera) y anula las esperas; devuelve (intentos, dormidas)."""
    original, intentos, dormidas = embeddings.embed_query, [], []

    def embed_query(texto):
        intentos.append(texto)
        if len(intentos) >= desde:
            raise error
        return original(texto)

    monkeypatch.setattr(embeddings, "embed_query", embed_query)
    monkeypatch.setattr(indexacion.time, "sleep", dormidas.append)
    return intentos, dormidas


@pytest.mark.parametrize("repeticiones", ["1", "2"])
def test_08b_una_corrida_completa_no_es_incompleta_y_no_tiene_error(tmp_path, monkeypatch, embeddings, repeticiones):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    assert evaluar.main(_argv(tmp_path, directorio, "--repeticiones", repeticiones)) == 0
    salida = _leer(tmp_path)
    assert list(salida) == CLAVES_SALIDA
    assert salida["incompleto"] is False and salida["error"] is None


def test_17_un_429_persistente_en_la_primera_pasada_guarda_lo_hecho_como_incompleto(tmp_path, monkeypatch, capsys, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    _falla_desde(monkeypatch, embeddings, 3, RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert evaluar.main(_argv(tmp_path, directorio, "--repeticiones", "1")) == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "Resultados parciales" in err
    salida = _leer(tmp_path)
    assert [c["id"] for c in salida["por_caso"]] == ["c1_parcheo", "c2_respaldos"]
    assert salida["incompleto"] is True and salida["error"] == TEXTO_LIMITE and TEXTO_LIMITE in err
    assert salida["interrumpido"] is False and salida["determinista"] is None
    assert salida["consultas_de_embeddings"] == 6
    assert salida["promedios_respondibles"]["1"]["precision"] == 1.0


def test_17b_un_429_persistente_en_la_segunda_pasada_cuenta_todos_los_intentos(tmp_path, monkeypatch, capsys, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    _falla_desde(monkeypatch, embeddings, 6, RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert evaluar.main(_argv(tmp_path, directorio)) == 1
    assert "Resultados parciales" in capsys.readouterr().err
    salida = _leer(tmp_path)
    assert len(salida["por_caso"]) == 4 and salida["incompleto"] is True and salida["error"] == TEXTO_LIMITE
    assert salida["determinista"] is None and salida["max_dif_distancia"] is None
    assert salida["consultas_de_embeddings"] == 9


def test_17c_un_value_error_a_mitad_de_corrida_guarda_lo_hecho(tmp_path, monkeypatch, capsys, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings, CASOS[:2])
    original, llamadas = recuperacion.recuperar, []

    def recuperar(vectorstore, pregunta, k=3):
        llamadas.append(pregunta)
        if len(llamadas) == 2:
            return [(Document(id="x", page_content="t", metadata={}), 0.1)]
        return original(vectorstore, pregunta, k)

    monkeypatch.setattr(recuperacion, "recuperar", recuperar)
    assert evaluar.main(_argv(tmp_path, directorio)) == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "Resultados parciales" in err
    salida = _leer(tmp_path)
    assert len(salida["por_caso"]) == 1 and salida["incompleto"] is True and "metadata" in salida["error"]


def test_17d_un_error_en_el_primer_caso_no_crea_el_archivo(tmp_path, monkeypatch, capsys, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    _falla_desde(monkeypatch, embeddings, 1, RuntimeError("API key not valid"))
    assert evaluar.main(_argv(tmp_path, directorio)) == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "Resultados parciales" not in err
    assert not (tmp_path / "salida.json").exists()


def test_17e_un_error_en_el_primer_caso_no_pisa_un_resultado_previo(tmp_path, monkeypatch, capsys, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    assert evaluar.main(_argv(tmp_path, directorio, "--repeticiones", "1")) == 0
    previo = (tmp_path / "salida.json").read_bytes()
    _falla_desde(monkeypatch, embeddings, 1, RuntimeError("fallo raro"))
    capsys.readouterr()
    assert evaluar.main(_argv(tmp_path, directorio)) == 1
    assert "Resultados parciales" not in capsys.readouterr().err
    assert (tmp_path / "salida.json").read_bytes() == previo


def test_18_el_tope_de_consultas_se_fuerza_en_ejecucion_sin_pasarse_ni_una_peticion(tmp_path, monkeypatch, capsys, embeddings):
    monkeypatch.setattr(evaluar, "MAX_CONSULTAS_EMBEDDING", 6)
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    original, peticiones, dormidas, por_pregunta = embeddings.embed_query, [], [], {}

    def embed_query(texto):
        peticiones.append(texto)
        por_pregunta[texto] = por_pregunta.get(texto, 0) + 1
        if por_pregunta[texto] <= 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return original(texto)

    monkeypatch.setattr(embeddings, "embed_query", embed_query)
    monkeypatch.setattr(indexacion.time, "sleep", dormidas.append)
    assert evaluar.main(_argv(tmp_path, directorio, "--repeticiones", "1")) == 1
    assert "Error:" in (err := capsys.readouterr().err) and "tope" in err and "Resultados parciales" in err
    assert len(peticiones) == 6 and len(dormidas) == 4
    salida = _leer(tmp_path)
    assert len(salida["por_caso"]) == 2 and salida["incompleto"] is True
    assert salida["consultas_de_embeddings"] == 6 and "tope" in salida["error"]


def test_18b_el_mensaje_del_tope_no_se_confunde_con_un_limite_de_la_api():
    corrida = {"consultas": evaluar.MAX_CONSULTAS_EMBEDDING}
    with pytest.raises(evaluar.TopeDeConsultas) as info:
        evaluar._consultar(None, "pregunta", corrida)
    mensaje = str(info.value)
    assert "429" not in mensaje and "RESOURCE_EXHAUSTED" not in mensaje and "tope" in mensaje
    assert corrida["consultas"] == evaluar.MAX_CONSULTAS_EMBEDDING


def test_19_la_consola_respeta_k_y_coincide_con_el_json(tmp_path, monkeypatch, capsys, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    lineas = {}
    for k in (1, 3, 5):
        assert evaluar.main(_argv(tmp_path, directorio, "--k", str(k), "--repeticiones", "1")) == 0
        out = capsys.readouterr().out
        lineas[k] = [linea for linea in out.splitlines() if linea.startswith("P=")]
        esperadas = [
            f"P={c['metricas'][str(k)]['precision']:.2f} R={c['metricas'][str(k)]['exhaustividad']:.2f} "
            f"F1={c['metricas'][str(k)]['f1']:.2f}  {c['id']}"
            for c in _leer(tmp_path)["por_caso"] if c["respondible"]
        ]
        assert lineas[k] == esperadas and len(esperadas) == 3
    assert lineas[1] != lineas[3]


def test_20_la_configuracion_guardada_refleja_el_k_pedido(tmp_path, monkeypatch, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    for k in (1, 5):
        assert evaluar.main(_argv(tmp_path, directorio, "--k", str(k), "--repeticiones", "1")) == 0
        assert _leer(tmp_path)["configuracion"]["k"] == k


def test_21_coleccion_se_respeta(tmp_path, monkeypatch, capsys, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings, indexar=False)
    indexacion.indexar(CORPUS, embeddings, str(directorio), coleccion="otra_coleccion")
    embeddings.llamadas_consulta = 0
    assert evaluar.main(_argv(tmp_path, directorio, "--coleccion", "otra_coleccion", "--repeticiones", "1")) == 0
    assert _leer(tmp_path)["configuracion"]["coleccion"] == "otra_coleccion"
    (tmp_path / "salida.json").unlink()
    embeddings.llamadas_consulta = 0
    capsys.readouterr()
    assert evaluar.main(_argv(tmp_path, directorio)) == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "reindexa" in err
    assert embeddings.llamadas_consulta == 0 and not (tmp_path / "salida.json").exists()


def test_22_la_fecha_guardada_es_la_de_hoy(tmp_path, monkeypatch, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    assert evaluar.main(_argv(tmp_path, directorio, "--repeticiones", "1")) == 0
    hoy = datetime.date.today()
    assert _leer(tmp_path)["fecha"] in {hoy.isoformat(), (hoy - datetime.timedelta(days=1)).isoformat()}


def test_23_las_distancias_del_archivo_van_redondeadas_a_6_decimales(tmp_path, monkeypatch, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings, CASOS[:3])
    original, llamadas = recuperacion.recuperar, []

    def recuperar(vectorstore, pregunta, k=3):
        llamadas.append(pregunta)
        desfase = 0.1234567891 if len(llamadas) > 3 else 0.0  # la segunda pasada se corre para que max_dif_distancia tenga decimales
        return [(documento, 0.123456789 + n / 7 + desfase) for n, (documento, _) in enumerate(original(vectorstore, pregunta, k))]

    monkeypatch.setattr(recuperacion, "recuperar", recuperar)
    assert evaluar.main(_argv(tmp_path, directorio)) == 0
    salida = _leer(tmp_path)
    distancias = [r["distancia"] for c in salida["por_caso"] for r in c["recuperados"]]
    distancias += [s["distancia_minima"] for s in salida["sin_respuesta"]] + [salida["max_dif_distancia"]]
    assert len(distancias) == 9 + 1 + 1 and salida["max_dif_distancia"] > 0.1
    assert all(round(d, 6) == d for d in distancias) and salida["por_caso"][0]["recuperados"][0]["distancia"] == 0.123457


def test_24_guardar_es_atomico_si_falla_el_reemplazo(tmp_path, monkeypatch, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    assert evaluar.main(_argv(tmp_path, directorio, "--repeticiones", "1")) == 0
    ruta = tmp_path / "salida.json"
    previo = ruta.read_bytes()
    documento = {**json.loads(previo), "consultas_de_embeddings": 999}

    def falla(origen, destino):
        raise OSError("disco lleno")

    monkeypatch.setattr(os, "replace", falla)
    with pytest.raises(OSError):
        evaluar._guardar(documento, str(ruta))
    assert ruta.read_bytes() == previo and list(tmp_path.glob("*.tmp")) == []


def test_25_salida_por_un_enlace_simbolico_dentro_del_indice_se_rechaza(tmp_path, monkeypatch, capsys, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    enlace = tmp_path / "enlace"
    enlace.symlink_to(directorio, target_is_directory=True)
    antes = sorted(directorio.rglob("*"))
    assert evaluar.main(_argv(tmp_path, directorio, salida=enlace / "res.json")) == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "dentro del directorio del índice" in err
    assert embeddings.llamadas_consulta == 0 and sorted(directorio.rglob("*")) == antes and not (directorio / "res.json").exists()


def test_26_salida_vacia_se_rechaza(tmp_path, monkeypatch, capsys, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    argv = _argv(tmp_path, directorio)
    argv[argv.index("--salida") + 1] = ""
    assert evaluar.main(argv) == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "no puede estar vacío" in err and embeddings.llamadas_consulta == 0


def test_27_se_recuperan_k_maximo_chunks_aunque_el_indice_sea_chico(tmp_path, monkeypatch, embeddings):
    relleno = [
        {"id": f"d{n}", "fuente": f"d{n}.md", "texto": f"Documento de relleno numero {n} sin temas."} for n in range(1, 8)
    ]
    corpus = [{"id": "d0", "fuente": "d0.md", "texto": "Plazo de parche: vulnerabilidad con cvss alto."}, *relleno]
    _escribir(tmp_path / "corpus.json", corpus)
    _escribir(tmp_path / "casos.json", [CASOS[0] | {"ids_relevantes": ["d0_0"]}])
    indexacion.indexar(corpus, embeddings, str(tmp_path / "indice"))
    monkeypatch.setattr(embeddings_gemini, "crear_embeddings", lambda: embeddings)
    assert evaluar.main(_argv(tmp_path, tmp_path / "indice", "--repeticiones", "1")) == 0
    caso = _leer(tmp_path)["por_caso"][0]
    assert [r["id"] for r in caso["recuperados"]][0] == "d0_0" and len(caso["recuperados"]) == metricas_recuperacion.K_MAXIMO == 5
    assert caso["metricas"]["5"]["precision"] == pytest.approx(1 / 5) and caso["metricas"]["5"]["exhaustividad"] == 1.0


def test_28_una_pregunta_demasiado_larga_se_rechaza_antes_de_gastar(tmp_path, monkeypatch, capsys, embeddings):
    maximo = recuperacion.MAX_LONGITUD_PREGUNTA
    directorio = _preparar(tmp_path, monkeypatch, embeddings, [{**CASOS[0], "pregunta": "a" * (maximo + 1)}])
    assert evaluar.main(_argv(tmp_path, directorio)) == 1
    assert "Error:" in capsys.readouterr().err
    assert embeddings.llamadas_consulta == 0 and not (tmp_path / "salida.json").exists()
    _escribir(tmp_path / "casos.json", [{**CASOS[0], "pregunta": "a" * maximo}])
    assert evaluar.main(_argv(tmp_path, directorio, "--repeticiones", "1")) == 0


def test_29_un_indice_con_el_mismo_conteo_pero_otros_ids_se_rechaza(tmp_path, monkeypatch, capsys, embeddings):
    directorio = _preparar(tmp_path, monkeypatch, embeddings, indexar=False)
    indexacion.indexar([{**d, "id": f"{d['id']}2"} for d in CORPUS], embeddings, str(directorio))
    embeddings.llamadas_consulta = 0
    assert evaluar.main(_argv(tmp_path, directorio)) == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "reindexa" in err
    assert embeddings.llamadas_consulta == 0 and not (tmp_path / "salida.json").exists()


def test_16_el_script_no_importa_nada_de_claude(tmp_path, monkeypatch, embeddings):
    for nombre in ("api_claude_refactor", "langchain_anthropic", "modelo_langchain"):
        monkeypatch.delitem(sys.modules, nombre, raising=False)
    directorio = _preparar(tmp_path, monkeypatch, embeddings)
    assert evaluar.main(_argv(tmp_path, directorio)) == 0
    assert not {"api_claude_refactor", "langchain_anthropic", "modelo_langchain"} & set(sys.modules)
