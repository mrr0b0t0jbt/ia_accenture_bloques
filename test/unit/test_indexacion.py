import fcntl
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import chromadb.errors
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

import embeddings_gemini
import indexacion

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_CORPUS = RAIZ / "JSON" / "corpus_miniapp.json"
IDS_CORPUS = [
    "politica_parcheo",
    "aviso_cve_2099_10001",
    "aviso_cve_2099_10002",
    "playbook_trafico_saliente",
    "playbook_phishing",
    "politica_contrasenas_mfa",
    "guia_respaldos",
    "clasificacion_incidentes",
    "agente_aprobacion_humana",
    "agente_limites_trazas",
]


@pytest.fixture
def corpus():
    return indexacion.cargar_corpus(str(RUTA_CORPUS))


def _embeddings(size: int = 16):
    return DeterministicFakeEmbedding(size=size)


def _ids_del_indice(vectorstore) -> list[str]:
    return sorted(vectorstore.get()["ids"])


def test_01_el_corpus_real_se_carga(corpus):
    assert [d["id"] for d in corpus] == IDS_CORPUS
    assert all(300 <= len(d["texto"]) <= 700 for d in corpus)
    assert sum(len(d["texto"]) for d in corpus) == 4375


def test_02_cargar_corpus_rechaza_corpus_invalidos(tmp_path):
    def doc(**cambios):
        base = {"id": "a", "fuente": "a.md", "texto": "hola"}
        base.update(cambios)
        return base

    sin_texto = doc()
    del sin_texto["texto"]
    casos = {
        "inexistente": None,
        "json roto": "{[",
        "objeto": json.dumps({"id": "a"}),
        "lista vacía": "[]",
        "no objeto": json.dumps(["a"]),
        "sin texto": json.dumps([sin_texto]),
        "id repetido": json.dumps([doc(), doc()]),
        "id con espacio": json.dumps([doc(id="a b")]),
        "id con tilde": json.dumps([doc(id="añ")]),
        "id con barra": json.dumps([doc(id="a/b")]),
        "texto en blanco": json.dumps([doc(texto="   ")]),
        "fuente vacía": json.dumps([doc(fuente="")]),
        "51 documentos": json.dumps([doc(id=f"d{n}") for n in range(51)]),
    }
    for nombre, contenido in casos.items():
        ruta = tmp_path / "corpus.json"
        ruta.unlink(missing_ok=True)
        if contenido is not None:
            ruta.write_text(contenido, encoding="utf-8")
        with pytest.raises(ValueError, match="Corpus:"):
            indexacion.cargar_corpus(str(ruta))
        assert nombre


def test_03_dividir_corpus_real_da_29_chunks_deterministas(corpus):
    chunks, ids = indexacion.dividir_corpus(corpus, 200, 30)
    assert len(chunks) == len(ids) == 29
    assert ids == indexacion.dividir_corpus(corpus, 200, 30)[1]
    assert len(set(ids)) == 29
    assert ids[0] == "politica_parcheo_0"
    for chunk, id_chunk in zip(chunks, ids):
        assert chunk.metadata["fuente"].endswith(".md")
        assert id_chunk == f"{chunk.metadata['documento']}_{chunk.metadata['fragmento']}"


def test_04_un_documento_largo_se_parte_en_fragmentos_numerados():
    documentos = [{"id": "largo", "fuente": "largo.md", "texto": "palabra " * 75}]
    chunks, ids = indexacion.dividir_corpus(documentos, 100, 10)
    assert len(chunks) > 3
    assert [c.metadata["fragmento"] for c in chunks] == list(range(len(chunks)))
    assert {c.metadata["fuente"] for c in chunks} == {"largo.md"}
    assert ids == [f"largo_{n}" for n in range(len(chunks))]


def test_05_dividir_corpus_rechaza_parametros_invalidos(corpus):
    for tamano, solape in [(49, 10), (1001, 10), (200, -1), (200, 101), ("200", 10), (True, 10), (200, 10.5)]:
        with pytest.raises(ValueError):
            indexacion.dividir_corpus(corpus, tamano, solape)


def test_06_integridad_del_corpus_real(corpus):
    chunks, _ = indexacion.dividir_corpus(corpus, 200, 30)
    informe = indexacion.verificar_integridad(corpus, chunks)
    assert informe["ok"] is True
    assert informe["chunks"] == 29
    assert informe["vacios"] == [] and informe["sin_fuente"] == []
    assert informe["duplicados"] is False
    assert informe["caracteres_corpus"] == 4375
    assert informe["caracteres_chunks"] == 4865


def test_07_integridad_detecta_vacio_repetido_y_sin_fuente():
    chunks = [
        Document(page_content="uno", metadata={"fuente": "a.md"}),
        Document(page_content="  ", metadata={"fuente": "a.md"}),
        Document(page_content="uno", metadata={"fuente": "a.md"}),
        Document(page_content="dos", metadata={}),
    ]
    informe = indexacion.verificar_integridad([{"texto": "uno dos"}], chunks)
    assert informe["ok"] is False
    assert informe["vacios"] == [1]
    assert informe["duplicados"] is True
    assert informe["sin_fuente"] == [3]


def _operacion_que_falla(veces: int, mensaje: str):
    llamadas = []

    def operacion():
        llamadas.append(1)
        if len(llamadas) <= veces:
            raise RuntimeError(mensaje)
        return "listo"

    return operacion, llamadas


def test_08_con_reintentos_reintenta_ante_429_y_resource_exhausted():
    for mensaje in ("429 Too Many Requests", "RESOURCE_EXHAUSTED: cuota"):
        operacion, llamadas = _operacion_que_falla(2, mensaje)
        esperas = []
        assert indexacion.con_reintentos(operacion, dormir=esperas.append) == "listo"
        assert esperas == [5, 10] and len(llamadas) == 3


def test_09_con_reintentos_agota_los_intentos_y_relanza_el_ultimo():
    errores = iter(f"429 intento {n}" for n in range(1, 5))

    def operacion():
        raise RuntimeError(next(errores))

    esperas = []
    with pytest.raises(RuntimeError, match="intento 4"):
        indexacion.con_reintentos(operacion, dormir=esperas.append)
    assert esperas == [5, 10, 20]


def test_10_con_reintentos_relanza_de_inmediato_otros_errores():
    llamadas, esperas = [], []

    def operacion():
        llamadas.append(1)
        raise ValueError("API key not valid")

    with pytest.raises(ValueError, match="API key"):
        indexacion.con_reintentos(operacion, dormir=esperas.append)
    assert len(llamadas) == 1 and esperas == []


def test_11_crear_embeddings_exige_clave_y_no_abre_conexiones(monkeypatch):
    monkeypatch.setattr(embeddings_gemini, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValueError, match=r"^Falta GEMINI_API_KEY en \.env\.$"):
        embeddings_gemini.crear_embeddings()

    def sin_red(*a, **k):
        raise AssertionError("no debe abrir conexiones")

    monkeypatch.setattr(socket.socket, "connect", sin_red)
    monkeypatch.setenv("GEMINI_API_KEY", "clave-falsa")
    assert embeddings_gemini.crear_embeddings().model == "models/gemini-embedding-001"


def test_12_indexar_el_corpus_real(corpus, tmp_path):
    directorio = tmp_path / "indice"
    informe = indexacion.indexar(corpus, _embeddings(), str(directorio))
    assert informe == {
        "documentos": 10,
        "chunks": 29,
        "caracteres_corpus": 4375,
        "caracteres_chunks": 4865,
        "directorio": os.path.abspath(directorio),
        "coleccion": "corpus_miniapp",
        "total_en_indice": 29,
        "eliminados": 0,
    }
    reabierto = indexacion.cargar_vectorstore(_embeddings(), str(directorio))
    assert reabierto._collection.count() == 29
    resultados = reabierto.similarity_search("plazo para vulnerabilidades críticas", k=3)
    assert resultados and all(isinstance(r, Document) and r.metadata["fuente"] for r in resultados)


def test_13_reindexar_el_mismo_corpus_es_idempotente(corpus, tmp_path):
    directorio = str(tmp_path / "indice")
    indexacion.indexar(corpus, _embeddings(), directorio)
    antes = _ids_del_indice(indexacion.cargar_vectorstore(_embeddings(), directorio))
    informe = indexacion.indexar(corpus, _embeddings(), directorio)
    assert informe["total_en_indice"] == 29 and informe["eliminados"] == 0
    assert _ids_del_indice(indexacion.cargar_vectorstore(_embeddings(), directorio)) == antes


def test_14_reindexar_con_un_documento_editado_y_otro_quitado(corpus, tmp_path):
    directorio = str(tmp_path / "indice")
    indexacion.indexar(corpus, _embeddings(), directorio)
    quitado = "playbook_phishing"
    ids_antes = indexacion.dividir_corpus(corpus, 200, 30)[1]
    nuevo = [dict(d) for d in corpus if d["id"] != quitado]
    for d in nuevo:
        if d["id"] == "guia_respaldos":
            d["texto"] = "Texto nuevo y corto de respaldos."
    nuevos_chunks, nuevos_ids = indexacion.dividir_corpus(nuevo, 200, 30)
    informe = indexacion.indexar(nuevo, _embeddings(), directorio)
    indice = indexacion.cargar_vectorstore(_embeddings(), directorio)
    contenido = indice.get()
    assert not [i for i in contenido["ids"] if i.startswith(quitado)]
    # los 3 chunks del quitado y el segundo fragmento de guia_respaldos, que el texto nuevo ya no produce
    assert informe["eliminados"] == len(set(ids_antes) - set(nuevos_ids)) == 4
    assert informe["total_en_indice"] == len(nuevos_ids)
    assert _ids_del_indice(indice) == sorted(nuevos_ids)
    textos = dict(zip(contenido["ids"], contenido["documents"]))
    assert textos["guia_respaldos_0"] == "Texto nuevo y corto de respaldos."


def test_15_reindexar_con_un_documento_nuevo_solo_agrega_sus_ids(corpus, tmp_path):
    directorio = str(tmp_path / "indice")
    indexacion.indexar(corpus, _embeddings(), directorio)
    antes = indexacion.cargar_vectorstore(_embeddings(), directorio).get()
    textos_antes = dict(zip(antes["ids"], antes["documents"]))
    nuevo = corpus + [{"id": "extra", "fuente": "extra.md", "texto": "Documento extra de prueba."}]
    informe = indexacion.indexar(nuevo, _embeddings(), directorio)
    despues = indexacion.cargar_vectorstore(_embeddings(), directorio).get()
    textos_despues = dict(zip(despues["ids"], despues["documents"]))
    assert informe["total_en_indice"] == 30 and informe["eliminados"] == 0
    assert set(textos_despues) - set(textos_antes) == {"extra_0"}
    assert all(textos_despues[i] == t for i, t in textos_antes.items())


def test_16_el_indice_persiste_entre_procesos(corpus, tmp_path):
    directorio = tmp_path / "indice"
    indexacion.indexar(corpus, _embeddings(), str(directorio))
    codigo = (
        "import sys\n"
        "from langchain_core.embeddings import DeterministicFakeEmbedding\n"
        "import indexacion\n"
        "v = indexacion.cargar_vectorstore(DeterministicFakeEmbedding(size=16), sys.argv[1])\n"
        "print(v._collection.count())\n"
    )
    entorno = {**os.environ, "PYTHONPATH": str(RAIZ / "src")}
    resultado = subprocess.run(
        [sys.executable, "-c", codigo, str(directorio)], capture_output=True, text=True, env=entorno, timeout=60
    )
    assert resultado.returncode == 0, resultado.stderr
    assert resultado.stdout.strip() == "29"


def test_17_cargar_vectorstore_sin_directorio_no_lo_crea(tmp_path):
    directorio = tmp_path / "no_existe"
    with pytest.raises(FileNotFoundError, match="indexar_corpus.py"):
        indexacion.cargar_vectorstore(_embeddings(), str(directorio))
    assert not directorio.exists()


def test_18_cargar_vectorstore_con_coleccion_vacia(tmp_path):
    directorio = tmp_path / "vacio"
    indexacion.abrir_vectorstore(_embeddings(), str(directorio), "corpus_miniapp")
    with pytest.raises(ValueError, match="vacío"):
        indexacion.cargar_vectorstore(_embeddings(), str(directorio))


def test_19_la_ruta_relativa_se_vuelve_absoluta(corpus, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    informe = indexacion.indexar(corpus, _embeddings(), "indice_rel")
    assert informe["directorio"] == os.path.realpath(tmp_path / "indice_rel")
    otro = tmp_path / "otro"
    otro.mkdir()
    monkeypatch.chdir(otro)
    reabierto = indexacion.cargar_vectorstore(_embeddings(), informe["directorio"])
    assert reabierto._collection.count() == 29


def test_20_otra_dimension_de_embeddings_falla_sin_tocar_el_indice(corpus, tmp_path):
    directorio = str(tmp_path / "indice")
    indexacion.indexar(corpus, _embeddings(16), directorio)
    with pytest.raises(chromadb.errors.InvalidArgumentError, match="dimension"):
        indexacion.indexar(corpus, _embeddings(32), directorio)
    assert indexacion.cargar_vectorstore(_embeddings(16), directorio)._collection.count() == 29


def test_21_nombres_de_coleccion_invalidos(corpus, tmp_path):
    for nombre in ["ab", "con espacios", "áéí", "a" * 513, "_x_", "x_"]:
        with pytest.raises(ValueError):
            indexacion.validar_nombre_coleccion(nombre)
        directorio = tmp_path / "indice"
        with pytest.raises(ValueError):
            indexacion.indexar(corpus, _embeddings(), str(directorio), coleccion=nombre)
        assert not directorio.exists()
    assert indexacion.validar_nombre_coleccion("ok-name_1") == "ok-name_1"


def test_22_un_corpus_con_demasiados_chunks_falla_antes_de_los_embeddings(tmp_path):
    class EmbeddingsContador(Embeddings):
        llamadas = 0

        def embed_documents(self, textos):
            self.llamadas += 1
            return [[0.0] * 4 for _ in textos]

        def embed_query(self, texto):
            self.llamadas += 1
            return [0.0] * 4

    documentos = [{"id": f"doc_{n}", "fuente": f"doc_{n}.md", "texto": "palabra " * 75} for n in range(40)]
    embeddings = EmbeddingsContador()
    directorio = tmp_path / "indice"
    with pytest.raises(ValueError, match="chunks"):
        indexacion.indexar(documentos, embeddings, str(directorio), chunk_size=50, chunk_overlap=0)
    assert embeddings.llamadas == 0 and not directorio.exists()


def test_23_si_la_segunda_indexacion_falla_el_indice_anterior_queda_intacto(corpus, tmp_path):
    class EmbeddingsQueFallanDespues(Embeddings):
        falla = False

        def embed_documents(self, textos):
            if self.falla:
                raise RuntimeError("fallo de la API")
            return [[float(len(texto)), 1.0] for texto in textos]

        def embed_query(self, texto):
            return [float(len(texto)), 1.0]

    embeddings = EmbeddingsQueFallanDespues()
    directorio = str(tmp_path / "indice")
    indexacion.indexar(corpus, embeddings, directorio)
    antes = indexacion.cargar_vectorstore(embeddings, directorio).get()
    embeddings.falla = True
    # un documento menos: borrar los huérfanos antes de agregar alteraría el índice
    with pytest.raises(RuntimeError, match="fallo de la API"):
        indexacion.indexar(corpus[:-1], embeddings, directorio)
    despues = indexacion.cargar_vectorstore(embeddings, directorio).get()
    assert len(despues["ids"]) == len(antes["ids"]) == 29
    assert sorted(despues["ids"]) == sorted(antes["ids"])
    assert any(i.startswith("agente_limites_trazas") for i in despues["ids"])


def test_24_describir_error_api_clasifica_limite_clave_y_otro():
    describir = embeddings_gemini.describir_error_api
    for texto in ("Error embedding content (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {}", "429", "RESOURCE_EXHAUSTED"):
        categoria, mensaje = describir(RuntimeError(texto))
        assert categoria == "limite"
        assert mensaje == "Se agotó el límite de peticiones de la API: espera un minuto y vuelve a ejecutar."
    for texto in ("400 INVALID_ARGUMENT. API key not valid. Please pass a valid API key.", "PERMISSION_DENIED", "UNAUTHENTICATED", "401", "403"):
        categoria, mensaje = describir(RuntimeError(texto))
        assert categoria == "clave"
        assert mensaje == "La API rechazó la clave o no tiene permiso para usar embeddings: revisa GEMINI_API_KEY en .env."
    categoria, mensaje = describir(RuntimeError("x" * 400))
    assert categoria == "otro"
    assert mensaje == f"No se pudo completar la operación con la API (RuntimeError): {'x' * 300}"
    # el límite tiene prioridad sobre la clave
    assert describir(RuntimeError("429 y además API key not valid"))[0] == "limite"


class _EmbeddingsContador(Embeddings):
    def __init__(self):
        self.llamadas = 0

    def embed_documents(self, textos):
        self.llamadas += 1
        return [[float(len(texto)), 1.0] for texto in textos]

    def embed_query(self, texto):
        self.llamadas += 1
        return [float(len(texto)), 1.0]


class _EmbeddingsQueFallan(_EmbeddingsContador):
    def embed_documents(self, textos):
        raise RuntimeError("fallo de la API")


def _documentos_distintos(cantidad: int, palabras: int = 60) -> list[dict]:
    return [
        {"id": f"d{n}", "fuente": f"d{n}.md", "texto": " ".join(f"w{n}x{i}" for i in range(palabras))}
        for n in range(cantidad)
    ]


def _tomar_bloqueo(directorio) -> int:
    """Toma el bloqueo del índice con un descriptor propio de la prueba; devuelve el descriptor."""
    descriptor = os.open(f"{os.path.abspath(directorio)}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return descriptor


def _esta_libre(directorio) -> bool:
    descriptor = os.open(f"{os.path.abspath(directorio)}.lock", os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(descriptor)


def test_25_con_el_bloqueo_tomado_indexar_falla_sin_tocar_el_indice(corpus, tmp_path):
    directorio = tmp_path / "indice"
    indexacion.indexar(corpus, _embeddings(), str(directorio))
    antes = _ids_del_indice(indexacion.cargar_vectorstore(_embeddings(), str(directorio)))
    descriptor = _tomar_bloqueo(directorio)
    try:
        with pytest.raises(ValueError, match="otra indexacion"):
            indexacion.indexar(corpus[:3], _embeddings(), str(directorio))
    finally:
        os.close(descriptor)
    assert _ids_del_indice(indexacion.cargar_vectorstore(_embeddings(), str(directorio))) == antes


def test_26_el_bloqueo_se_libera_tras_indexar_y_tras_un_fallo(corpus, tmp_path):
    directorio = tmp_path / "indice"
    indexacion.indexar(corpus, _embeddings(), str(directorio))
    assert _esta_libre(directorio)
    with pytest.raises(RuntimeError, match="fallo de la API"):
        indexacion.indexar(corpus[:2], _EmbeddingsQueFallan(), str(directorio))
    assert _esta_libre(directorio)
    assert indexacion.indexar(corpus, _embeddings(), str(directorio))["total_en_indice"] == 29


def test_27_el_archivo_de_bloqueo_queda_al_lado_del_indice_con_permisos_600(corpus, tmp_path):
    directorio = tmp_path / "indice"
    indexacion.indexar(corpus, _embeddings(), str(directorio))
    bloqueo = tmp_path / "indice.lock"
    assert bloqueo.is_file() and (bloqueo.stat().st_mode & 0o777) == 0o600
    assert not [p for p in directorio.rglob("*") if p.suffix == ".lock"]


def test_28_cargar_vectorstore_no_toma_el_bloqueo(corpus, tmp_path):
    directorio = tmp_path / "indice"
    indexacion.indexar(corpus, _embeddings(), str(directorio))
    descriptor = _tomar_bloqueo(directorio)
    try:
        assert indexacion.cargar_vectorstore(_embeddings(), str(directorio))._collection.count() == 29
    finally:
        os.close(descriptor)


SCRIPT_INDEXADOR = """
import os, sys, time
from langchain_core.embeddings import Embeddings
import indexacion

directorio, nombre, marca, liberar = sys.argv[1:5]


class Espera(Embeddings):
    def embed_documents(self, textos):
        open(marca, "w").close()
        limite = time.time() + 20
        while not os.path.exists(liberar) and time.time() < limite:
            time.sleep(0.02)
        return [[float(len(t)), 1.0] for t in textos]

    def embed_query(self, texto):
        return [1.0, 1.0]


documentos = [{"id": nombre, "fuente": nombre + ".md", "texto": "texto de " + nombre}]
try:
    indexacion.indexar(documentos, Espera(), directorio)
except ValueError as error:
    print(error, file=sys.stderr)
    sys.exit(1)
"""


def _esperar(ruta, segundos=30):
    limite = time.time() + segundos
    while not ruta.exists() and time.time() < limite:
        time.sleep(0.02)
    assert ruta.exists(), f"no apareció {ruta}"


def test_29_dos_procesos_sobre_el_mismo_indice_no_lo_dejan_vacio(tmp_path):
    script = tmp_path / "indexador.py"
    script.write_text(SCRIPT_INDEXADOR, encoding="utf-8")
    directorio, liberar = tmp_path / "indice", tmp_path / "liberar"
    entorno = {**os.environ, "PYTHONPATH": str(RAIZ / "src")}

    def lanzar(nombre):
        argv = [sys.executable, str(script), str(directorio), nombre, str(tmp_path / f"{nombre}.marca"), str(liberar)]
        return subprocess.Popen(argv, env=entorno, stderr=subprocess.PIPE, text=True)

    primero = lanzar("aaa")
    try:
        _esperar(tmp_path / "aaa.marca")  # el primero ya tiene el bloqueo y está dentro de embed_documents
        segundo = lanzar("bbb")
        error_segundo = segundo.communicate(timeout=30)[1]
        liberar.write_text("", encoding="utf-8")
        primero.communicate(timeout=30)
    finally:
        liberar.write_text("", encoding="utf-8")
        primero.kill()
    assert (primero.returncode, segundo.returncode) == (0, 1)
    assert "otra indexacion" in error_segundo
    assert _ids_del_indice(indexacion.cargar_vectorstore(_embeddings(), str(directorio))) == ["aaa_0"]


def test_30_validar_division_rechaza_exceso_duplicados_y_vacios_y_acepta_el_corpus_real(corpus):
    chunks, _ = indexacion.dividir_corpus(corpus, 200, 30)
    assert indexacion.validar_division(chunks, corpus)["chunks"] == 29
    muchos = _documentos_distintos(40)
    chunks_muchos, _ = indexacion.dividir_corpus(muchos, 50, 0)
    assert len(chunks_muchos) > indexacion.MAX_CHUNKS
    with pytest.raises(ValueError, match="chunks"):
        indexacion.validar_division(chunks_muchos, muchos)
    iguales = [{"id": n, "fuente": f"{n}.md", "texto": "mismo texto"} for n in ("a", "b")]
    chunks_iguales, _ = indexacion.dividir_corpus(iguales, 200, 30)
    with pytest.raises(ValueError, match="duplicados"):
        indexacion.validar_division(chunks_iguales, iguales)
    with pytest.raises(ValueError, match="vacíos"):
        indexacion.validar_division([Document(page_content=" ", metadata={"fuente": "a.md"})], [{"texto": " "}])
    with pytest.raises(ValueError, match="sin fuente"):
        indexacion.validar_division([Document(page_content="x", metadata={})], [{"texto": "x"}])


def test_31_indexar_rechaza_exceso_y_duplicados_antes_de_los_embeddings(tmp_path):
    embeddings = _EmbeddingsContador()
    iguales = [{"id": n, "fuente": f"{n}.md", "texto": "mismo texto"} for n in ("a", "b")]
    for documentos, tamano, solape in ((_documentos_distintos(40), 50, 0), (iguales, 200, 30)):
        with pytest.raises(ValueError):
            indexacion.indexar(documentos, embeddings, str(tmp_path / "indice"), chunk_size=tamano, chunk_overlap=solape)
    assert embeddings.llamadas == 0 and not (tmp_path / "indice").exists()
    assert not (tmp_path / "indice.lock").exists()


def test_32_ruta_absoluta_expande_la_tilde_y_no_crea_una_carpeta_tilde(corpus, tmp_path, monkeypatch):
    casa, cwd = tmp_path / "casa", tmp_path / "cwd"
    casa.mkdir()
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(casa))
    monkeypatch.chdir(cwd)
    assert indexacion.ruta_absoluta("~/idx") == str(casa / "idx")
    assert indexacion.indexar(corpus, _embeddings(), "~/idx")["directorio"] == str(casa / "idx")
    assert indexacion.cargar_vectorstore(_embeddings(), "~/idx")._collection.count() == 29
    assert list(cwd.iterdir()) == []
