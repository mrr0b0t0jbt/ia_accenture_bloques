import json
import os
import socket
import subprocess
import sys
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
