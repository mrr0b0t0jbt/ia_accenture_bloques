import gc
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from chromadb.api.shared_system_client import SharedSystemClient

import indexacion
from embeddings_falsos import EmbeddingsPorTema

RAIZ = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location("preflight_rag", RAIZ / "scripts" / "preflight_rag.py")
preflight = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(preflight)

TEMAS = {"parcheo": ["parche", "cvss"], "respaldos": ["respaldo", "rpo"], "phishing": ["phishing", "correo"]}
# cada documento cabe en un solo chunk: ids parcheo_0, respaldos_0 y phishing_0
CORPUS = [
    {"id": "parcheo", "fuente": "parcheo.md", "texto": "Politica de parcheo: el parche critico se aplica en cuatro dias con cvss alto."},
    {"id": "respaldos", "fuente": "respaldos.md", "texto": "Guia de respaldos: copia diaria, prueba de restauracion mensual y rpo de 24 horas."},
    {"id": "phishing", "fuente": "phishing.md", "texto": "Playbook phishing: reportar el correo y no abrir el enlace sospechoso."},
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


def _montar(tmp_path, embeddings, corpus_indexado=CORPUS, coleccion=indexacion.COLECCION_POR_DEFECTO, gitignore="vectorstore/\n"):
    """Escribe corpus y .gitignore en tmp_path e indexa corpus_indexado (None: no crea el índice); devuelve (directorio, argumentos)."""
    directorio = tmp_path / "vectorstore"
    (tmp_path / "corpus.json").write_text(json.dumps(CORPUS), encoding="utf-8")
    if gitignore is not None:
        (tmp_path / ".gitignore").write_text(gitignore, encoding="utf-8")
    if corpus_indexado is not None:
        indexacion.indexar(corpus_indexado, embeddings, str(directorio), coleccion=coleccion)
    return directorio, (str(directorio), indexacion.COLECCION_POR_DEFECTO, str(tmp_path / "corpus.json"), str(tmp_path / ".gitignore"))


def _foto(directorio: Path) -> dict:
    return {str(p.relative_to(directorio)): p.stat().st_size for p in sorted(directorio.rglob("*")) if p.is_file()}


# P1
def test_p1_todo_en_orden(tmp_path, embeddings, capsys):
    directorio, args = _montar(tmp_path, embeddings)
    assert preflight.preflight_check(*args) == []
    assert preflight.main(["--directorio", args[0], "--corpus", args[2], "--gitignore", args[3]]) == 0
    assert "Preflight OK" in capsys.readouterr().out


# P2
def test_p2_directorio_inexistente_no_se_crea_ni_se_abre_chroma(tmp_path, embeddings, monkeypatch):
    directorio, args = _montar(tmp_path, embeddings, corpus_indexado=None)

    def prohibido(*a, **k):
        raise AssertionError("no debe abrir Chroma")

    monkeypatch.setattr(preflight.chromadb, "PersistentClient", prohibido)
    problemas = preflight.preflight_check(*args)
    assert len(problemas) == 1 and "indexa" in problemas[0]
    assert not directorio.exists()


# P3
@pytest.mark.parametrize("estado", ["inexistente", "vacia"])
def test_p3_coleccion_inexistente_o_vacia(tmp_path, embeddings, estado):
    if estado == "inexistente":
        directorio, args = _montar(tmp_path, embeddings, coleccion="otra_coleccion")
    else:
        directorio, args = _montar(tmp_path, embeddings, corpus_indexado=None)
        indexacion.abrir_vectorstore(embeddings, str(directorio), indexacion.COLECCION_POR_DEFECTO)
    problemas = preflight.preflight_check(*args)
    assert len(problemas) == 1 and "ndexa" in problemas[0]


# P4
@pytest.mark.parametrize("estado", ["conteo", "ids"])
def test_p4_el_indice_no_coincide_con_el_corpus(tmp_path, embeddings, estado):
    indexado = CORPUS[:2] if estado == "conteo" else [{**d, "id": f"{d['id']}2"} for d in CORPUS]
    directorio, args = _montar(tmp_path, embeddings, corpus_indexado=indexado)
    problemas = preflight.preflight_check(*args)
    assert len(problemas) == 1 and "reindexa" in problemas[0]
    if estado == "conteo":
        assert "tiene 2 chunks" in problemas[0] and "produce 3" in problemas[0] and "faltan 1 (p. ej. phishing_0)" in problemas[0]
        assert "sobran 0" in problemas[0]
    else:
        assert "faltan 3" in problemas[0] and "sobran 3" in problemas[0] and "parcheo2_0" in problemas[0] and "parcheo_0" in problemas[0]


# P5
def test_p5_corpus_invalido_es_un_problema_sin_excepcion(tmp_path, embeddings):
    directorio, args = _montar(tmp_path, embeddings)
    for contenido, mensaje in (("{roto", "no es un JSON válido"), (json.dumps([{"id": "x"}]), "no tiene la clave fuente")):
        (tmp_path / "corpus.json").write_text(contenido, encoding="utf-8")
        problemas = preflight.preflight_check(*args)
        assert len(problemas) == 1 and "Corpus" in problemas[0] and mensaje in problemas[0]


# P6
def test_p6_gitignore_inexistente(tmp_path, embeddings):
    directorio, args = _montar(tmp_path, embeddings, gitignore=None)
    problemas = preflight.preflight_check(*args)
    assert len(problemas) == 1 and ".gitignore" in problemas[0]


# P7
@pytest.mark.parametrize(
    "contenido, ignora",
    [
        ("vectorstore/\n", True),
        ("vectorstore\n", True),
        ("/vectorstore/\n", True),
        ("vectorstore*\n", True),
        ("vector*\n", True),
        ("vectorstore_viejo/\n", False),
        ("# vectorstore/\n", False),
        ("otro/vectorstore_x\n", False),
        ("vectorstore/\n!vectorstore/\n", False),
        ("!vectorstore/\nvectorstore/\n", True),
    ],
    ids=["barra_final", "sin_barra", "barra_inicial", "comodin_final", "comodin_prefijo", "viejo", "comentario", "otra_ruta", "negada", "negada_antes"],
)
def test_p7_lectura_del_gitignore(tmp_path, embeddings, contenido, ignora):
    directorio, args = _montar(tmp_path, embeddings, gitignore=contenido)
    problemas = preflight.preflight_check(*args)
    assert (problemas == []) is ignora
    assert ignora or (len(problemas) == 1 and "no ignora" in problemas[0])


# P8
def test_p8_main_con_problemas_limpia_el_nombre_hostil(tmp_path, embeddings, capsys):
    directorio, args = _montar(tmp_path, embeddings, corpus_indexado=None, gitignore=None)
    hostil = tmp_path / "vector\x1b[31m\nstore"
    assert preflight.main(["--directorio", str(hostil), "--corpus", args[2], "--gitignore", args[3]]) == 1
    out = capsys.readouterr().out
    lineas = out.splitlines()
    assert "\x1b" not in out and lineas[0] == "Preflight con problemas:"
    assert len(lineas) == 3 and all(linea.startswith(" - ") for linea in lineas[1:])
    assert "vector [31m store" in lineas[1]


# P9
def test_p9_sin_apis_ni_env_y_sin_tocar_el_indice(tmp_path, embeddings, monkeypatch):
    directorio, args = _montar(tmp_path, embeddings)
    monkeypatch.setitem(sys.modules, "embeddings_gemini", None)
    monkeypatch.setitem(sys.modules, "langchain_google_genai", None)

    def prohibido(*a, **k):
        raise AssertionError("no debe leer .env")

    monkeypatch.setattr("dotenv.load_dotenv", prohibido)
    spec = importlib.util.spec_from_file_location("preflight_rag_nuevo", RAIZ / "scripts" / "preflight_rag.py")
    nuevo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nuevo)  # falla si importa alguno de los dos módulos bloqueados
    antes, archivos = _foto(directorio), sorted(p.name for p in tmp_path.iterdir())
    assert nuevo.main(["--directorio", args[0], "--corpus", args[2], "--gitignore", args[3]]) == 0
    assert _foto(directorio) == antes and sorted(p.name for p in tmp_path.iterdir()) == archivos
