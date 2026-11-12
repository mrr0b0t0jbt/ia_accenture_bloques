import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

import embeddings_gemini
import indexacion

RAIZ = Path(__file__).resolve().parents[2]
SCRIPT = RAIZ / "scripts" / "indexar_corpus.py"
_spec = importlib.util.spec_from_file_location("indexar_corpus", SCRIPT)
indexar_corpus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(indexar_corpus)


class _EmbeddingsQueFallan(Embeddings):
    def __init__(self, error: Exception):
        self.error = error

    def embed_documents(self, textos):
        raise self.error

    def embed_query(self, texto):
        raise self.error


def _embeddings_falsos(monkeypatch):
    monkeypatch.setattr(embeddings_gemini, "crear_embeddings", lambda: DeterministicFakeEmbedding(size=16))


def test_01_modo_revision_no_usa_la_api_ni_escribe_el_indice(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "langchain_google_genai", None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def prohibido():
        raise AssertionError("el modo revisión no debe crear embeddings")

    monkeypatch.setattr(embeddings_gemini, "crear_embeddings", prohibido)
    assert indexar_corpus.main(["--directorio", str(tmp_path / "vs")]) == 0
    salida = capsys.readouterr().out
    for texto in (
        "Modo revisión: no se llama a ninguna API ni se escribe ningún índice.",
        "Corpus: 10 documentos, 4375 caracteres",
        "29 chunks, 4865 caracteres",
        "  politica_parcheo: 3 chunks",
        "Integridad: sin vacíos, sin duplicados, sin chunks sin fuente",
    ):
        assert texto in salida
    assert not (tmp_path / "vs").exists()


def test_02_comparar_tamanos_usa_el_solapamiento_indicado(capsys):
    assert indexar_corpus.main(["--comparar-tamanos", "120,200,300"]) == 0
    salida = capsys.readouterr().out
    assert "tamaño 120: 50 chunks, 5408 caracteres" in salida
    assert "tamaño 200: 29 chunks, 4865 caracteres" in salida
    assert "tamaño 300: 20 chunks, 4628 caracteres" in salida


def test_03_confirmar_api_real_indexa_el_corpus(tmp_path, monkeypatch, capsys):
    _embeddings_falsos(monkeypatch)
    directorio = tmp_path / "vs"
    assert indexar_corpus.main(["--directorio", str(directorio), "--confirmar-api-real"]) == 0
    salida = capsys.readouterr().out
    assert "Indexados 29 chunks (4865 caracteres)" in salida
    assert "Eliminados: 0" in salida
    assert "Chunks en el índice: 29" in salida
    assert "no se estima un costo en dólares" in salida
    reabierto = indexacion.cargar_vectorstore(DeterministicFakeEmbedding(size=16), str(directorio))
    assert reabierto._collection.count() == 29


def test_04_la_segunda_ejecucion_no_duplica(tmp_path, monkeypatch, capsys):
    _embeddings_falsos(monkeypatch)
    argv = ["--directorio", str(tmp_path / "vs"), "--confirmar-api-real"]
    assert indexar_corpus.main(argv) == 0
    capsys.readouterr()
    assert indexar_corpus.main(argv) == 0
    salida = capsys.readouterr().out
    assert "Eliminados: 0" in salida and "Chunks en el índice: 29" in salida


def test_05_falta_de_clave_es_un_error_claro(tmp_path, monkeypatch, capsys):
    def sin_clave():
        raise ValueError("Falta GEMINI_API_KEY en .env.")

    monkeypatch.setattr(embeddings_gemini, "crear_embeddings", sin_clave)
    directorio = tmp_path / "vs"
    assert indexar_corpus.main(["--directorio", str(directorio), "--confirmar-api-real"]) == 1
    error = capsys.readouterr().err
    assert "Error: Falta GEMINI_API_KEY en .env." in error and "Traceback" not in error
    assert not directorio.exists()


def test_06_un_429_persistente_se_informa_con_las_esperas_de_indexacion(tmp_path, monkeypatch, capsys):
    class Embeddings429(Embeddings):
        def embed_documents(self, textos):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

        def embed_query(self, texto):
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    esperas = []
    monkeypatch.setattr(embeddings_gemini, "crear_embeddings", lambda: Embeddings429())
    monkeypatch.setattr(indexacion.time, "sleep", esperas.append)
    assert indexar_corpus.main(["--directorio", str(tmp_path / "vs"), "--confirmar-api-real"]) == 1
    error = capsys.readouterr().err
    assert "Error:" in error and "límite de peticiones" in error and "Traceback" not in error
    assert esperas == [5, 10, 20]


def test_07_corpus_inexistente_o_invalido(tmp_path, capsys):
    roto = tmp_path / "roto.json"
    roto.write_text("{[", encoding="utf-8")
    for ruta in (tmp_path / "no_existe.json", roto):
        assert indexar_corpus.main(["--corpus", str(ruta)]) == 1
        assert "Error:" in capsys.readouterr().err


def test_08_argumentos_invalidos_terminan_con_error(capsys):
    casos = [
        ["--chunk-size", "10"],
        ["--chunk-overlap", "500"],
        ["--comparar-tamanos", "abc"],
        ["--comparar-tamanos", "120,5000"],
    ]
    for argv in casos:
        assert indexar_corpus.main(argv) == 1
        assert "Error:" in capsys.readouterr().err


def test_11_el_directorio_y_el_corpus_por_defecto_salen_de_la_raiz_y_no_del_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    recibido = {}

    def indexar_falso(documentos, embeddings, directorio, **kwargs):
        recibido["directorio"] = directorio
        recibido["documentos"] = documentos
        return {"chunks": 0, "caracteres_chunks": 0, "directorio": directorio, "coleccion": "x", "eliminados": 0, "total_en_indice": 0}

    monkeypatch.setattr(embeddings_gemini, "crear_embeddings", lambda: object())
    monkeypatch.setattr(indexacion, "indexar", indexar_falso)
    assert indexar_corpus.main(["--confirmar-api-real"]) == 0
    assert recibido["directorio"] == str(indexar_corpus.RAIZ / "vectorstore")
    assert recibido["documentos"] == indexacion.cargar_corpus(str(indexar_corpus.RAIZ / "JSON" / "corpus_miniapp.json"))
    assert list(tmp_path.iterdir()) == []


def test_12_un_corpus_con_textos_repetidos_en_modo_revision_devuelve_1(tmp_path, capsys):
    corpus = [{"id": n, "fuente": f"{n}.md", "texto": "mismo texto de prueba"} for n in ("a", "b")]
    ruta = tmp_path / "corpus.json"
    ruta.write_text(json.dumps(corpus), encoding="utf-8")
    assert indexar_corpus.main(["--corpus", str(ruta)]) == 1
    assert "Integridad: problemas" in capsys.readouterr().out


def test_13_una_clave_rechazada_o_un_403_se_informan_sin_traceback(tmp_path, monkeypatch, capsys):
    errores = (
        "Error embedding content (INVALID_ARGUMENT): 400 INVALID_ARGUMENT. {'error': {'code': 400, "
        "'message': 'API key not valid. Please pass a valid API key.', 'status': 'INVALID_ARGUMENT'}}",
        "Error embedding content (PERMISSION_DENIED): 403 PERMISSION_DENIED. {'error': {'code': 403, "
        "'message': 'The caller does not have permission', 'status': 'PERMISSION_DENIED'}}",
    )
    for texto in errores:
        monkeypatch.setattr(embeddings_gemini, "crear_embeddings", lambda texto=texto: _EmbeddingsQueFallan(RuntimeError(texto)))
        assert indexar_corpus.main(["--directorio", str(tmp_path / "vs"), "--confirmar-api-real"]) == 1
        error = capsys.readouterr().err
        assert "Error:" in error and "rechazó la clave" in error and "Traceback" not in error


def test_14_un_error_cualquiera_se_informa_sin_traceback_y_un_argumento_roto_es_de_argparse(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(embeddings_gemini, "crear_embeddings", lambda: _EmbeddingsQueFallan(RuntimeError("boom")))
    assert indexar_corpus.main(["--directorio", str(tmp_path / "vs"), "--confirmar-api-real"]) == 1
    error = capsys.readouterr().err
    assert "Error:" in error and "RuntimeError" in error and "boom" in error and "Traceback" not in error
    with pytest.raises(SystemExit) as salida:
        indexar_corpus.main(["--chunk-size", "abc"])
    assert salida.value.code == 2


def test_10_gitignore_y_el_script_no_carga_gemini_al_importarse(tmp_path):
    lineas = (RAIZ / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "vectorstore/" in lineas and "vectorstore*/" in lineas
    # intérprete limpio: lo que ya importó pytest no enmascara lo que carga el script
    codigo = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('indexar_corpus', sys.argv[1])\n"
        "modulo = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(modulo)\n"
        "assert 'langchain_google_genai' not in sys.modules, 'al importar'\n"
        "assert modulo.main(['--directorio', sys.argv[2]]) == 0\n"
        "assert 'langchain_google_genai' not in sys.modules, 'en modo revisión'\n"
    )
    resultado = subprocess.run(
        [sys.executable, "-c", codigo, str(SCRIPT), str(tmp_path / "vs")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert resultado.returncode == 0, resultado.stderr
    assert not (tmp_path / "vs").exists()
