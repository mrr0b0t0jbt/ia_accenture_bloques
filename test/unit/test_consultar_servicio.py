import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import embeddings_gemini
import indexacion
import modelo_langchain

RAIZ = Path(__file__).resolve().parent.parent.parent
RUTA_SCRIPT = RAIZ / "scripts" / "consultar_servicio.py"
_spec = importlib.util.spec_from_file_location("consultar_servicio", RUTA_SCRIPT)
consultar_servicio = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(consultar_servicio)

PREGUNTA = "pregunta distintiva sobre devoluciones ana.perez@example.test"
BASE = ["--confirmar-api-real", "--pregunta", PREGUNTA]


class VectorstoreFalso:
    def __init__(self):
        self.llamadas = []

    def similarity_search_with_score(self, pregunta, k):
        self.llamadas.append(k)
        return [(Document(page_content="Devoluciones en 30 dias.", metadata={"fuente": "manual.pdf"}), 0.1)]


class ModeloFalso:
    def __init__(self, texto, stop_reason="end_turn"):
        self.texto = texto
        self.stop_reason = stop_reason

    def invoke(self, mensajes):
        return AIMessage(content=self.texto, usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
                         response_metadata={"stop_reason": self.stop_reason})


@pytest.fixture
def falsos(monkeypatch):
    """Reemplaza embeddings, índice y modelo por falsos; registra con qué se llamaron."""
    registro = {"vectorstore": VectorstoreFalso(), "texto": "Devoluciones en 30 dias [1]."}

    def cargar(embeddings, directorio, coleccion):
        registro["cargar"] = (embeddings, directorio, coleccion)
        return registro["vectorstore"]

    def crear_modelo(max_tokens):
        registro["max_tokens"] = max_tokens
        return ModeloFalso(registro["texto"], registro.get("stop_reason", "end_turn"))

    monkeypatch.setattr(embeddings_gemini, "crear_embeddings", lambda: "embeddings-falsos")
    monkeypatch.setattr(indexacion, "cargar_vectorstore", cargar)
    monkeypatch.setattr(modelo_langchain, "crear_modelo", crear_modelo)
    return registro


def test_sin_flag_sale_con_2_sin_cargar_langchain():
    codigo = (
        "import runpy, sys\n"
        f"sys.argv = ['consultar_servicio.py', '--pregunta', 'hola']\n"
        "try:\n"
        f"    runpy.run_path({str(RUTA_SCRIPT)!r}, run_name='__main__')\n"
        "except SystemExit as salida:\n"
        "    cargados = [m for m in ('langchain_anthropic', 'langchain_core', 'indexacion', 'modelo_langchain') if m in sys.modules]\n"
        "    print(salida.code, cargados)\n"
    )
    proceso = subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True, timeout=60)
    assert proceso.stdout.strip() == "2 []"
    assert "--confirmar-api-real" in proceso.stderr


def test_sin_flag_no_llama_a_nada(falsos, capsys):
    assert consultar_servicio.main(["--pregunta", PREGUNTA]) == 2
    assert "cargar" not in falsos
    assert PREGUNTA not in capsys.readouterr().err


def test_pregunta_es_obligatoria():
    with pytest.raises(SystemExit) as salida:
        consultar_servicio.main(["--confirmar-api-real"])
    assert salida.value.code == 2


def test_abreviatura_del_flag_no_se_acepta():
    with pytest.raises(SystemExit) as salida:
        consultar_servicio.main(["--conf", "--pregunta", PREGUNTA])
    assert salida.value.code == 2


@pytest.mark.parametrize("extra, texto", [
    (["--k", "0"], "--k"), (["--k", "11"], "--k"),
    (["--max-tokens", "0"], "--max-tokens"), (["--max-tokens", "1025"], "--max-tokens"),
])
def test_argumentos_fuera_de_rango(falsos, capsys, extra, texto):
    assert consultar_servicio.main(BASE + extra) == 1
    assert texto in capsys.readouterr().err
    assert "cargar" not in falsos


def test_pregunta_en_blanco(falsos, capsys):
    assert consultar_servicio.main(["--confirmar-api-real", "--pregunta", "   "]) == 1
    assert "--pregunta" in capsys.readouterr().err


def test_camino_feliz(falsos, capsys, tmp_path):
    codigo = consultar_servicio.main(BASE + ["--k", "2", "--max-tokens", "64", "--directorio", str(tmp_path),
                                            "--coleccion", "c_prueba"])
    salida = capsys.readouterr()
    assert codigo == 0
    assert "Fuentes:" in salida.out
    assert "manual.pdf" in salida.out
    assert falsos["cargar"] == ("embeddings-falsos", str(tmp_path), "c_prueba")
    assert falsos["max_tokens"] == 64
    assert falsos["vectorstore"].llamadas == [2]
    assert PREGUNTA not in salida.out + salida.err


def test_registro_json_sin_la_pregunta(falsos, capsys, tmp_path):
    assert consultar_servicio.main(BASE + ["--directorio", str(tmp_path), "--registro-json"]) == 0
    err = capsys.readouterr().err
    assert '"msg": "respuesta_generada"' in err
    assert PREGUNTA not in err
    assert "ana.perez@example.test" not in err


def test_sin_cita_da_aviso_y_codigo_0(falsos, capsys, tmp_path):
    falsos["texto"] = "Devoluciones en 30 dias."
    assert consultar_servicio.main(BASE + ["--directorio", str(tmp_path)]) == 0
    salida = capsys.readouterr()
    assert "Aviso:" in salida.err
    assert "sin_cita_valida" in salida.err


def test_respuesta_truncada_da_aviso_y_codigo_0(falsos, capsys, tmp_path):
    falsos["stop_reason"] = "max_tokens"
    assert consultar_servicio.main(BASE + ["--directorio", str(tmp_path)]) == 0
    salida = capsys.readouterr()
    assert "Aviso: la respuesta salio cortada por --max-tokens" in salida.err
    assert str(consultar_servicio.MAX_TOKENS_MAXIMO) in salida.err
    assert PREGUNTA not in salida.out + salida.err


def test_respuesta_completa_no_da_aviso_de_truncado(falsos, capsys, tmp_path):
    assert consultar_servicio.main(BASE + ["--directorio", str(tmp_path)]) == 0
    assert "cortada por --max-tokens" not in capsys.readouterr().err


def test_error_del_indice_sale_con_1_sin_traceback(monkeypatch, falsos, capsys, tmp_path):
    def fallar(*args):
        raise RuntimeError("indice roto " + "x" * 1000)

    monkeypatch.setattr(indexacion, "cargar_vectorstore", fallar)
    assert consultar_servicio.main(BASE + ["--directorio", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("Error:")
    assert len(err.strip()) <= consultar_servicio.MAX_LARGO_ERROR
    assert "Traceback" not in err


def test_error_de_la_clave_sale_con_1(monkeypatch, falsos, capsys, tmp_path):
    def sin_clave(max_tokens):
        raise ValueError("falta CLAUDE_API_KEY")

    monkeypatch.setattr(modelo_langchain, "crear_modelo", sin_clave)
    assert consultar_servicio.main(BASE + ["--directorio", str(tmp_path)]) == 1
    assert "CLAUDE_API_KEY" in capsys.readouterr().err


def test_pregunta_demasiado_larga_sale_con_1_sin_repetirla(falsos, capsys, tmp_path):
    larga = "secreto-" + "a" * 600
    assert consultar_servicio.main(["--confirmar-api-real", "--pregunta", larga, "--directorio", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "secreto-" not in err
    assert len(err.strip()) <= consultar_servicio.MAX_LARGO_ERROR


def test_ctrl_c_sale_con_130(monkeypatch, falsos, capsys, tmp_path):
    def interrumpir(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(indexacion, "cargar_vectorstore", interrumpir)
    assert consultar_servicio.main(BASE + ["--directorio", str(tmp_path)]) == 130
