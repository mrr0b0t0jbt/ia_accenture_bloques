import io
import logging

import pytest
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage

import registro_prompts
from registro_estructurado import crear_logger
from servicio_rag import SIN_CONTEXTO_VERIFICADO, ejecutar_servicio
from supervision_despliegue import SupervisionDespliegue

CORREO = "ana.perez@example.com"
PREGUNTA = "Cual es la politica de devoluciones? Mi correo es " + CORREO


class ModeloFalso:
    def __init__(self, texto):
        self.texto = texto

    def invoke(self, mensajes):
        return AIMessage(content=self.texto, usage_metadata={"input_tokens": 180, "output_tokens": 22, "total_tokens": 202})


def _indice(tmp_path, textos):
    docs = [Document(page_content=t, metadata={"fuente": f}) for t, f in textos]
    return Chroma.from_documents(docs, DeterministicFakeEmbedding(size=64), collection_name="gate2_test",
                                 persist_directory=str(tmp_path / "indice"))


@pytest.fixture
def servicio(tmp_path):
    indice = _indice(tmp_path, [("Politica de devoluciones: 30 dias desde la compra.", "manual.pdf"),
                                ("Politica de envios: 5 dias habiles a todo el pais.", "manual.pdf")])
    flujo = io.StringIO()
    logger = crear_logger("gate2_test", flujo)
    modelo = ModeloFalso("Puedes devolver un producto en 30 dias [1].")
    return lambda pregunta=PREGUNTA: ejecutar_servicio(pregunta, indice, modelo, k=2, logger=logger), flujo


def test_1_pipeline_sin_excepciones(servicio):
    ejecutar, _ = servicio
    respuesta, _ = ejecutar()
    assert respuesta.strip()


def test_2_respuesta_cita_fuente(servicio):
    ejecutar, _ = servicio
    respuesta, fuentes = ejecutar()
    assert fuentes
    assert any(f in respuesta for f in fuentes)


def test_3_logs_sin_contenido_crudo(servicio, caplog):
    ejecutar, flujo = servicio
    with caplog.at_level(logging.DEBUG):
        ejecutar()
    texto = flujo.getvalue() + " ".join(r.getMessage() + " " + str(r.__dict__) for r in caplog.records)
    assert flujo.getvalue().strip()
    assert CORREO not in texto
    assert PREGUNTA not in texto


def test_4_supervision_pausa_y_reanuda(tmp_path):
    with SupervisionDespliegue(str(tmp_path / "registro.jsonl")) as sup:
        propuesta = sup.iniciar("prompt-rag", "Texto del prompt", "autor", "nota")
        assert "motivo" in propuesta["pendiente"]
        assert not (tmp_path / "registro.jsonl").exists() or not registro_prompts.leer_registro(str(tmp_path / "registro.jsonl"))
        resultado = sup.reanudar(propuesta["thread_id"], True)
        assert registro_prompts.leer_registro(str(tmp_path / "registro.jsonl"))
        print(resultado)
