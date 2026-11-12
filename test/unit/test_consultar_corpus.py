import importlib.util
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

import embeddings_gemini
import indexacion
import modelo_langchain
import recuperacion
from api_claude_refactor import INPUT_PRICING, OUTPUT_PRICING
from embeddings_falsos import EmbeddingsPorTema

RAIZ = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location("consultar_corpus", RAIZ / "scripts" / "consultar_corpus.py")
consultar_corpus = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(consultar_corpus)

RUTA_CASOS = RAIZ / "JSON" / "casos_recuperacion.json"
RUTA_CORPUS = RAIZ / "JSON" / "corpus_miniapp.json"
TEMAS = {
    "parcheo": ["parche", "parchear", "parcheo", "plazo", "vulnerabilidad", "cvss"],
    "phishing": ["phishing", "correo", "enlace", "clave", "remitente"],
    "respaldos": ["respaldo", "respaldos", "copia", "restauracion", "rpo", "rto", "ransomware"],
}
CLAVES_RESULTADO = {
    "id", "pregunta", "respondible", "fuentes_esperadas", "recuperados", "acierto_recuperacion", "respuesta", "citas",
    "admite_falta_de_contexto", "llamadas_modelo", "tokens_entrada", "tokens_salida", "costo_estimado", "truncado",
}
CASO_SIMPLE = {"id": "c1", "pregunta": "¿Cuál es el plazo de parcheo?", "fuentes_esperadas": ["politica_parcheo"], "respondible": True}
CASO_SIN_RESPUESTA = {"id": "c2", "pregunta": "¿Cuál es el presupuesto?", "fuentes_esperadas": [], "respondible": False}


def _mensaje(contenido="Respuesta [1].", entrada=100, salida=50, **metadata) -> AIMessage:
    uso = {"input_tokens": entrada, "output_tokens": salida, "total_tokens": entrada + salida}
    return AIMessage(content=contenido, usage_metadata=uso, response_metadata=metadata)


class ModeloGuionado(BaseChatModel):
    """Modelo falso mínimo: `guion(n, prompt)` devuelve un AIMessage o lanza la excepción del caso n (desde 1)."""

    guion: Callable
    prompts: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(messages[0].content)
        return ChatResult(generations=[ChatGeneration(message=self.guion(len(self.prompts), self.prompts[-1]))])

    @property
    def _llm_type(self) -> str:
        return "guionado"


@pytest.fixture
def embeddings():
    return EmbeddingsPorTema(TEMAS)


def _montar(tmp_path, monkeypatch, embeddings, guion=lambda n, prompt: _mensaje(), indexar=True):
    """Indexa el corpus real en tmp_path y parchea las fábricas para devolver el índice y el modelo falsos."""
    directorio = tmp_path / "indice"
    if indexar:
        indexacion.indexar(indexacion.cargar_corpus(str(RUTA_CORPUS)), embeddings, str(directorio))
    embeddings.llamadas_documentos = 0
    modelo = ModeloGuionado(guion=guion)
    fabrica = MagicMock(return_value=modelo)
    monkeypatch.setattr(embeddings_gemini, "crear_embeddings", lambda: embeddings)
    monkeypatch.setattr(modelo_langchain, "crear_modelo", fabrica)
    return directorio, modelo, fabrica


def _argv(tmp_path, directorio, casos=RUTA_CASOS, *extra):
    return ["--confirmar-api-real", "--casos", str(casos), "--directorio", str(directorio), "--salida", str(tmp_path / "salida.json"), *extra]


def _casos_en(tmp_path, casos) -> Path:
    ruta = tmp_path / "casos.json"
    ruta.write_text(json.dumps(casos), encoding="utf-8")
    return ruta


def _leer(tmp_path) -> dict:
    return json.loads((tmp_path / "salida.json").read_text(encoding="utf-8"))


def test_01_sin_confirmar_devuelve_2_sin_langchain_ni_claves_ni_disco(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)
    monkeypatch.setitem(sys.modules, "langchain_google_genai", None)
    embeddings_falso, modelo_falso = MagicMock(), MagicMock()
    monkeypatch.setattr(embeddings_gemini, "crear_embeddings", embeddings_falso)
    monkeypatch.setattr(modelo_langchain, "crear_modelo", modelo_falso)
    assert consultar_corpus.main(["--directorio", str(tmp_path / "indice"), "--salida", str(tmp_path / "s.json")]) == 2
    assert "--confirmar-api-real" in capsys.readouterr().err
    embeddings_falso.assert_not_called()
    modelo_falso.assert_not_called()
    assert list(tmp_path.iterdir()) == []


def test_02_los_casos_reales_son_validos():
    casos = json.loads(RUTA_CASOS.read_text(encoding="utf-8"))
    assert consultar_corpus.validar_casos_recuperacion(casos) is None
    assert len(casos) == 8 and len({c["id"] for c in casos}) == 8
    assert sum(c["respondible"] for c in casos) == 6
    ids_corpus = {documento["id"] for documento in indexacion.cargar_corpus(str(RUTA_CORPUS))}
    for caso in casos:
        assert set(caso["fuentes_esperadas"]) <= ids_corpus
        assert caso["respondible"] or caso["fuentes_esperadas"] == []


def test_03_validar_casos_rechaza_estructuras_invalidas():
    valido = dict(CASO_SIMPLE)
    invalidos = [
        {"id": "x"},
        [],
        [{**valido, "id": f"c{n}"} for n in range(11)],
        [{**valido, "pregunta": "  "}],
        [{**valido, "pregunta": "a" * 501}],
        [valido, dict(valido)],
        [{**valido, "fuentes_esperadas": "politica_parcheo"}],
        [{**valido, "respondible": "si"}],
        [{**valido, "fuentes_esperadas": []}],
        [{**CASO_SIN_RESPUESTA, "fuentes_esperadas": ["politica_parcheo"]}],
    ]
    assert consultar_corpus.validar_casos_recuperacion([valido, CASO_SIN_RESPUESTA]) is None
    for casos in invalidos:
        assert consultar_corpus.validar_casos_recuperacion(casos), casos


def test_04_corrida_completa_con_los_8_casos_reales(tmp_path, monkeypatch, capsys, embeddings):
    directorio, modelo, fabrica = _montar(tmp_path, monkeypatch, embeddings)
    assert consultar_corpus.main(_argv(tmp_path, directorio)) == 0
    salida = _leer(tmp_path)
    fabrica.assert_called_once_with(max_tokens=consultar_corpus.MAX_TOKENS_POR_DEFECTO)
    assert salida["k"] == 3 and salida["preguntas_procesadas"] == 8
    assert salida["consultas_de_embeddings"] == 8 and salida["llamadas_a_claude"] == 8
    assert "no es el cobro real" in salida["nota_costo"] and "embeddings" in salida["nota_costo"]
    assert salida["cortado_por_tope"] is False and len(salida["resultados"]) == 8
    costo = 100 * INPUT_PRICING + 50 * OUTPUT_PRICING
    assert salida["costo_total_estimado"] == pytest.approx(8 * costo)
    for resultado in salida["resultados"]:
        assert CLAVES_RESULTADO <= set(resultado) and "error" not in resultado
        assert resultado["costo_estimado"] == pytest.approx(costo) and resultado["tokens_entrada"] == 100
        assert len(resultado["recuperados"]) == 3
        assert resultado["citas"]["validas"][0][0] == 1
        for rango, recuperado in enumerate(resultado["recuperados"], start=1):
            assert set(recuperado) == {"rango", "documento", "fragmento", "fuente", "distancia"}
            assert recuperado["rango"] == rango
    assert salida["resultados"][6]["acierto_recuperacion"] is None
    for resultado in salida["resultados"][:6]:
        recuperados = {r["documento"] for r in resultado["recuperados"]}
        assert resultado["acierto_recuperacion"] is set(resultado["fuentes_esperadas"]).issubset(recuperados)
    assert embeddings.llamadas_documentos == 0
    assert "distancia" in capsys.readouterr().out


def test_05_no_respondible_detecta_si_admite_la_falta_de_contexto(tmp_path, monkeypatch, capsys, embeddings):
    casos = _casos_en(tmp_path, [CASO_SIN_RESPUESTA])
    for respuesta, admite in ((recuperacion.FRASE_SIN_CONTEXTO, True), ("El presupuesto es de USD 5 [1].", False)):
        directorio, _, _ = _montar(tmp_path, monkeypatch, embeddings, lambda n, prompt: _mensaje(respuesta))
        assert consultar_corpus.main(_argv(tmp_path, directorio, casos)) == 0
        assert _leer(tmp_path)["resultados"][0]["admite_falta_de_contexto"] is admite
        salida = capsys.readouterr().out
        assert ("NO ADMITIO la falta de contexto" in salida) is not admite
        assert ("ADMITIO la falta de contexto" in salida.replace("NO ADMITIO", "")) is admite


def test_06_cita_invalida_y_respuesta_sin_citas(tmp_path, monkeypatch, capsys, embeddings):
    respuestas = iter(["Dato [1] y otro [99].", "Sin corchetes."])
    directorio, _, _ = _montar(tmp_path, monkeypatch, embeddings, lambda n, prompt: _mensaje(next(respuestas)))
    casos = _casos_en(tmp_path, [CASO_SIMPLE, {**CASO_SIMPLE, "id": "c1b"}])
    assert consultar_corpus.main(_argv(tmp_path, directorio, casos)) == 0
    primero, segundo = _leer(tmp_path)["resultados"]
    assert primero["citas"]["invalidas"] == [99] and len(primero["citas"]["validas"]) == 1
    assert primero["citas"]["validas"][0][1].endswith(".md")
    assert segundo["citas"]["sin_citas"] is True
    salida = capsys.readouterr().out
    assert "Cita INVALIDA [99]" in salida and "SIN CITAS" in salida and "Cita [1] -> " in salida


def test_07_el_tope_de_gasto_corta_la_corrida(tmp_path, monkeypatch, embeddings):
    directorio, modelo, _ = _montar(tmp_path, monkeypatch, embeddings, lambda n, prompt: _mensaje(entrada=100_000))
    assert consultar_corpus.main(_argv(tmp_path, directorio, RUTA_CASOS, "--tope-usd", "0.0001")) == 0
    salida = _leer(tmp_path)
    assert salida["cortado_por_tope"] is True and salida["preguntas_procesadas"] == 1
    assert len(salida["resultados"]) == 1 and len(modelo.prompts) == 1
    assert salida["costo_total_estimado"] == pytest.approx(salida["resultados"][0]["costo_estimado"])
    assert salida["costo_total_estimado"] > 0.0001


def test_08_la_respuesta_con_bloques_guarda_solo_el_texto(tmp_path, monkeypatch, embeddings):
    bloques = [{"type": "thinking", "thinking": "pienso", "signature": "FIRMA_SECRETA"}, {"type": "text", "text": "Plazo de 4 días [1]."}]
    directorio, _, _ = _montar(tmp_path, monkeypatch, embeddings, lambda n, prompt: _mensaje(bloques))
    assert consultar_corpus.main(_argv(tmp_path, directorio, _casos_en(tmp_path, [CASO_SIMPLE]))) == 0
    assert _leer(tmp_path)["resultados"][0]["respuesta"] == "Plazo de 4 días [1]."
    assert "FIRMA_SECRETA" not in (tmp_path / "salida.json").read_text(encoding="utf-8")


def test_09_un_error_del_modelo_no_corta_los_demas_casos(tmp_path, monkeypatch, embeddings):
    def guion(n, prompt):
        if n == 2:
            raise RuntimeError("fallo del modelo")
        return _mensaje()

    directorio, _, _ = _montar(tmp_path, monkeypatch, embeddings, guion)
    assert consultar_corpus.main(_argv(tmp_path, directorio)) == 0
    resultados = _leer(tmp_path)["resultados"]
    assert len(resultados) == 8
    assert "fallo del modelo" in resultados[1]["error"]
    assert [n for n, r in enumerate(resultados) if "error" in r] == [1]


def test_10_indice_inexistente_vacio_o_sin_las_fuentes_esperadas(tmp_path, monkeypatch, capsys, embeddings):
    directorio, _, _ = _montar(tmp_path, monkeypatch, embeddings, indexar=False)
    assert consultar_corpus.main(_argv(tmp_path, directorio)) == 1
    assert "Error:" in capsys.readouterr().err and not directorio.exists()
    # sin --directorio ni --casos, la raíz sale de la ubicación del script y no del directorio actual
    monkeypatch.chdir(tmp_path)
    assert consultar_corpus.main(["--confirmar-api-real", "--salida", str(tmp_path / "salida.json")]) == 1
    assert str(RAIZ / "vectorstore") in capsys.readouterr().err
    indexacion.abrir_vectorstore(embeddings, str(directorio), indexacion.COLECCION_POR_DEFECTO)
    assert consultar_corpus.main(_argv(tmp_path, directorio)) == 1
    assert "Error:" in capsys.readouterr().err
    directorio, modelo, _ = _montar(tmp_path, monkeypatch, embeddings)
    casos = _casos_en(tmp_path, [{**CASO_SIMPLE, "fuentes_esperadas": ["documento_fantasma"]}])
    assert consultar_corpus.main(_argv(tmp_path, directorio, casos)) == 1
    assert "documento_fantasma" in capsys.readouterr().err
    assert modelo.prompts == []


def test_11_claves_faltantes_dan_error_sin_traceback(tmp_path, monkeypatch, capsys, embeddings):
    directorio, _, fabrica = _montar(tmp_path, monkeypatch, embeddings)
    fabrica.side_effect = ValueError("Falta CLAUDE_API_KEY en .env.")
    assert consultar_corpus.main(_argv(tmp_path, directorio)) == 1
    err = capsys.readouterr().err
    assert "Error: Falta CLAUDE_API_KEY" in err and "Traceback" not in err

    def sin_clave():
        raise ValueError("Falta GEMINI_API_KEY en .env.")

    monkeypatch.setattr(embeddings_gemini, "crear_embeddings", sin_clave)
    assert consultar_corpus.main(_argv(tmp_path, directorio)) == 1
    err = capsys.readouterr().err
    assert "Error: Falta GEMINI_API_KEY" in err and "Traceback" not in err


def test_12_salida_igual_a_casos_y_argumentos_fuera_de_rango_se_rechazan(tmp_path, monkeypatch, capsys, embeddings):
    directorio, _, fabrica = _montar(tmp_path, monkeypatch, embeddings)
    casos = _casos_en(tmp_path, [CASO_SIMPLE])
    original = casos.read_text(encoding="utf-8")
    enlace, duro = tmp_path / "enlace.json", tmp_path / "duro.json"
    enlace.symlink_to(casos)
    os.link(casos, duro)
    for salida in (casos, enlace, duro):
        argv = ["--confirmar-api-real", "--casos", str(casos), "--directorio", str(directorio), "--salida", str(salida)]
        assert consultar_corpus.main(argv) == 1
        assert "--salida" in capsys.readouterr().err
        assert casos.read_text(encoding="utf-8") == original
    for malo in (["--max-tokens", "2000"], ["--max-tokens", "0"], ["--tope-usd", "0.5"], ["--tope-usd", "0"], ["--k", "6"], ["--k", "0"]):
        assert consultar_corpus.main(_argv(tmp_path, directorio, casos, *malo)) == 1
        assert "Error:" in capsys.readouterr().err
    fabrica.assert_not_called()


def test_13_la_consola_limpia_los_controles_y_un_429_persistente_queda_como_error(tmp_path, monkeypatch, capsys, embeddings):
    crudo = "Hola \x1b[31mrojo\x1b[0m [1]"
    directorio, _, _ = _montar(tmp_path, monkeypatch, embeddings, lambda n, prompt: _mensaje(crudo))
    casos = _casos_en(tmp_path, [CASO_SIMPLE])
    assert consultar_corpus.main(_argv(tmp_path, directorio, casos)) == 0
    assert "\x1b" not in capsys.readouterr().out
    assert _leer(tmp_path)["resultados"][0]["respuesta"] == crudo

    def limite(texto):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    esperas = []
    monkeypatch.setattr(embeddings, "embed_query", limite)
    monkeypatch.setattr(indexacion.time, "sleep", esperas.append)
    directorio, modelo, _ = _montar(tmp_path, monkeypatch, embeddings, indexar=False)
    assert consultar_corpus.main(_argv(tmp_path, directorio, casos)) == 1
    resultado = _leer(tmp_path)["resultados"][0]
    assert "límite" in resultado["error"] and resultado["llamadas_modelo"] == 0 and resultado["respuesta"] == ""
    assert esperas == [5, 10, 20] and modelo.prompts == []
