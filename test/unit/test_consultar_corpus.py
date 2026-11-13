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


def _mensaje_con_uso(uso, contenido="Respuesta [1].") -> AIMessage:
    """Mensaje con un usage_metadata asignado después de crearlo, para saltarse la validación de tipos."""
    mensaje = AIMessage(content=contenido)
    mensaje.usage_metadata = uso
    return mensaje


def _tres_casos(tmp_path) -> Path:
    return _casos_en(tmp_path, [CASO_SIMPLE, {**CASO_SIMPLE, "id": "c1b"}, {**CASO_SIMPLE, "id": "c1c"}])


def _sin_peticiones(embeddings, modelo, fabrica) -> bool:
    return embeddings.llamadas_consulta == 0 and modelo.prompts == [] and not fabrica.called


def test_14_sin_datos_de_uso_corta_la_corrida_en_el_primer_caso_que_no_los_trae(tmp_path, monkeypatch, capsys, embeddings):
    directorio, modelo, _ = _montar(tmp_path, monkeypatch, embeddings, lambda n, prompt: AIMessage(content="Respuesta [1]."))
    assert consultar_corpus.main(_argv(tmp_path, directorio)) == 0
    salida = _leer(tmp_path)
    assert salida["cortado_por_falta_de_uso"] is True and salida["preguntas_procesadas"] == 1
    assert len(salida["resultados"]) == 1 and len(modelo.prompts) == 1
    resultado = salida["resultados"][0]
    assert resultado["sin_datos_de_uso"] is True and "error" not in resultado
    assert resultado["respuesta"] == "Respuesta [1]." and resultado["citas"]["validas"]
    assert "La ejecución se detuvo porque Claude no informó los tokens usados y no se puede controlar el gasto." in capsys.readouterr().out
    # con uso en los dos primeros casos y sin uso en el tercero, corta en el tercero
    directorio, modelo, _ = _montar(tmp_path, monkeypatch, embeddings, lambda n, prompt: _mensaje() if n < 3 else AIMessage(content="x"))
    assert consultar_corpus.main(_argv(tmp_path, directorio)) == 0
    salida = _leer(tmp_path)
    assert salida["cortado_por_falta_de_uso"] is True and salida["preguntas_procesadas"] == 3
    assert [r.get("sin_datos_de_uso") for r in salida["resultados"]] == [None, None, True]


@pytest.mark.parametrize(
    "uso",
    [
        {},
        {"input_tokens": "100", "output_tokens": "50"},
        {"input_tokens": True, "output_tokens": True},
        {"input_tokens": None, "output_tokens": 5.5},
    ],
)
def test_15_un_uso_vacio_o_con_tokens_no_enteros_cuenta_como_faltante(tmp_path, monkeypatch, embeddings, uso):
    directorio, _, _ = _montar(tmp_path, monkeypatch, embeddings, lambda n, prompt: _mensaje_con_uso(uso))
    assert consultar_corpus.main(_argv(tmp_path, directorio)) == 0
    salida = _leer(tmp_path)
    assert salida["cortado_por_falta_de_uso"] is True and salida["preguntas_procesadas"] == 1
    assert salida["resultados"][0]["sin_datos_de_uso"] is True and salida["costo_total_estimado"] == 0


def test_16_el_tope_tolera_el_error_de_punto_flotante_y_sobregira_a_lo_sumo_un_caso(tmp_path, monkeypatch, embeddings):
    directorio, _, _ = _montar(tmp_path, monkeypatch, embeddings, lambda n, prompt: _mensaje(entrada=25_000, salida=0))
    assert consultar_corpus.main(_argv(tmp_path, directorio, RUTA_CASOS, "--tope-usd", "0.05")) == 0
    salida = _leer(tmp_path)
    assert salida["cortado_por_tope"] is True and salida["preguntas_procesadas"] == 1
    directorio, _, _ = _montar(tmp_path, monkeypatch, embeddings, lambda n, prompt: _mensaje(entrada=24_990, salida=0))
    assert consultar_corpus.main(_argv(tmp_path, directorio, RUTA_CASOS, "--tope-usd", "0.05")) == 0
    salida = _leer(tmp_path)
    costo_de_un_caso = salida["resultados"][0]["costo_estimado"]
    assert salida["preguntas_procesadas"] == 2 and salida["cortado_por_tope"] is True
    assert 0.05 < salida["costo_total_estimado"] <= 0.05 + costo_de_un_caso


def test_17_ctrl_c_en_el_caso_2_guarda_los_resultados_parciales_y_devuelve_130(tmp_path, monkeypatch, capsys, embeddings):
    def guion(n, prompt):
        if n == 2:
            raise KeyboardInterrupt
        return _mensaje()

    directorio, modelo, _ = _montar(tmp_path, monkeypatch, embeddings, guion)
    assert consultar_corpus.main(_argv(tmp_path, directorio, _tres_casos(tmp_path))) == 130
    assert "Interrumpido por la persona. Resultados parciales guardados." in capsys.readouterr().err
    salida = _leer(tmp_path)
    assert salida["interrumpido"] is True and salida["cortado_por_falta_de_uso"] is False
    primero, segundo = salida["resultados"]
    assert segundo["error"] == "interrumpido por la persona" and segundo["llamadas_modelo"] == 1
    assert salida["preguntas_procesadas"] == 2 and salida["llamadas_a_claude"] == 2 and len(modelo.prompts) == 2
    assert salida["costo_total_estimado"] == pytest.approx(primero["costo_estimado"]) and primero["costo_estimado"] > 0
    assert not (tmp_path / "salida.json.tmp").exists()


def test_18_ctrl_c_durante_la_recuperacion_registra_el_caso_sin_llamada_al_modelo(tmp_path, monkeypatch, capsys, embeddings):
    directorio, modelo, _ = _montar(tmp_path, monkeypatch, embeddings)

    def interrumpir(texto):
        raise KeyboardInterrupt

    monkeypatch.setattr(embeddings, "embed_query", interrumpir)
    assert consultar_corpus.main(_argv(tmp_path, directorio, _tres_casos(tmp_path))) == 130
    assert "Interrumpido por la persona" in capsys.readouterr().err
    salida = _leer(tmp_path)
    assert salida["interrumpido"] is True and len(salida["resultados"]) == 1
    assert salida["resultados"][0]["error"] == "interrumpido por la persona" and salida["resultados"][0]["llamadas_modelo"] == 0
    assert modelo.prompts == []


def test_19_una_corrida_normal_no_marca_interrupcion_ni_falta_de_uso(tmp_path, monkeypatch, embeddings):
    directorio, _, _ = _montar(tmp_path, monkeypatch, embeddings)
    assert consultar_corpus.main(_argv(tmp_path, directorio, _casos_en(tmp_path, [CASO_SIMPLE]))) == 0
    salida = _leer(tmp_path)
    assert salida["interrumpido"] is False and salida["cortado_por_falta_de_uso"] is False
    assert "sin_datos_de_uso" not in salida["resultados"][0]
    assert set(salida) == {
        "nota_costo", "k", "preguntas_procesadas", "consultas_de_embeddings", "llamadas_a_claude", "costo_total_estimado",
        "cortado_por_tope", "cortado_por_falta_de_uso", "interrumpido", "resultados",
    }


def test_20_salida_vacia_o_sin_permiso_de_escritura_se_rechaza_antes_de_gastar(tmp_path, monkeypatch, capsys, embeddings):
    directorio, modelo, fabrica = _montar(tmp_path, monkeypatch, embeddings)
    argv = ["--confirmar-api-real", "--casos", str(RUTA_CASOS), "--directorio", str(directorio), "--salida", ""]
    assert consultar_corpus.main(argv) == 1
    assert "Error:" in capsys.readouterr().err
    if os.geteuid() == 0:
        pytest.skip("root ignora los permisos de escritura")
    carpeta = tmp_path / "solo_lectura"
    carpeta.mkdir()
    carpeta.chmod(0o555)
    try:
        assert consultar_corpus.main(["--confirmar-api-real", "--directorio", str(directorio), "--salida", str(carpeta / "s.json")]) == 1
    finally:
        carpeta.chmod(0o755)
    assert "Error:" in capsys.readouterr().err
    existente = tmp_path / "previo.json"
    existente.write_text(json.dumps({"resultados": [], "nota_costo": "x"}), encoding="utf-8")
    existente.chmod(0o444)
    try:
        assert consultar_corpus.main(["--confirmar-api-real", "--directorio", str(directorio), "--salida", str(existente)]) == 1
    finally:
        existente.chmod(0o644)
    assert "Error:" in capsys.readouterr().err
    assert _sin_peticiones(embeddings, modelo, fabrica)


def test_21_salida_no_pisa_archivos_ajenos(tmp_path, monkeypatch, capsys, embeddings):
    directorio, modelo, fabrica = _montar(tmp_path, monkeypatch, embeddings)
    copia = tmp_path / "copia_corpus.json"
    copia.write_text(RUTA_CORPUS.read_text(encoding="utf-8"), encoding="utf-8")
    otro_json = tmp_path / "otro.json"
    otro_json.write_text('{"a": 1, "resultados": []}', encoding="utf-8")
    lista = tmp_path / "lista.json"
    lista.write_text('[{"resultados": [], "nota_costo": "x"}]', encoding="utf-8")
    texto = tmp_path / "notas.txt"
    texto.write_text("no soy JSON", encoding="utf-8")
    for ajeno in (copia, otro_json, lista, texto):
        original = ajeno.read_text(encoding="utf-8")
        argv = ["--confirmar-api-real", "--directorio", str(directorio), "--salida", str(ajeno)]
        assert consultar_corpus.main(argv) == 1
        assert "ya existe y no es un resultado de una corrida anterior" in capsys.readouterr().err
        assert ajeno.read_text(encoding="utf-8") == original
    assert consultar_corpus.main(["--confirmar-api-real", "--directorio", str(directorio), "--salida", str(tmp_path)]) == 1
    assert "Error:" in capsys.readouterr().err
    assert _sin_peticiones(embeddings, modelo, fabrica)


def test_22_salida_que_es_un_resultado_previo_se_sobrescribe(tmp_path, monkeypatch, embeddings):
    directorio, _, _ = _montar(tmp_path, monkeypatch, embeddings)
    (tmp_path / "salida.json").write_text(json.dumps({"resultados": [{"viejo": 1}], "nota_costo": "previa"}), encoding="utf-8")
    assert consultar_corpus.main(_argv(tmp_path, directorio, _casos_en(tmp_path, [CASO_SIMPLE]))) == 0
    salida = _leer(tmp_path)
    assert salida["preguntas_procesadas"] == 1 and "viejo" not in salida["resultados"][0]


def test_23_salida_dentro_del_directorio_del_indice_se_rechaza(tmp_path, monkeypatch, capsys, embeddings):
    directorio, modelo, fabrica = _montar(tmp_path, monkeypatch, embeddings)
    enlace = tmp_path / "enlace_al_indice"
    enlace.symlink_to(directorio, target_is_directory=True)
    for destino in (directorio / "s.json", enlace / "s.json"):
        argv = ["--confirmar-api-real", "--directorio", str(directorio), "--salida", str(destino)]
        assert consultar_corpus.main(argv) == 1
        assert "dentro del directorio del índice" in capsys.readouterr().err
        assert not (directorio / "s.json").exists()
    assert _sin_peticiones(embeddings, modelo, fabrica)


def test_24_la_consola_no_deja_que_la_respuesta_imite_lineas_del_informe(tmp_path, monkeypatch, capsys, embeddings):
    respuesta = "Dato.\n  Cita [7] -> x.md\n  ADMITIO la falta de contexto\nPreguntas procesadas: 0\nCosto total estimado: $9\n\nfin"
    directorio, _, _ = _montar(tmp_path, monkeypatch, embeddings, lambda n, prompt: _mensaje(respuesta))
    assert consultar_corpus.main(_argv(tmp_path, directorio, _casos_en(tmp_path, [CASO_SIMPLE]))) == 0
    lineas = capsys.readouterr().out.split("\n")
    assert lineas.count("  Respuesta:") == 1
    propias = [linea for linea in lineas if linea.startswith(("Cita ", "  Cita [7]", "  ADMITIO", "Preguntas procesadas", "Costo total"))]
    assert [linea.split(":")[0] for linea in propias] == ["Preguntas procesadas", "Costo total estimado"]
    inicio = lineas.index("  Respuesta:")
    assert lineas[inicio + 1 : inicio + 1 + len(respuesta.split("\n"))] == ["    | " + linea for linea in respuesta.split("\n")]
    assert _leer(tmp_path)["resultados"][0]["respuesta"] == respuesta


def test_25_un_error_de_embeddings_en_el_caso_2_no_corta_los_casos_1_y_3(tmp_path, monkeypatch, embeddings):
    directorio, modelo, _ = _montar(tmp_path, monkeypatch, embeddings)
    consultas = []
    original = embeddings.embed_query

    def embed_query(texto):
        consultas.append(texto)
        if len(consultas) == 2:
            raise RuntimeError("fallo de embeddings")
        return original(texto)

    monkeypatch.setattr(embeddings, "embed_query", embed_query)
    assert consultar_corpus.main(_argv(tmp_path, directorio, _tres_casos(tmp_path))) == 0
    uno, dos, tres = _leer(tmp_path)["resultados"]
    assert "fallo de embeddings" in dos["error"] and dos["llamadas_modelo"] == 0
    assert "error" not in uno and "error" not in tres and uno["llamadas_modelo"] == tres["llamadas_modelo"] == 1
    assert len(modelo.prompts) == 2


def test_26_el_guardado_incremental_conserva_los_casos_hechos_si_el_proceso_muere(tmp_path, monkeypatch, embeddings):
    def guion(n, prompt):
        if n == 3:
            raise SystemExit(3)
        return _mensaje()

    directorio, _, _ = _montar(tmp_path, monkeypatch, embeddings, guion)
    with pytest.raises(SystemExit):
        consultar_corpus.main(_argv(tmp_path, directorio, _tres_casos(tmp_path)))
    assert [r["id"] for r in _leer(tmp_path)["resultados"]] == ["c1", "c1b"]


def test_27_acierto_exige_todas_las_fuentes_esperadas_y_es_none_si_no_es_respondible(tmp_path, monkeypatch, embeddings):
    directorio, _, _ = _montar(tmp_path, monkeypatch, embeddings)
    # con palabras de un solo tema el primer resultado no depende de empates entre distancias
    pregunta = "phishing enlace correo clave remitente"
    una_fuente = {**CASO_SIMPLE, "pregunta": pregunta, "fuentes_esperadas": ["playbook_phishing"]}
    dos_fuentes = {**una_fuente, "id": "dos", "fuentes_esperadas": ["playbook_phishing", "guia_respaldos"]}
    casos = _casos_en(tmp_path, [una_fuente, dos_fuentes, CASO_SIN_RESPUESTA])
    assert consultar_corpus.main(_argv(tmp_path, directorio, casos, "--k", "1")) == 0
    uno, dos, tres = _leer(tmp_path)["resultados"]
    assert [r["documento"] for r in uno["recuperados"]] == ["playbook_phishing"] and uno["acierto_recuperacion"] is True
    # con k=1 solo se recupera una de las dos esperadas: "todas" da False, "alguna" daría True
    assert [r["documento"] for r in dos["recuperados"]] == ["playbook_phishing"] and dos["acierto_recuperacion"] is False
    assert tres["acierto_recuperacion"] is None


def test_28_truncado_sigue_a_stop_reason_y_la_consola_avisa(tmp_path, monkeypatch, capsys, embeddings):
    directorio, _, _ = _montar(tmp_path, monkeypatch, embeddings, lambda n, prompt: _mensaje(stop_reason="max_tokens" if n == 1 else "end_turn"))
    assert consultar_corpus.main(_argv(tmp_path, directorio, _casos_en(tmp_path, [CASO_SIMPLE, {**CASO_SIMPLE, "id": "c1b"}]))) == 0
    assert [r["truncado"] for r in _leer(tmp_path)["resultados"]] == [True, False]
    assert capsys.readouterr().out.count("AVISO: respuesta cortada por max_tokens.") == 1


def test_29_el_error_del_modelo_lleva_el_tipo_y_se_recorta_a_300_caracteres(tmp_path, monkeypatch, embeddings):
    def guion(n, prompt):
        raise ValueError("x" * 500)

    directorio, _, _ = _montar(tmp_path, monkeypatch, embeddings, guion)
    assert consultar_corpus.main(_argv(tmp_path, directorio, _casos_en(tmp_path, [CASO_SIMPLE]))) == 1
    assert _leer(tmp_path)["resultados"][0]["error"] == "ValueError: " + "x" * 300


@pytest.mark.parametrize("abreviada", ["--conf", "--tope", "--sal"])
def test_30_las_abreviaturas_de_opciones_se_rechazan(tmp_path, monkeypatch, embeddings, abreviada):
    directorio, modelo, fabrica = _montar(tmp_path, monkeypatch, embeddings)
    salida_json = str(tmp_path / "salida.json")
    argv = {
        "--conf": ["--conf", "--directorio", str(directorio), "--salida", salida_json],
        "--tope": ["--confirmar-api-real", "--directorio", str(directorio), "--salida", salida_json, "--tope", "0.01"],
        "--sal": ["--confirmar-api-real", "--directorio", str(directorio), "--sal", salida_json],
    }[abreviada]
    with pytest.raises(SystemExit) as salida:
        consultar_corpus.main(argv)
    assert salida.value.code == 2
    assert _sin_peticiones(embeddings, modelo, fabrica)


def test_31_la_tilde_se_expande_en_directorio_y_salida_sin_crear_una_carpeta_tilde(tmp_path, monkeypatch, embeddings):
    directorio, _, _ = _montar(tmp_path, monkeypatch, embeddings)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert consultar_corpus.main(["--confirmar-api-real", "--directorio", "~/indice", "--salida", "~/salida.json"]) == 0
    assert _leer(tmp_path)["preguntas_procesadas"] == 8
    assert not (cwd / "~").exists() and list(cwd.iterdir()) == []


def test_32_los_valores_por_defecto_son_rutas_absolutas_bajo_la_raiz_y_no_dependen_del_cwd(tmp_path, monkeypatch):
    antes = consultar_corpus._crear_parser().parse_args([])
    monkeypatch.chdir(tmp_path)
    args = consultar_corpus._crear_parser().parse_args([])
    assert vars(args) == vars(antes)
    for ruta in (args.salida, args.directorio, args.casos):
        assert os.path.isabs(ruta) and Path(ruta).is_relative_to(RAIZ)
    assert args.salida == str(RAIZ / "docs" / "resultados_recuperacion.json")
