import datetime
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from chromadb.api.shared_system_client import SharedSystemClient

import embeddings_gemini
import indexacion
import medicion_rag
import modelo_langchain
from embeddings_falsos import EmbeddingsPorTema

RAIZ = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location("medir_latencia_costo", RAIZ / "scripts" / "medir_latencia_costo.py")
medir = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(medir)

TEMAS = {
    "parcheo": ["parche", "parchear", "parcheo", "plazo", "vulnerabilidad", "cvss"],
    "respaldos": ["respaldo", "respaldos", "copia", "restauracion", "rpo"],
    "phishing": ["phishing", "correo", "enlace", "clave", "remitente"],
}
# cada documento cabe en un solo chunk (200 caracteres)
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
CASOS = [
    {"id": "c1", "pregunta": "plazo parche vulnerabilidad cvss"},
    {"id": "c2", "pregunta": "respaldo copia restauracion rpo"},
    {"id": "c3", "pregunta": "phishing enlace correo clave"},
]
CLAVES_REGISTRO = [
    "timestamp", "corrida", "id_caso", "pregunta", "k", "max_tokens", "tiempos_ms", "tiempo_total_ms", "reintentos_embed",
    "ids_recuperados", "caracteres_prompt", "tokens_prompt_aprox", "tokens", "costo_estimado", "truncado", "sin_datos_de_uso",
    "admite_falta_de_contexto", "respuesta", "error",
]
HORA_1 = datetime.datetime(2026, 11, 16, 19, 30, 0, tzinfo=datetime.timezone.utc)
HORA_2 = datetime.datetime(2026, 11, 16, 20, 15, 5, tzinfo=datetime.timezone.utc)
TEXTO_LIMITE = embeddings_gemini.describir_error_api(RuntimeError("429 RESOURCE_EXHAUSTED"))[1]


@pytest.fixture(autouse=True)
def _libera_chroma():
    """Cierra los clientes de Chroma al terminar: sin esto los descriptores abiertos superan 1024 y rompen select() en test_supervision."""
    yield
    gc.collect()
    SharedSystemClient.clear_system_cache()


@pytest.fixture
def embeddings():
    return EmbeddingsPorTema(TEMAS)


def _mensaje(entrada, salida=50):
    uso = {"input_tokens": entrada, "output_tokens": salida, "total_tokens": entrada + salida}
    return SimpleNamespace(content="Respuesta [1].", usage_metadata=uso, response_metadata={})


class ModeloGuionado:
    """Modelo falso: invoke n llama a guion(n, prompt), que devuelve un mensaje o lanza (por defecto la entrada es len(prompt) // 4)."""

    def __init__(self, guion=None):
        self.guion = guion or (lambda n, prompt: _mensaje(len(prompt) // 4))
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return self.guion(len(self.prompts), prompt)


def _preparar(tmp_path, monkeypatch, embeddings, guion=None, casos=CASOS, indexar=True):
    """Indexa el corpus de prueba en tmp_path y parchea las fábricas, la hora y el sleep; devuelve el entorno."""
    (tmp_path / "casos.json").write_text(json.dumps(casos), encoding="utf-8")
    directorio = tmp_path / "indice"
    if indexar:
        indexacion.indexar(CORPUS, embeddings, str(directorio))
    embeddings.llamadas_documentos = embeddings.llamadas_consulta = 0
    modelo, dormidas = ModeloGuionado(guion), []
    fabrica = MagicMock(return_value=modelo)
    monkeypatch.setattr(embeddings_gemini, "crear_embeddings", lambda: embeddings)
    monkeypatch.setattr(modelo_langchain, "crear_modelo", fabrica)
    monkeypatch.setattr(medir, "_ahora", lambda: HORA_1)
    monkeypatch.setattr(indexacion.time, "sleep", dormidas.append)
    return SimpleNamespace(tmp=tmp_path, directorio=directorio, modelo=modelo, fabrica=fabrica, dormidas=dormidas, salida=tmp_path / "registros.jsonl")


def _argv(entorno, *extra, salida=None, casos=None):
    return [
        "--confirmar-api-real", "--casos", str(casos or entorno.tmp / "casos.json"), "--directorio", str(entorno.directorio),
        "--salida", str(entorno.salida if salida is None else salida), *extra,
    ]


def _codigo(argv) -> int:
    """Código de salida de main; un rechazo de argparse (SystemExit) también cuenta."""
    try:
        return medir.main(argv)
    except SystemExit as salida:
        return salida.code


def _lineas(ruta) -> list[dict]:
    return [json.loads(linea) for linea in Path(ruta).read_text(encoding="utf-8").splitlines()]


def _sin_peticiones(entorno, embeddings) -> bool:
    return embeddings.llamadas_consulta == 0 and entorno.modelo.prompts == []


def _guion_embedding(monkeypatch, embeddings, falla):
    """Reemplaza embed_query por una función que cuenta las peticiones reales y llama a falla(n, texto, original)."""
    original, peticiones = embeddings.embed_query, []

    def embed_query(texto):
        peticiones.append(texto)
        return falla(len(peticiones), texto, original)

    monkeypatch.setattr(embeddings, "embed_query", embed_query)
    return peticiones


# S1
def test_s1_sin_confirmar_devuelve_2_sin_importar_nada_ni_leer_casos(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "embeddings_gemini", None)
    monkeypatch.setitem(sys.modules, "modelo_langchain", None)
    assert medir.main(["--casos", str(tmp_path / "no_existe.json")]) == 2
    captura = capsys.readouterr()
    assert "--confirmar-api-real" in captura.err and "no se pudieron leer" not in captura.err
    assert list(tmp_path.iterdir()) == []


# S2
@pytest.mark.parametrize(
    "malo",
    [
        ["--ks", "0"], ["--ks", "7"], ["--ks", "x"], ["--ks", "3,3"], ["--ks", "1,2,3"], ["--ks", ""],
        ["--max-tokens", "0"], ["--max-tokens", "1025"], ["--tope-usd", "0"], ["--tope-usd", "-1"], ["--tope-usd", "0.09"],
    ],
    ids=lambda malo: "_".join(malo) or "vacio",
)
def test_s2_argumentos_invalidos_devuelven_2_sin_peticiones(tmp_path, monkeypatch, embeddings, malo):
    entorno = _preparar(tmp_path, monkeypatch, embeddings)
    assert _codigo(_argv(entorno, *malo)) == 2
    assert _sin_peticiones(entorno, embeddings) and not entorno.salida.exists()


def test_s2b_la_abreviatura_conf_no_activa_la_confirmacion(tmp_path, monkeypatch, embeddings):
    entorno = _preparar(tmp_path, monkeypatch, embeddings)
    assert _codigo(["--conf", "--directorio", str(entorno.directorio), "--salida", str(entorno.salida)]) == 2
    assert _sin_peticiones(entorno, embeddings) and not entorno.salida.exists()


# S3
def _largo():
    return [{"id": "c1", "pregunta": "a" * 501}]


@pytest.mark.parametrize(
    "contenido",
    [
        None,
        "{no es json",
        {"id": "x", "pregunta": "p"},
        [{"id": "x"}],
        [{"id": "x", "pregunta": "p"}, {"id": "x", "pregunta": "q"}],
        [{"id": f"c{n}", "pregunta": "p"} for n in range(7)],
        _largo(),
    ],
    ids=["inexistente", "json_invalido", "no_lista", "sin_pregunta", "id_repetido", "mas_de_max_casos", "pregunta_larga"],
)
def test_s3_casos_invalidos_devuelven_1_sin_peticiones(tmp_path, monkeypatch, capsys, embeddings, contenido):
    entorno = _preparar(tmp_path, monkeypatch, embeddings)
    ruta = tmp_path / "otros_casos.json"
    if isinstance(contenido, str):
        ruta.write_text(contenido, encoding="utf-8")
    elif contenido is not None:
        ruta.write_text(json.dumps(contenido), encoding="utf-8")
    assert medir.main(_argv(entorno, casos=ruta)) == 1
    assert "Error:" in capsys.readouterr().err
    assert _sin_peticiones(entorno, embeddings) and not entorno.salida.exists()


# S4
def test_s4_tope_de_llamadas_antes_de_gastar(tmp_path, monkeypatch, capsys, embeddings):
    monkeypatch.setattr(medir, "MAX_LLAMADAS", 5)
    entorno = _preparar(tmp_path, monkeypatch, embeddings)
    assert medir.main(_argv(entorno, "--ks", "3,6")) == 1  # 3 casos x 2 ks = 6 > 5
    err = capsys.readouterr().err
    assert "Error:" in err and "llamadas" in err
    assert _sin_peticiones(entorno, embeddings) and not entorno.salida.exists()


# S5
@pytest.mark.parametrize(
    "caso", ["padre_inexistente", "igual_casos", "dentro_indice", "ajeno", "directorio", "vacia", "no_escribible"]
)
def test_s5_salida_insegura_se_rechaza_antes_de_gastar(tmp_path, monkeypatch, capsys, embeddings, caso):
    entorno = _preparar(tmp_path, monkeypatch, embeddings)
    ajeno = tmp_path / "ajeno.json"
    ajeno.write_text('{"a": 1}', encoding="utf-8")
    (tmp_path / "carpeta").mkdir()
    (tmp_path / "solo_lectura").mkdir()
    destino = {
        "padre_inexistente": tmp_path / "no_hay" / "s.jsonl",
        "igual_casos": tmp_path / "casos.json",
        "dentro_indice": entorno.directorio / "s.jsonl",
        "ajeno": ajeno,
        "directorio": tmp_path / "carpeta",
        "vacia": "",
        "no_escribible": tmp_path / "solo_lectura" / "s.jsonl",
    }[caso]
    if caso == "no_escribible":
        if os.geteuid() == 0:
            pytest.skip("root ignora los permisos")
        (tmp_path / "solo_lectura").chmod(0o555)
    intactos = {ruta: ruta.read_bytes() for ruta in (tmp_path / "casos.json", ajeno)}
    try:
        assert medir.main(_argv(entorno, salida=destino)) == 1
    finally:
        (tmp_path / "solo_lectura").chmod(0o755)
    assert "Error:" in capsys.readouterr().err
    assert {ruta: ruta.read_bytes() for ruta in intactos} == intactos
    assert _sin_peticiones(entorno, embeddings) and not (entorno.directorio / "s.jsonl").exists()


def test_s5b_una_salida_existente_de_0_bytes_se_acepta(tmp_path, monkeypatch, embeddings):
    entorno = _preparar(tmp_path, monkeypatch, embeddings)
    entorno.salida.write_bytes(b"")
    assert medir.main(_argv(entorno, "--ks", "3")) == 0
    assert len(_lineas(entorno.salida)) == 3


def test_s5c_una_salida_previa_valida_se_acepta_y_se_anexa(tmp_path, monkeypatch, embeddings):
    entorno = _preparar(tmp_path, monkeypatch, embeddings)
    previo = {"timestamp": "2026-01-01T00:00:00+00:00", "tiempos_ms": {}}
    medicion_rag.anexar_registro(str(entorno.salida), previo)
    assert medir.main(_argv(entorno, "--ks", "3")) == 0
    lineas = _lineas(entorno.salida)
    assert len(lineas) == 4 and lineas[0] == previo


# S6
@pytest.mark.parametrize("estado", ["inexistente", "vacio"])
def test_s6_indice_inexistente_o_vacio(tmp_path, monkeypatch, capsys, embeddings, estado):
    entorno = _preparar(tmp_path, monkeypatch, embeddings, indexar=False)
    if estado == "vacio":
        indexacion.abrir_vectorstore(embeddings, str(entorno.directorio), indexacion.COLECCION_POR_DEFECTO)
    assert medir.main(_argv(entorno)) == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "indexar" in err
    assert entorno.directorio.exists() is (estado == "vacio")
    assert _sin_peticiones(entorno, embeddings) and not entorno.salida.exists()


# S7
def test_s7_flujo_feliz(tmp_path, monkeypatch, capsys, embeddings):
    entorno = _preparar(tmp_path, monkeypatch, embeddings)
    assert medir.main(_argv(entorno, "--ks", "3,6")) == 0
    registros = _lineas(entorno.salida)
    assert len(registros) == 6 and all(list(r) == CLAVES_REGISTRO for r in registros)
    assert {r["corrida"] for r in registros} == {"20261116T193000Z"}
    assert embeddings.llamadas_consulta == 6 and len(entorno.modelo.prompts) == 6
    assert [(r["id_caso"], r["k"]) for r in registros] == [(c["id"], k) for c in CASOS for k in (3, 6)]
    entorno.fabrica.assert_called_once_with(max_tokens=512)
    out = capsys.readouterr().out
    assert "input_tokens crece con k: si" in out and "generacion es la etapa mas lenta en" in out
    assert "Consultas de embedding (incluidos reintentos): 6" in out and "Llamadas a Claude: 6" in out
    assert list(tmp_path.glob("*.tmp")) == [] and list(tmp_path.glob("**/*.tmp")) == []


# S8
def test_s8_una_segunda_corrida_anexa(tmp_path, monkeypatch, embeddings):
    entorno = _preparar(tmp_path, monkeypatch, embeddings)
    assert medir.main(_argv(entorno, "--ks", "3,6")) == 0
    primera = entorno.salida.read_bytes()
    monkeypatch.setattr(medir, "_ahora", lambda: HORA_2)
    assert medir.main(_argv(entorno, "--ks", "3,6")) == 0
    contenido = entorno.salida.read_bytes()
    registros = _lineas(entorno.salida)
    assert len(registros) == 12 and contenido.startswith(primera)
    assert {r["corrida"] for r in registros[:6]} == {"20261116T193000Z"} and {r["corrida"] for r in registros[6:]} == {"20261116T201505Z"}


# S9
def test_s9_tope_de_gasto(tmp_path, monkeypatch, capsys, embeddings):
    entorno = _preparar(tmp_path, monkeypatch, embeddings, guion=lambda n, prompt: _mensaje(1000, 50))
    assert medir.main(_argv(entorno, "--ks", "3,6", "--tope-usd", "0.003")) == 0
    assert len(entorno.modelo.prompts) == 2 and len(_lineas(entorno.salida)) == 2
    assert "tope de gasto" in capsys.readouterr().out


# S10
def test_s10_el_tope_de_consultas_se_fuerza_en_ejecucion(tmp_path, monkeypatch, capsys, embeddings):
    monkeypatch.setattr(medir, "MAX_CONSULTAS_EMBEDDING", 4)
    entorno = _preparar(tmp_path, monkeypatch, embeddings)
    por_pregunta = {}

    def falla(n, texto, original):
        por_pregunta[texto] = por_pregunta.get(texto, 0) + 1
        if por_pregunta[texto] <= 2:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return original(texto)

    peticiones = _guion_embedding(monkeypatch, embeddings, falla)
    assert medir.main(_argv(entorno, "--ks", "3")) == 1
    captura = capsys.readouterr()
    assert "Error:" in captura.err and "tope" in captura.err and "Traceback" not in captura.err
    assert len(peticiones) == 4 and len(_lineas(entorno.salida)) == 1
    assert "c1 k=3" in captura.out and "Consultas de embedding (incluidos reintentos): 4" in captura.out
    assert entorno.dormidas == [5, 10, 5]


# S11
def test_s11_un_429_persistente_no_deja_registro(tmp_path, monkeypatch, capsys, embeddings):
    entorno = _preparar(tmp_path, monkeypatch, embeddings)

    def siempre(n, texto, original):
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    peticiones = _guion_embedding(monkeypatch, embeddings, siempre)
    assert medir.main(_argv(entorno)) == 1
    err = capsys.readouterr().err
    assert TEXTO_LIMITE in err and "Traceback" not in err
    assert len(peticiones) == 4 and entorno.dormidas == [5, 10, 20] and not entorno.salida.exists()


# S12
def test_s12_clave_rechazada(tmp_path, monkeypatch, capsys, embeddings):
    entorno = _preparar(tmp_path, monkeypatch, embeddings)

    def rechazada(n, texto, original):
        raise RuntimeError("API key not valid")

    peticiones = _guion_embedding(monkeypatch, embeddings, rechazada)
    assert medir.main(_argv(entorno)) == 1
    err = capsys.readouterr().err
    assert "Error:" in err and "rechazó la clave" in err and "Traceback" not in err
    assert len(peticiones) == 1 and entorno.dormidas == [] and not entorno.salida.exists()


# S13
def test_s13_una_falla_del_modelo_no_detiene_la_corrida(tmp_path, monkeypatch, capsys, embeddings):
    def guion(n, prompt):
        if n == 2:
            raise RuntimeError("fallo")
        return _mensaje(len(prompt) // 4)

    entorno = _preparar(tmp_path, monkeypatch, embeddings, guion=guion)
    assert medir.main(_argv(entorno, "--ks", "3,6")) == 0
    registros = _lineas(entorno.salida)
    assert len(registros) == 6 and registros[1]["error"] == "RuntimeError: fallo" and registros[1]["tokens"] is None
    out = capsys.readouterr().out
    assert "ERROR: RuntimeError: fallo" in out and "con error: 1" in out
    assert "k=3 (3 registros)" in out and "k=6 (2 registros)" in out


# S14
def test_s14_sin_datos_de_uso_detiene_la_corrida(tmp_path, monkeypatch, capsys, embeddings):
    entorno = _preparar(tmp_path, monkeypatch, embeddings, guion=lambda n, prompt: _mensaje(0 if n == 2 else 100, 0 if n == 2 else 10))
    assert medir.main(_argv(entorno, "--ks", "3,6")) == 1
    registros = _lineas(entorno.salida)
    assert len(registros) == 2 and registros[1]["sin_datos_de_uso"] is True and registros[0]["sin_datos_de_uso"] is False
    assert "sin uso no se controla el gasto" in capsys.readouterr().err


# S15
def test_s15_ctrl_c_en_el_modelo(tmp_path, monkeypatch, capsys, embeddings):
    def guion(n, prompt):
        if n == 3:
            raise KeyboardInterrupt
        return _mensaje(len(prompt) // 4)

    entorno = _preparar(tmp_path, monkeypatch, embeddings, guion=guion)
    assert medir.main(_argv(entorno, "--ks", "3,6")) == 130
    captura = capsys.readouterr()
    assert len(_lineas(entorno.salida)) == 2 and "Interrumpido" in captura.err and "Traceback" not in captura.err
    assert "Llamadas a Claude: 2" in captura.out


def test_s15b_ctrl_c_en_el_embedding(tmp_path, monkeypatch, capsys, embeddings):
    entorno = _preparar(tmp_path, monkeypatch, embeddings)

    def interrumpe(n, texto, original):
        if n == 3:
            raise KeyboardInterrupt
        return original(texto)

    _guion_embedding(monkeypatch, embeddings, interrumpe)
    assert medir.main(_argv(entorno, "--ks", "3,6")) == 130
    captura = capsys.readouterr()
    assert len(_lineas(entorno.salida)) == 2 and "Interrumpido" in captura.err and "Traceback" not in captura.err
    assert "Llamadas a Claude: 2" in captura.out


# S16
def test_s16_la_consola_limpia_los_controles_del_id(tmp_path, monkeypatch, capsys, embeddings):
    entorno = _preparar(tmp_path, monkeypatch, embeddings, casos=[{"id": "uno\x1b[31mrojo\nsegunda", "pregunta": "plazo parche cvss"}])
    assert medir.main(_argv(entorno, "--ks", "3")) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out and "uno [31mrojo segunda k=3 embed" in out


# S17
def test_s17_la_salida_expande_la_tilde(tmp_path, monkeypatch, embeddings):
    entorno = _preparar(tmp_path, monkeypatch, embeddings)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert medir.main(_argv(entorno, "--ks", "3", salida="~/mis_registros.jsonl")) == 0
    assert len(_lineas(tmp_path / "mis_registros.jsonl")) == 3 and not Path("~").exists()


# S18
def test_s18_no_indexa_ni_escribe_en_el_indice(tmp_path, monkeypatch, embeddings):
    entorno = _preparar(tmp_path, monkeypatch, embeddings)
    antes = indexacion.abrir_vectorstore(embeddings, str(entorno.directorio), indexacion.COLECCION_POR_DEFECTO)._collection.count()
    assert medir.main(_argv(entorno, "--ks", "3,6")) == 0
    despues = indexacion.abrir_vectorstore(embeddings, str(entorno.directorio), indexacion.COLECCION_POR_DEFECTO)._collection.count()
    assert embeddings.llamadas_documentos == 0 and antes == despues == len(CORPUS)
