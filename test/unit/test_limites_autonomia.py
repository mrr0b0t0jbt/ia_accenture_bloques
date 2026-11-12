import inspect
import re
from pathlib import Path

from herramientas_agente import HERRAMIENTAS, RIESGO
from supervision import MAX_APROBACIONES_POR_CORRIDA, MAX_APROBACIONES_POR_TURNO

RAIZ = Path(__file__).parents[2]
DOCUMENTO = RAIZ / "docs" / "limites_autonomia_agente.md"
SCRIPT = RAIZ / "scripts" / "ejecutar_agente.py"
TITULOS = [
    "## Herramientas y nivel de riesgo",
    "## Herramientas que requieren aprobación humana",
    "## Revisión de trazas",
    "## Qué hacer ante un comportamiento inesperado",
]


def _texto() -> str:
    return DOCUMENTO.read_text(encoding="utf-8")


def _seccion(titulo: str) -> list[str]:
    """Devuelve las líneas de la sección, desde su título hasta el siguiente título de nivel 2."""
    lineas, dentro, seccion = _texto().splitlines(), False, []
    for linea in lineas:
        if linea.startswith("## "):
            dentro = linea == titulo
        elif dentro:
            seccion.append(linea)
    return seccion


def _riesgo_alto() -> list:
    return [herramienta for herramienta in HERRAMIENTAS if RIESGO[herramienta.name] == "alto"]


def test_el_documento_existe_con_los_cuatro_titulos_de_la_plantilla():
    lineas = _texto().splitlines()
    assert all(titulo in lineas for titulo in TITULOS)


def test_cada_herramienta_aparece_con_su_nivel_de_riesgo_en_su_seccion():
    seccion = _seccion(TITULOS[0])
    for herramienta in HERRAMIENTAS:
        nivel = RIESGO[herramienta.name]
        assert any(herramienta.name in linea and nivel in linea for linea in seccion), herramienta.name


def test_la_seccion_de_aprobacion_nombra_las_de_riesgo_alto_y_ninguna_de_riesgo_bajo():
    seccion = "\n".join(_seccion(TITULOS[1]))
    assert _riesgo_alto()
    for herramienta in HERRAMIENTAS:
        assert (herramienta.name in seccion) is (RIESGO[herramienta.name] == "alto"), herramienta.name


def test_el_codigo_de_cada_herramienta_de_riesgo_alto_llama_a_un_aprobador():
    for herramienta in _riesgo_alto():
        assert "aprobador(" in inspect.getsource(herramienta.func), herramienta.name


def test_el_documento_y_el_script_comparten_las_opciones_de_aprobacion_y_trazas():
    documento, script = _texto(), SCRIPT.read_text(encoding="utf-8")
    for texto in ("--aprobacion", "--tiempo-aprobacion", "scripts/revisar_trazas.py"):
        assert texto in documento, texto
    for opcion in ("--aprobacion", "--tiempo-aprobacion"):
        assert f'"{opcion}"' in script, opcion


def test_el_documento_dice_que_la_herramienta_de_riesgo_alto_esta_simulada():
    assert "simulada" in _texto()


def test_el_documento_fija_los_topes_de_aprobacion_y_las_frases_clave_del_codigo():
    documento = _texto()
    por_turno, por_corrida = MAX_APROBACIONES_POR_TURNO, MAX_APROBACIONES_POR_CORRIDA
    maximo_script = int(re.search(r"^TIEMPO_APROBACION_MAXIMO = (\d+)$", SCRIPT.read_text(encoding="utf-8"), re.M)[1])
    minutos = por_corrida * maximo_script // 60
    for frase in (
        f"máximo {por_turno} aprobaciones solicitadas por turno y {por_corrida} por corrida",
        f"{por_corrida} aprobaciones x {maximo_script} s = {minutos} minutos",
        "Tiempo agotado: no aprobado.",
        "0600",
        "0700",
        "Interrumpido por la persona. Resultados parciales guardados.",
        "tope de aprobaciones",
        "ya denegado en este turno",
    ):
        assert frase in documento, frase
    # el documento ya no aconseja "desactivar" la herramienta ni dice que el aprobador personalizado solo falla con errores de lectura
    assert "Desactivar la herramienta" not in documento
    assert "dejar el aprobador" not in documento
    assert "distinta de un error de lectura" not in documento
