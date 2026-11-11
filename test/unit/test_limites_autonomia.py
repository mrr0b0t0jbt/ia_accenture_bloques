import inspect
from pathlib import Path

from herramientas_agente import HERRAMIENTAS, RIESGO

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
