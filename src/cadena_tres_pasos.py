from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSerializable

# Paso 1: extrae lo esencial del aviso, sin desarrollarlo.
PLANTILLA_EXTRAER = PromptTemplate.from_template(
    "Extrae los 3 puntos clave del siguiente aviso de seguridad, uno por línea, sin desarrollarlos.\n\n"
    "Aviso:\n{texto}"
)
# Paso 2: traduce los puntos para alguien sin conocimientos técnicos.
PLANTILLA_SIMPLIFICAR = PromptTemplate.from_template(
    "Reescribe estos puntos en lenguaje simple, para alguien sin conocimientos técnicos.\n\n"
    "Puntos:\n{puntos}"
)
# Paso 3: da el formato final de checklist.
PLANTILLA_FORMATEAR = PromptTemplate.from_template(
    "Formatea este texto como checklist de remediación en viñetas markdown, máximo 3 líneas, sin introducción.\n\n"
    "Texto:\n{texto_simple}"
)


def _envolver_puntos(puntos: str) -> dict:
    """Envuelve la salida del paso 1 en el dict que espera el paso 2."""
    return {"puntos": puntos}


def _envolver_texto_simple(texto_simple: str) -> dict:
    """Envuelve la salida del paso 2 en el dict que espera el paso 3."""
    return {"texto_simple": texto_simple}


def construir_pasos(modelo) -> tuple[RunnableSerializable, RunnableSerializable, RunnableSerializable]:
    """Devuelve las tres sub-chains por separado, sin ejecutarlas."""
    return (
        PLANTILLA_EXTRAER | modelo | StrOutputParser(),
        PLANTILLA_SIMPLIFICAR | modelo | StrOutputParser(),
        PLANTILLA_FORMATEAR | modelo | StrOutputParser(),
    )


def construir_cadena(modelo) -> RunnableSerializable:
    """Une los tres pasos en una sola chain, sin ejecutarla."""
    extraer, simplificar, formatear = construir_pasos(modelo)
    envolver_puntos = RunnableLambda(_envolver_puntos, name="envolver_puntos")
    envolver_texto_simple = RunnableLambda(_envolver_texto_simple, name="envolver_texto_simple")
    return extraer | envolver_puntos | simplificar | envolver_texto_simple | formatear
