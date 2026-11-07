import logging
import string

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough, RunnableSerializable

logger = logging.getLogger(__name__)

RESPUESTA_GENERICA = "Gracias por tu mensaje. Un miembro del equipo te responderá pronto."
MENSAJE_ERROR = "No pudimos procesar tu mensaje en este momento. Intenta de nuevo más tarde."
respuesta_generica = RunnableLambda(lambda _: RESPUESTA_GENERICA, name="respuesta_generica")
CATEGORIAS = ("incidente", "vulnerabilidad", "consulta")
_RELLENO_CATEGORIA = string.whitespace + string.punctuation + "¿¡«»“”‘’"

PLANTILLA_CLASIFICACION = PromptTemplate.from_template(
    "Clasifica el siguiente mensaje en una sola palabra: incidente, vulnerabilidad o consulta. "
    "Responde solo con esa palabra.\n\nMensaje:\n{mensaje}"
)
PLANTILLAS_RUTAS = {
    "incidente": PromptTemplate.from_template(
        "Eres parte de un equipo de respuesta a incidentes. Indica cómo contener y priorizar este incidente, "
        "con los pasos inmediatos a seguir. Responde en un máximo de 10 pasos numerados y menos de 350 palabras."
        "\n\nMensaje:\n{mensaje}"
    ),
    "vulnerabilidad": PromptTemplate.from_template(
        "Eres analista de vulnerabilidades. Evalúa la severidad de este hallazgo y propón su remediación. "
        "Responde en un máximo de 8 pasos o secciones y menos de 400 palabras."
        "\n\nMensaje:\n{mensaje}"
    ),
    "consulta": PromptTemplate.from_template(
        "Eres parte de un equipo de seguridad. Responde esta consulta con precisión.\n\nMensaje:\n{mensaje}"
    ),
}


def _normalizar_categoria(categoria: str) -> str:
    """Quita mayúsculas, y espacios y puntuación de los extremos, de la categoría que devolvió el modelo."""
    normalizada = categoria.lower().strip(_RELLENO_CATEGORIA)
    if normalizada not in CATEGORIAS:
        logger.warning("Categoria inesperada: %r", categoria)
    return normalizada


def construir_paso_clasificacion(modelo) -> RunnableSerializable:
    """Devuelve la chain que clasifica un mensaje en una categoría normalizada."""
    normalizar = RunnableLambda(_normalizar_categoria, name="normalizar_categoria")
    return PLANTILLA_CLASIFICACION | modelo | StrOutputParser() | normalizar


def construir_rutas(modelo) -> dict[str, RunnableSerializable]:
    """Devuelve una chain por categoría, cada una con su propio prompt."""
    return {
        categoria: plantilla | modelo | StrOutputParser()
        for categoria, plantilla in PLANTILLAS_RUTAS.items()
    }


def construir_cadena_con_enrutamiento(
    paso_clasificacion, cadena_incidente=None, cadena_vulnerabilidad=None, cadena_consulta=None
) -> RunnableSerializable:
    """Agrega la categoría al mensaje y lo envía a su ruta; consulta es la ruta por defecto."""
    bifurcacion = RunnableBranch(
        (lambda x: x["categoria"] == "incidente", cadena_incidente or RunnablePassthrough()),
        (lambda x: x["categoria"] == "vulnerabilidad", cadena_vulnerabilidad or RunnablePassthrough()),
        cadena_consulta or RunnablePassthrough(),
    )
    return RunnablePassthrough.assign(categoria=paso_clasificacion) | bifurcacion


def _registrar_fallo(cadena) -> RunnableLambda:
    """Envuelve una ruta para registrar su excepción y relanzarla hacia el respaldo siguiente."""

    def invocar(entrada, config):
        try:
            return cadena.invoke(entrada, config)
        except Exception:
            logger.exception("Fallo una ruta")
            raise

    return RunnableLambda(invocar, name="ruta_con_registro")


def construir_ruta_segura(cadena, respaldos=None) -> RunnableSerializable:
    """Envuelve una ruta con sus respaldos en orden, registrando sus fallos, y la respuesta genérica al final."""
    return _registrar_fallo(cadena).with_fallbacks(
        [*(_registrar_fallo(respaldo) for respaldo in respaldos or []), respuesta_generica]
    )


def construir_rutas_seguras(modelo, modelo_secundario=None) -> dict[str, RunnableSerializable]:
    """Devuelve las tres rutas con respaldo; el primero es la misma plantilla con modelo_secundario, si lo hay."""
    rutas = construir_rutas(modelo)
    respaldos = construir_rutas(modelo_secundario) if modelo_secundario else None
    return {
        categoria: construir_ruta_segura(ruta, [respaldos[categoria]] if respaldos else None)
        for categoria, ruta in rutas.items()
    }


def procesar_mensaje_seguro(cadena, mensaje: str, config=None) -> str:
    """Ejecuta el flujo y, si falla lo que los respaldos no cubren, devuelve un texto fijo."""
    try:
        return cadena.invoke({"mensaje": mensaje}, config=config)
    except Exception:
        logger.exception("Error procesando mensaje")
        return MENSAJE_ERROR


def construir_flujo(modelo, modelo_secundario=None) -> RunnableSerializable:
    """Arma la clasificación y las tres rutas seguras."""
    rutas = construir_rutas_seguras(modelo, modelo_secundario)
    return construir_cadena_con_enrutamiento(
        construir_paso_clasificacion(modelo),
        cadena_incidente=rutas["incidente"],
        cadena_vulnerabilidad=rutas["vulnerabilidad"],
        cadena_consulta=rutas["consulta"],
    )
