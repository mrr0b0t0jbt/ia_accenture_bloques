from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough, RunnableSerializable

PLANTILLA_CLASIFICACION = PromptTemplate.from_template(
    "Clasifica el siguiente mensaje en una sola palabra: incidente, vulnerabilidad o consulta. "
    "Responde solo con esa palabra.\n\nMensaje:\n{mensaje}"
)
PLANTILLAS_RUTAS = {
    "incidente": PromptTemplate.from_template(
        "Eres parte de un equipo de respuesta a incidentes. Indica cómo contener y priorizar este incidente, "
        "con los pasos inmediatos a seguir.\n\nMensaje:\n{mensaje}"
    ),
    "vulnerabilidad": PromptTemplate.from_template(
        "Eres analista de vulnerabilidades. Evalúa la severidad de este hallazgo y propón su remediación."
        "\n\nMensaje:\n{mensaje}"
    ),
    "consulta": PromptTemplate.from_template(
        "Eres parte de un equipo de seguridad. Responde esta consulta con precisión.\n\nMensaje:\n{mensaje}"
    ),
}


def _normalizar_categoria(categoria: str) -> str:
    """Quita espacios y mayúsculas de la categoría que devolvió el modelo."""
    return categoria.strip().lower()


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


def construir_flujo(modelo) -> RunnableSerializable:
    """Arma la clasificación y las tres rutas con un solo modelo."""
    rutas = construir_rutas(modelo)
    return construir_cadena_con_enrutamiento(
        construir_paso_clasificacion(modelo),
        cadena_incidente=rutas["incidente"],
        cadena_vulnerabilidad=rutas["vulnerabilidad"],
        cadena_consulta=rutas["consulta"],
    )
