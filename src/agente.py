from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

from herramientas_agente import HERRAMIENTAS

TEXTO_SISTEMA = (
    "Eres un asistente de seguridad de una miniaplicacion de apoyo a equipos tecnicos. "
    "Usa las herramientas disponibles solo cuando la pregunta las necesite; si no, responde directamente."
)


def construir_prompt() -> ChatPromptTemplate:
    """Devuelve el prompt del agente, con historial opcional y espacio para sus pasos."""
    return ChatPromptTemplate.from_messages(
        [
            ("system", TEXTO_SISTEMA),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{entrada}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )


def construir_ejecutor(modelo, herramientas=None, max_iterations: int = 4, **kwargs) -> AgentExecutor:
    """Arma el agente con sus herramientas y devuelve su ejecutor con la traza de pasos."""
    herramientas = HERRAMIENTAS if herramientas is None else herramientas
    return AgentExecutor(
        agent=create_tool_calling_agent(modelo, herramientas, construir_prompt()),
        tools=herramientas,
        return_intermediate_steps=True,
        max_iterations=max_iterations,
        verbose=False,
        **kwargs,
    )


class AlmacenHistoriales:
    """Guarda en memoria un historial de mensajes por sesión."""

    def __init__(self) -> None:
        self._historiales: dict[str, InMemoryChatMessageHistory] = {}

    def obtener(self, session_id: str) -> InMemoryChatMessageHistory:
        """Devuelve el historial de la sesión; lo crea en el primer uso."""
        return self._historiales.setdefault(session_id, InMemoryChatMessageHistory())

    def cerrar(self, session_id: str) -> None:
        """Elimina el historial de la sesión, si existe."""
        self._historiales.pop(session_id, None)

    def mensajes(self, session_id: str) -> int:
        """Devuelve la cantidad de mensajes de la sesión, sin crearla si no existe."""
        historial = self._historiales.get(session_id)
        return len(historial.messages) if historial else 0


def normalizar_salida(resultado: dict) -> dict:
    """Devuelve el resultado con el output como texto cuando el modelo lo entregó como lista de bloques."""
    salida = resultado["output"]
    if not isinstance(salida, list):
        return resultado
    texto = "".join(
        bloque if isinstance(bloque, str) else bloque.get("text", "")
        for bloque in salida
        if isinstance(bloque, str) or (isinstance(bloque, dict) and bloque.get("type") == "text")
    )
    return {**resultado, "output": texto}


def construir_agente_con_memoria(ejecutor, almacen: AlmacenHistoriales) -> RunnableWithMessageHistory:
    """Envuelve el ejecutor para que cada session_id conserve su historial."""
    # RunnableWithMessageHistory está deprecado; se usa porque la clase lo usa y es lo más simple sobre AgentExecutor
    return RunnableWithMessageHistory(
        ejecutor | RunnableLambda(normalizar_salida),
        almacen.obtener,
        input_messages_key="entrada",
        output_messages_key="output",
        history_messages_key="chat_history",
    )
