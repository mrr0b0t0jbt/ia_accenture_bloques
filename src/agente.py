from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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
