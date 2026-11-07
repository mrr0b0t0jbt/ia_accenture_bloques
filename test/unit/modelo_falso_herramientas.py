from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


def pedir_herramienta(nombre: str, argumentos: dict) -> AIMessage:
    """Devuelve el mensaje con el que el modelo pide una herramienta."""
    return AIMessage(content="", tool_calls=[{"name": nombre, "args": argumentos, "id": f"llamada-{nombre}"}])


class ModeloFalsoHerramientas(BaseChatModel):
    """Modelo sin red que responde según un guion y guarda las entradas que recibió."""

    guion: list[AIMessage]
    entradas: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        """Guarda la entrada y devuelve el siguiente mensaje del guion."""
        mensaje = self.guion[len(self.entradas)]
        self.entradas.append(messages)
        return ChatResult(generations=[ChatGeneration(message=mensaje)])

    @property
    def _llm_type(self) -> str:
        return "falso-con-herramientas"

    def bind_tools(self, tools, **kwargs) -> "ModeloFalsoHerramientas":
        """Ignora las herramientas: lo que pide el modelo ya viene en el guion."""
        return self

    def texto_recibido(self, llamada: int) -> str:
        """Une el contenido de los mensajes que recibió el modelo en una llamada."""
        return "\n".join(str(mensaje.content) for mensaje in self.entradas[llamada])
