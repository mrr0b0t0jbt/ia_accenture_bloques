"""Supervisión humana del despliegue de un prompt con langgraph (Bloque 32, Unidad 2.7).

El grafo ENVUELVE a supervision.py: proponer (comprueba la cadena del registro y fija la huella pendiente), supervisar
(interrupt: el grafo se detiene hasta que alguien decide) y desplegar (anexa la versión SOLO si la reanudación fue True y
la cadena no cambió desde la propuesta). El aprobador de supervision.py (esperar_aprobacion, denegar_siempre) decide;
este módulo solo guarda el estado entre la propuesta y la decisión. Sin red ni claves.

Checkpointer: InMemorySaver por defecto (el estado vive en el proceso); con ruta_sqlite se usa SqliteSaver y la decisión
puede tomarse en otro proceso. El payload de la interrupción NO lleva el texto del prompt; el checkpoint SÍ lo guarda
(el texto debe estar para desplegarlo), igual que el registro: tratar la base SQLite como el registro.
"""
import sqlite3
import uuid
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import registro_prompts


class Estado(TypedDict, total=False):
    ruta_registro: str
    nombre: str
    texto: str
    autor: str
    nota: str
    huella_pendiente: str
    huella_base: str | None
    version_propuesta: int
    aprobado: bool
    resultado: dict


def proponer(estado: Estado) -> dict:
    """Valida la cadena del registro y fija la huella pendiente y la última huella registrada; no escribe nada."""
    entradas = registro_prompts.leer_registro(estado["ruta_registro"])
    problemas = registro_prompts.problemas_del_registro(entradas)
    if problemas:
        raise ValueError(f"el registro tiene problemas: {problemas[0]}")
    anterior = registro_prompts.ultima(entradas, estado["nombre"])
    huella_pendiente = registro_prompts.huella_prompt(estado["texto"])
    if anterior is not None and anterior["huella"] == huella_pendiente:
        raise ValueError("sin cambios: la huella ya es la de la ultima version")
    return {
        "huella_pendiente": huella_pendiente,
        "huella_base": None if anterior is None else anterior["huella"],
        "version_propuesta": 1 if anterior is None else anterior["version"] + 1,
    }


def supervisar(estado: Estado) -> dict:
    """Detiene el grafo con un payload sin el texto; solo un True exacto aprueba, cualquier otro valor rechaza."""
    decision = interrupt(
        {
            "motivo": "aprobar el despliegue de una version de prompt",
            "nombre": estado["nombre"],
            "version_propuesta": estado["version_propuesta"],
            "huella_pendiente": estado["huella_pendiente"],
            "huella_base": estado["huella_base"],
        }
    )
    return {"aprobado": decision is True}


def desplegar(estado: Estado) -> dict:
    """Anexa la versión si fue aprobada y la cadena sigue donde estaba al proponer; si no, no escribe."""
    if not estado["aprobado"]:
        return {"resultado": {"estado": "rechazado"}}
    anterior = registro_prompts.ultima(registro_prompts.leer_registro(estado["ruta_registro"]), estado["nombre"])
    if (None if anterior is None else anterior["huella"]) != estado["huella_base"]:
        return {"resultado": {"estado": "desactualizado"}}
    entrada = registro_prompts.registrar_version(estado["ruta_registro"], estado["nombre"], estado["texto"], estado["autor"], estado["nota"])
    return {"resultado": {"estado": "desplegado", "version": entrada["version"], "huella": entrada["huella"]}}


class SupervisionDespliegue:
    """Grafo de supervisión del despliegue de prompts; usar close() (o with) al terminar si hay ruta_sqlite."""

    def __init__(self, ruta_registro: str, ruta_sqlite: str | None = None) -> None:
        self.ruta_registro = ruta_registro
        self._conexion = None
        if ruta_sqlite is None:
            checkpointer = InMemorySaver()
        else:
            try:
                self._conexion = sqlite3.connect(ruta_sqlite, check_same_thread=False)
                checkpointer = SqliteSaver(self._conexion)
            except (sqlite3.Error, OSError) as error:
                self.close()
                raise ValueError(f"no se pudo abrir la base de checkpoints ({type(error).__name__})") from None
        grafo = StateGraph(Estado)
        grafo.add_node("proponer", proponer)
        grafo.add_node("supervisar", supervisar)
        grafo.add_node("desplegar", desplegar)
        grafo.add_edge(START, "proponer")
        grafo.add_edge("proponer", "supervisar")
        grafo.add_edge("supervisar", "desplegar")
        grafo.add_edge("desplegar", END)
        self._app = grafo.compile(checkpointer=checkpointer)

    def close(self) -> None:
        if self._conexion is not None:
            self._conexion.close()
            self._conexion = None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def _config(thread_id: str) -> dict:
        return {"configurable": {"thread_id": thread_id}}

    def iniciar(self, nombre: str, texto: str, autor: str, nota: str) -> dict:
        """Propone una versión y deja el grafo esperando la decisión; devuelve {thread_id, pendiente}."""
        for valor, campo in ((nombre, "nombre"), (texto, "texto"), (autor, "autor"), (nota, "nota")):
            if not isinstance(valor, str) or not valor.strip():
                raise ValueError(f"{campo} invalido: debe ser un texto no vacio")
        thread_id = uuid.uuid4().hex
        entrada = {"ruta_registro": self.ruta_registro, "nombre": nombre, "texto": texto, "autor": autor, "nota": nota}
        resultado = self._app.invoke(entrada, config=self._config(thread_id))
        return {"thread_id": thread_id, "pendiente": resultado["__interrupt__"][0].value}

    def reanudar(self, thread_id: str, aprobado: bool) -> dict:
        """Reanuda con la decisión; solo True exacto despliega. Un thread_id desconocido o ya terminado lanza ValueError."""
        if not isinstance(thread_id, str) or not self._app.get_state(self._config(thread_id)).next:
            raise ValueError("thread_id desconocido o ya resuelto")
        return self._app.invoke(Command(resume=aprobado is True), config=self._config(thread_id))["resultado"]

    def ejecutar(self, nombre: str, texto: str, autor: str, nota: str, aprobador) -> dict:
        """Camino completo: propone, pregunta con el aprobador de supervision.py y reanuda con su respuesta."""
        propuesta = self.iniciar(nombre, texto, autor, nota)
        pendiente = propuesta["pendiente"]
        mensaje = f"Desplegar {pendiente['nombre']} v{pendiente['version_propuesta']} (huella {pendiente['huella_pendiente']})? [s/N] "
        return self.reanudar(propuesta["thread_id"], aprobador(mensaje))
