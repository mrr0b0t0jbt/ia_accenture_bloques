import ast
import logging
import math
import json
import operator
import re

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from supervision import RegistroAprobaciones, denegar_siempre

logger = logging.getLogger(__name__)

DOCUMENTACION = {
    "reintentos": (
        "with_retry espera 2 s y duplica la espera en cada intento (2, 4, 8...), con un tope de 30 s "
        "y un jitter de hasta 10 %. Solo se reintenta TimeoutError; un 401 o 403 falla de inmediato."
    ),
    "costo": (
        "El costo se calcula con los tokens de la respuesta y los precios por millon de tokens del proyecto "
        "(Claude Sonnet 5.5: USD 2 de entrada y USD 10 de salida)."
    ),
    "topes": (
        "Los scripts con API real exigen --confirmar-api-real y llevan un tope de gasto "
        "y un maximo de llamadas forzados en el codigo."
    ),
}
CVES_LABORATORIO = {
    "CVE-2099-10001": {
        "resumen": "Fuga de datos ficticia en el panel de pruebas de una aplicacion de laboratorio.",
        "severidad": "media (CVSS 5.3)",
        "mitigacion": "Desactivar el panel de pruebas ficticio en el laboratorio.",
    },
    "CVE-2099-10002": {
        "resumen": "Deserializacion insegura ficticia en la cola de mensajes de una aplicacion de laboratorio.",
        "severidad": "critica (CVSS 9.8)",
        "mitigacion": "Aceptar solo mensajes firmados en la cola ficticia del laboratorio.",
    },
}
MAX_LONGITUD_EXPRESION = 100
MAX_LONGITUD_CVE_REFLEJADO = 64
_ERROR_EXPRESION = "Expresión no permitida."
_OPERADORES_BINARIOS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}
_OPERADORES_UNARIOS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _evaluar_nodo(nodo: ast.AST) -> int | float:
    """Evalúa un nodo del árbol y rechaza lo que no sea aritmética permitida."""
    # type() y no isinstance(): bool es subclase de int y no se admite
    if isinstance(nodo, ast.Constant) and type(nodo.value) in (int, float):
        return nodo.value
    if isinstance(nodo, ast.BinOp) and type(nodo.op) in _OPERADORES_BINARIOS:
        return _OPERADORES_BINARIOS[type(nodo.op)](_evaluar_nodo(nodo.left), _evaluar_nodo(nodo.right))
    if isinstance(nodo, ast.UnaryOp) and type(nodo.op) in _OPERADORES_UNARIOS:
        return _OPERADORES_UNARIOS[type(nodo.op)](_evaluar_nodo(nodo.operand))
    raise ValueError(_ERROR_EXPRESION)


def evaluar_expresion(expresion: str) -> int | float:
    """Evalúa una expresión aritmética recorriendo su árbol, sin ejecutar código."""
    expresion = expresion.strip()
    if not expresion or len(expresion) > MAX_LONGITUD_EXPRESION:
        raise ValueError(_ERROR_EXPRESION)
    return _evaluar_nodo(ast.parse(expresion, mode="eval").body)


@tool
def buscar_en_documentacion(consulta: str) -> str:
    """Úsala para preguntas sobre cómo funciona esta miniaplicación: reintentos, costo y topes."""
    consulta_normalizada = consulta.lower()
    for clave, texto in DOCUMENTACION.items():
        if clave in consulta_normalizada:
            return texto
    return "No se encontró información sobre esa consulta en la documentación."


@tool
def calcular(expresion: str) -> str:
    """Úsala para preguntas que requieren un cálculo aritmético con + - * / // % y paréntesis."""
    try:
        resultado = evaluar_expresion(expresion)
    except ValueError as error:
        return str(error)
    except ZeroDivisionError:
        return "No se puede dividir entre cero."
    except Exception as error:
        return f"Error al evaluar la expresión: {error}"
    if isinstance(resultado, float) and not math.isfinite(resultado):
        return "Resultado fuera de rango."
    # sobre 2**53 un float ya no representa enteros exactos: int() inventaría dígitos de ruido
    if isinstance(resultado, float) and resultado.is_integer() and abs(resultado) < 2**53:
        return str(int(resultado))
    return str(resultado)


class ArgumentosConsultaCve(BaseModel):
    cve_id: str = Field(description="Identificador del CVE de laboratorio, por ejemplo CVE-2099-10001.")
    incluir_mitigacion: bool = Field(default=False, description="Si es True, agrega la mitigación a la respuesta.")


@tool("consultar_cve_de_laboratorio", args_schema=ArgumentosConsultaCve)
def consultar_cve(cve_id: str, incluir_mitigacion: bool = False) -> str:
    """Úsala para consultar el resumen, la severidad y opcionalmente la mitigación de un CVE de laboratorio."""
    try:
        cve_id = cve_id.strip().upper()
        cve = CVES_LABORATORIO.get(cve_id)
        if cve is None:
            reflejado = cve_id[:MAX_LONGITUD_CVE_REFLEJADO] + ("…" if len(cve_id) > MAX_LONGITUD_CVE_REFLEJADO else "")
            return f"No se encontró el CVE de laboratorio {reflejado}."
        respuesta = f"{cve_id}: {cve['resumen']} Severidad: {cve['severidad']}."
        if incluir_mitigacion:
            respuesta += f" Mitigación: {cve['mitigacion']}"
        return respuesta
    except Exception:
        logger.exception("Error al consultar el CVE de laboratorio")
        return "Error al consultar el CVE de laboratorio."


# bajo: solo lectura; medio: escritura reversible; alto: escritura irreversible o con impacto ("medio" no se usa todavía)
RIESGO = {
    "buscar_en_documentacion": "bajo",
    "calcular": "bajo",
    "consultar_cve_de_laboratorio": "bajo",
    "aislar_host_de_laboratorio": "alto",
}
MAX_LONGITUD_MOTIVO = 200
MAX_LONGITUD_HOST = 100
_PATRON_HOST_LABORATORIO = re.compile(r"(?=.{1,100}$)[a-z0-9-]+(\.[a-z0-9-]+)*\.ejemplo\.test")


def limpiar_texto(texto: str, maximo: int, vacio: str) -> str:
    """Deja el texto del modelo en una sola línea imprimible de a lo más `maximo` caracteres."""
    saneado = "".join(c if c.isprintable() else " " for c in texto)
    return " ".join(saneado.split())[:maximo].strip() or vacio


def limpiar_motivo(motivo: str) -> str:
    """Deja el motivo del modelo en una sola línea imprimible de a lo más 200 caracteres."""
    return limpiar_texto(motivo, MAX_LONGITUD_MOTIVO, "(sin motivo)")


def construir_herramientas(aprobador=None, registro=None) -> list:
    """Devuelve las herramientas del agente; la de riesgo alto consulta al aprobador y registra su decisión."""
    aprobador = denegar_siempre if aprobador is None else aprobador
    registro = RegistroAprobaciones() if registro is None else registro

    @tool
    def aislar_host_de_laboratorio(host: str, motivo: str) -> str:
        """Úsala solo para aislar de la red un host de laboratorio (dominio .ejemplo.test) por un incidente; es una acción de riesgo alto que requiere aprobación humana."""
        host = host.strip().lower()
        if not _PATRON_HOST_LABORATORIO.fullmatch(host):
            # un intento sobre un host ajeno es una señal de seguridad: se registra como no aprobado, sin pedir aprobación
            registro.registrar(
                "aislar_host_de_laboratorio",
                {
                    "host": limpiar_texto(host, MAX_LONGITUD_HOST, "(host vacío)"),
                    "motivo": limpiar_motivo(motivo),
                    "rechazo": "host no permitido",
                },
                False,
            )
            return "Host no permitido: solo se pueden aislar hosts de laboratorio (*.ejemplo.test)."
        motivo = limpiar_motivo(motivo)
        # json.dumps cita el texto del modelo para que no se confunda con el resto del mensaje
        aprobado = aprobador(
            f"El agente quiere aislar el host {host}. Motivo (texto del modelo, no verificado): "
            f"{json.dumps(motivo, ensure_ascii=False)}. ¿Aprobar? (s/n): "
        )
        registro.registrar("aislar_host_de_laboratorio", {"host": host, "motivo": motivo}, aprobado)
        if not aprobado:
            return "Aislamiento no aprobado por el supervisor humano."
        # SIMULACIÓN: no toca ninguna red, proceso ni archivo; la acción real no existe
        return f"Host {host} aislado (simulado: no se modificó ninguna red)."

    return [buscar_en_documentacion, calcular, consultar_cve, aislar_host_de_laboratorio]


HERRAMIENTAS = construir_herramientas()
