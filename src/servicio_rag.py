"""Servicio RAG de punta a punta para el Gate 2: una sola entrada que recupera, filtra, genera, cita y registra.

Fachada sobre consulta_rag: no duplica la logica. Corta ANTES de llamar al modelo si no queda contexto limpio, y
reemplaza la respuesta del modelo si no trae ninguna cita valida: asi cada respuesta que sale con texto de modelo
cita al menos una fuente real. Sin red ni claves mientras se inyecten vectorstore y modelo.
"""
import logging

import recuperacion
from consulta_rag import preparar_consulta, responder_consulta
from registro_estructurado import con_trace_id, registrar_evento, validar_precios

SIN_CONTEXTO_VERIFICADO = "No tengo informacion suficiente y verificada para responder esto."


def consultar(pregunta: str, vectorstore, modelo, k: int = recuperacion.K_POR_DEFECTO,
              logger: logging.Logger | None = None, precios=None) -> dict:
    """Devuelve {respuesta, fuentes, trace_id, estado, truncado}.

    estado: respondida, sin_contexto_confiable, sin_cita_valida o sin_respuesta.
    truncado: True si el modelo cortó la respuesta por max_tokens; False si no se invocó el modelo (sin_contexto_confiable).
    """
    validar_precios(precios)
    with con_trace_id() as trace_id:
        preparacion = preparar_consulta(pregunta, vectorstore, k, logger)
        if not preparacion["citas"]:
            registrar_evento(logger, logging.WARNING, "sin_contexto_confiable",
                             n_recuperados=len(preparacion["resultados"]), n_descartados=len(preparacion["descartados"]))
            return {"respuesta": SIN_CONTEXTO_VERIFICADO, "fuentes": [], "trace_id": trace_id,
                    "estado": "sin_contexto_confiable", "truncado": False}
        resultado = responder_consulta(preparacion, modelo, logger, precios)
        truncado = resultado["truncado"]
        if resultado["admite_falta_de_contexto"]:
            return {"respuesta": resultado["respuesta"], "fuentes": [], "trace_id": trace_id, "estado": "sin_respuesta",
                    "truncado": truncado}
        validas = resultado["citas"]["validas"]
        if not validas:
            registrar_evento(logger, logging.WARNING, "respuesta_sin_cita_valida",
                             n_citas_invalidas=len(resultado["citas"]["invalidas"]))
            return {"respuesta": SIN_CONTEXTO_VERIFICADO, "fuentes": [], "trace_id": trace_id, "estado": "sin_cita_valida",
                    "truncado": truncado}
        fuentes = sorted({fuente for _, fuente in validas})
        lineas = [f"[{numero}] {fuente}" for numero, fuente in validas]
        texto = resultado["respuesta"].rstrip() + "\n\nFuentes:\n" + "\n".join(lineas)
        return {"respuesta": texto, "fuentes": fuentes, "trace_id": trace_id, "estado": "respondida", "truncado": truncado}


def ejecutar_servicio(pregunta: str, vectorstore, modelo, k: int = recuperacion.K_POR_DEFECTO,
                      logger: logging.Logger | None = None, precios=None) -> tuple[str, list[str]]:
    """Contrato del Gate 2: devuelve (respuesta, fuentes)."""
    salida = consultar(pregunta, vectorstore, modelo, k, logger, precios)
    return salida["respuesta"], salida["fuentes"]
