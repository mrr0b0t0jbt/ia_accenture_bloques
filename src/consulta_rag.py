"""Consulta RAG de punta a punta, instrumentada (Bloque 31, Unidad 2.7): preparar, responder y procesar.

Cada paso registra eventos SIN contenido (solo longitud, huella e ids). Un documento sospechoso de inyección se descarta,
se responde con el resto y la consulta queda marcada con requiere_revision; no se frena la respuesta. Sin red ni claves.
"""
import logging

import recuperacion
from defensa_inyeccion import HUELLA_PROMPT, construir_mensajes, filtrar_documentos, fuente_saneada
from registro_estructurado import con_trace_id, registrar_consumo, registrar_evento, resumen_de_texto, validar_precios


def preparar_consulta(pregunta: str, vectorstore, k: int = recuperacion.K_POR_DEFECTO, logger: logging.Logger | None = None) -> dict:
    """Recupera, filtra los documentos sospechosos y arma los mensajes; usa el trace_id que esté activo."""
    registrar_evento(logger, logging.INFO, "consulta_recibida", pregunta=resumen_de_texto(pregunta), k=k)
    try:
        resultados = recuperacion.recuperar(vectorstore, pregunta, k)
    except ValueError:
        registrar_evento(logger, logging.ERROR, "consulta_rechazada", tipo="ValueError")
        raise
    except Exception as error:
        registrar_evento(logger, logging.ERROR, "recuperacion_fallo", tipo=type(error).__name__)
        raise
    documentos = [documento for documento, _ in resultados]
    registrar_evento(
        logger,
        logging.INFO,
        "documentos_recuperados",
        n_documentos=len(documentos),
        fuentes=[fuente_saneada(documento) for documento in documentos],
        distancia_minima=resultados[0][1] if resultados else None,
    )
    limpios, descartados = filtrar_documentos(documentos)
    for descartado in descartados:
        registrar_evento(logger, logging.WARNING, "documento_descartado_por_inyeccion", **descartado)
    requiere_revision = bool(descartados)
    registrar_evento(
        logger, logging.INFO, "contexto_filtrado",
        n_limpios=len(limpios), n_descartados=len(descartados), requiere_revision=requiere_revision,
    )
    contexto, citas = recuperacion.formatear_contexto(limpios)
    return {
        "mensajes": construir_mensajes(pregunta, contexto or recuperacion.SIN_CONTEXTO),
        "citas": citas,
        "resultados": resultados,
        "descartados": descartados,
        "requiere_revision": requiere_revision,
        "huella_prompt": HUELLA_PROMPT,
    }


def responder_consulta(preparacion: dict, modelo, logger: logging.Logger | None = None, precios=None) -> dict:
    """Llama al modelo con los mensajes preparados, registra el consumo y verifica las citas."""
    validar_precios(precios)  # un precio invalido no se descubre despues de pagar la llamada
    mensajes = preparacion["mensajes"]
    registrar_evento(logger, logging.INFO, "llamada_modelo", n_mensajes=len(mensajes), huella_prompt=preparacion["huella_prompt"])
    try:
        mensaje = modelo.invoke(mensajes)
    except Exception as error:  # KeyboardInterrupt y demás BaseException pasan sin capturar
        registrar_evento(logger, logging.ERROR, "modelo_fallo", tipo=type(error).__name__)
        raise
    consumo = registrar_consumo(logger, mensaje, precios)
    texto = recuperacion.texto_de_respuesta(mensaje)
    citas = recuperacion.extraer_citas(texto, preparacion["citas"])
    admite = recuperacion.admite_falta_de_contexto(texto)
    truncado = (getattr(mensaje, "response_metadata", None) or {}).get("stop_reason") == "max_tokens"
    registrar_evento(
        logger, logging.INFO, "respuesta_generada",
        respuesta=resumen_de_texto(texto),
        n_citas_validas=len(citas["validas"]),
        n_citas_invalidas=len(citas["invalidas"]),
        sin_citas=citas["sin_citas"],
        admite_falta_de_contexto=admite,
        truncado=truncado,
    )
    return {"respuesta": texto, "citas": citas, "admite_falta_de_contexto": admite, "consumo": consumo, "truncado": truncado, "mensaje": mensaje}


def procesar_consulta(
    pregunta: str, vectorstore, modelo, k: int = recuperacion.K_POR_DEFECTO, logger: logging.Logger | None = None, precios=None
) -> dict:
    """Prepara y responde una consulta bajo un trace_id nuevo; devuelve la respuesta con el id y el resultado del filtro."""
    validar_precios(precios)
    with con_trace_id() as trace_id:
        preparacion = preparar_consulta(pregunta, vectorstore, k, logger)
        resultado = responder_consulta(preparacion, modelo, logger, precios)
    return {
        **resultado,
        "trace_id": trace_id,
        "descartados": preparacion["descartados"],
        "requiere_revision": preparacion["requiere_revision"],
        "huella_prompt": preparacion["huella_prompt"],
    }
