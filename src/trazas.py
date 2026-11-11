import json
import os
from datetime import datetime, timezone

from herramientas_agente import RIESGO

RUTA_TRAZAS = "trazas/trazas_agente.jsonl"


def registrar_traza(
    session_id: str, entrada: str, resultado: dict, ruta: str = RUTA_TRAZAS, aprobaciones: list | None = None
) -> None:
    """Agrega al JSONL una línea con la entrada, los pasos del agente, su respuesta y las aprobaciones."""
    traza = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "entrada": entrada,
        "acciones": [
            {"herramienta": accion.tool, "argumentos": accion.tool_input, "observacion": str(observacion)}
            for accion, observacion in resultado.get("intermediate_steps", [])
        ],
        "respuesta_final": resultado["output"],
        "aprobaciones": list(aprobaciones or []),
    }
    directorio = os.path.dirname(ruta)
    if directorio:
        os.makedirs(directorio, exist_ok=True)
    # una sola llamada write con el salto de línea para que cada traza quede en su propia línea
    with open(ruta, "a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(traza, ensure_ascii=False, default=str) + "\n")


def leer_trazas(ruta: str = RUTA_TRAZAS) -> list[dict]:
    """Lee el JSONL de trazas; una línea que no es JSON válido lanza ValueError con su número."""
    registros = []
    with open(ruta, encoding="utf-8") as archivo:
        for numero, linea in enumerate(archivo, start=1):
            if not linea.strip():
                continue
            try:
                registros.append(json.loads(linea))
            except json.JSONDecodeError as error:
                raise ValueError(f"La línea {numero} de {ruta} no es JSON válido: {error}") from error
    return registros


def marcar_repeticiones_sospechosas(registros: list[dict]) -> list[str]:
    """Marca los registros donde una misma herramienta se usó dos veces seguidas (candidatos a revisar)."""
    marcas = []
    for registro in registros:
        herramientas = [accion["herramienta"] for accion in registro["acciones"]]
        for anterior, actual in zip(herramientas, herramientas[1:]):
            if anterior == actual:
                marcas.append(f"Revisar: {registro['session_id']} usó '{actual}' dos veces seguidas")
                break
    return marcas


def marcar_riesgo_alto_sin_aprobacion(registros: list[dict], riesgo: dict | None = None) -> list[str]:
    """Marca las acciones de riesgo alto que no tienen ninguna aprobación registrada (aprobada o no)."""
    riesgo = RIESGO if riesgo is None else riesgo
    marcas = []
    for registro in registros:
        decididas = {aprobacion["herramienta"] for aprobacion in registro["aprobaciones"]}
        for accion in registro["acciones"]:
            herramienta = accion["herramienta"]
            if riesgo.get(herramienta) == "alto" and herramienta not in decididas:
                marcas.append(
                    f"Revisar: {registro['session_id']} usó la herramienta de riesgo alto '{herramienta}' sin registro de aprobación"
                )
    return marcas
