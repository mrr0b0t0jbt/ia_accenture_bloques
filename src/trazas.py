import json
import os
from datetime import datetime, timezone

from herramientas_agente import RIESGO, limpiar_texto

RUTA_TRAZAS = "trazas/trazas_agente.jsonl"
MAX_LONGITUD_MARCA = 100


def registrar_traza(
    session_id: str,
    entrada: str,
    resultado: dict,
    ruta: str = RUTA_TRAZAS,
    aprobaciones: list | None = None,
    interrumpido: bool = False,
) -> None:
    """Agrega al JSONL una línea con la entrada, los pasos del agente, su respuesta y las aprobaciones; privada (0600)."""
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
    if interrumpido:
        traza["interrumpido"] = True
    directorio = os.path.dirname(ruta)
    if directorio:
        os.makedirs(directorio, mode=0o700, exist_ok=True)
    # el modo solo se aplica al crear el archivo; uno que ya existe no se toca. Las trazas guardan las entradas completas
    descriptor = os.open(ruta, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    # una sola llamada write con el salto de línea para que cada traza quede en su propia línea
    with os.fdopen(descriptor, "a", encoding="utf-8") as archivo:
        archivo.write(json.dumps(traza, ensure_ascii=False, default=str) + "\n")


def _problema_de_forma(registro) -> str | None:
    """Devuelve el motivo corto por el que un registro de traza no tiene la forma esperada, o None."""
    if not isinstance(registro, dict):
        return "no es un objeto"
    acciones, aprobaciones = registro.get("acciones"), registro.get("aprobaciones")
    if not isinstance(acciones, list) or not all(isinstance(a, dict) and isinstance(a.get("herramienta"), str) for a in acciones):
        return "acciones no es una lista de objetos con herramienta"
    if not isinstance(aprobaciones, list) or not all(isinstance(a, dict) for a in aprobaciones):
        return "aprobaciones no es una lista de objetos"
    return None


def leer_trazas(ruta: str = RUTA_TRAZAS) -> list[dict]:
    """Lee el JSONL de trazas; una línea que no es JSON válido o no tiene la forma esperada lanza ValueError con su número."""
    registros = []
    with open(ruta, encoding="utf-8") as archivo:
        for numero, linea in enumerate(archivo, start=1):
            if not linea.strip():
                continue
            try:
                registro = json.loads(linea)
            except json.JSONDecodeError as error:
                raise ValueError(f"La línea {numero} de {ruta} no es JSON válido: {error}") from error
            problema = _problema_de_forma(registro)
            if problema:
                raise ValueError(f"Línea {numero}: la traza no tiene la forma esperada ({problema}).")
            registros.append(registro)
    return registros


def _seguro(valor) -> str:
    """Deja un dato de la traza (que puede venir de un archivo ajeno) en una línea imprimible para la consola."""
    return limpiar_texto(str(valor), MAX_LONGITUD_MARCA, "(vacío)")


def _sesion(registro: dict) -> str:
    return _seguro(registro.get("session_id", "(sin sesion)"))


def marcar_repeticiones_sospechosas(registros: list[dict]) -> list[str]:
    """Marca los registros donde una misma herramienta se usó dos veces seguidas (candidatos a revisar)."""
    marcas = []
    for registro in registros:
        herramientas = [accion["herramienta"] for accion in registro["acciones"]]
        for anterior, actual in zip(herramientas, herramientas[1:]):
            if anterior == actual:
                marcas.append(f"Revisar: {_sesion(registro)} usó '{_seguro(actual)}' dos veces seguidas")
                break
    return marcas


def marcar_riesgo_alto_sin_aprobacion(registros: list[dict], riesgo: dict | None = None) -> list[str]:
    """Marca las acciones de riesgo alto que no tienen ninguna aprobación registrada (aprobada o no)."""
    riesgo = RIESGO if riesgo is None else riesgo
    marcas = []
    for registro in registros:
        decididas = {aprobacion.get("herramienta") for aprobacion in registro["aprobaciones"]}
        for accion in registro["acciones"]:
            herramienta = accion["herramienta"]
            if herramienta not in riesgo:
                marcas.append(f"Revisar: {_sesion(registro)} usó la herramienta '{_seguro(herramienta)}' sin nivel de riesgo definido")
            elif riesgo[herramienta] == "alto" and herramienta not in decididas:
                marcas.append(
                    f"Revisar: {_sesion(registro)} usó la herramienta de riesgo alto '{_seguro(herramienta)}' sin registro de aprobación"
                )
    return marcas
