"""Agente con herramientas y memoria sobre respuestas REALES de Claude (Bloques 21 y 22).

Procesa los turnos de JSON/casos_agente.json y escribe los resultados en
docs/resultados_agente.json después de cada turno.

Cómo correrlo (desde la raíz del proyecto):

    python scripts/ejecutar_agente.py --confirmar-api-real

Hace llamadas REALES a la API, con costo; los reintentos internos del SDK no se cuentan como llamadas.
Con la aprobación por defecto "denegar", las acciones de riesgo alto nunca se ejecutan en una corrida desatendida.
Las trazas van a un directorio fuera de git (trazas/), porque guardan las entradas de las consultas.
"""
import argparse
import importlib.util
import logging
import os
import sys

_DIRECTORIO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DIRECTORIO, "..", "src"))

_spec = importlib.util.spec_from_file_location("ejecutar_cadenas", os.path.join(_DIRECTORIO, "ejecutar_cadenas.py"))
ejecutar_cadenas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ejecutar_cadenas)

logger = logging.getLogger(__name__)

TOPE_USD_MAXIMO = 0.10
MAX_TOKENS_MAXIMO = 2048
MAX_TOKENS_POR_DEFECTO = 1024
MAX_LLAMADAS = 30
MAX_TURNOS = 10
MAX_ITERACIONES = 4
MAX_CARACTERES_MENSAJE = 2000
MOTIVO_LLAMADAS = "limite de llamadas"
MOTIVO_TOPE = "tope de gasto"
SALIDA_MAX_ITERACIONES = "Agent stopped due to max iterations."
# igual a trazas.RUTA_TRAZAS (una prueba lo comprueba); no se importa aquí para no cargar langchain sin el flag
RUTA_TRAZAS_POR_DEFECTO = "trazas/trazas_agente.jsonl"
MODOS_APROBACION = ("denegar", "interactiva")
TIEMPO_APROBACION_POR_DEFECTO = 60
TIEMPO_APROBACION_MAXIMO = 300


class LimiteExcedido(Exception):
    """Se lanza antes de una llamada al modelo que superaría un límite de la corrida."""


def validar_casos_agente(casos) -> str | None:
    """Devuelve el primer problema de estructura de los turnos, o None si no hay ninguno."""
    # import diferido: herramientas_agente carga langchain_core, y aquí ya se pasó el flag
    from herramientas_agente import HERRAMIENTAS

    if not isinstance(casos, dict):
        return "el archivo de casos no es un objeto JSON."
    if not isinstance(casos.get("turnos"), list) or not casos["turnos"]:
        return "el archivo de casos no tiene la lista turnos con al menos un turno."
    nombres = {herramienta.name for herramienta in HERRAMIENTAS}
    for n, turno in enumerate(casos["turnos"]):
        if not isinstance(turno, dict):
            return f"turnos[{n}] no es un objeto."
        for campo in ("id", "sesion", "mensaje"):
            if not ejecutar_cadenas._es_texto_no_vacio(turno.get(campo)):
                return f"turnos[{n}] no tiene el campo {campo} válido."
        if len(turno["mensaje"]) > MAX_CARACTERES_MENSAJE:
            return f"turnos[{n}] tiene un mensaje de más de {MAX_CARACTERES_MENSAJE} caracteres."
        esperadas = turno.get("herramientas_esperadas")
        if not isinstance(esperadas, list) or not all(isinstance(nombre, str) for nombre in esperadas):
            return f"turnos[{n}] no tiene el campo herramientas_esperadas válido."
        for nombre in esperadas:
            if nombre not in nombres:
                return f"turnos[{n}] espera una herramienta inexistente: {nombre}."
    return None


def _crear_limitador(max_llamadas: int, tope_usd: float, costo_actual):
    """Crea el callback que frena la corrida antes de una llamada que superaría un límite."""
    # import diferido: sin el flag no se carga langchain
    from langchain_core.callbacks import BaseCallbackHandler

    class Limitador(BaseCallbackHandler):
        # sin raise_error, la librería registra la excepción del callback y la llamada al modelo sigue
        raise_error = True
        llamadas = 0
        motivo = None

        def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
            if self.llamadas >= max_llamadas:
                self.motivo = MOTIVO_LLAMADAS
            elif costo_actual() >= tope_usd:
                self.motivo = MOTIVO_TOPE
            if self.motivo:
                raise LimiteExcedido(self.motivo)
            self.llamadas += 1

    return Limitador()


def ejecutar_turnos(
    casos: dict,
    modelo,
    tope_usd: float,
    ruta_salida: str,
    max_llamadas: int = MAX_LLAMADAS,
    max_iteraciones: int = MAX_ITERACIONES,
    aprobador=None,
    ruta_trazas: str | None = None,
) -> dict:
    """Procesa los turnos en orden, mide su uso y corta la corrida al llegar a un límite."""
    from langchain_core.callbacks import UsageMetadataCallbackHandler

    from agente import AlmacenHistoriales, construir_agente_con_memoria, construir_ejecutor
    from api_claude_refactor import INPUT_PRICING, OUTPUT_PRICING
    from herramientas_agente import construir_herramientas
    from supervision import RegistroAprobaciones
    from trazas import registrar_traza

    # el medidor solo suma los mensajes cuyo response_metadata trae model_name; ChatAnthropic lo informa
    medidor = UsageMetadataCallbackHandler()

    def tokens_acumulados() -> tuple[int, int]:
        usos = medidor.usage_metadata.values()
        return sum(uso["input_tokens"] for uso in usos), sum(uso["output_tokens"] for uso in usos)

    def costo_de(tokens_entrada: int, tokens_salida: int) -> float:
        return tokens_entrada * INPUT_PRICING + tokens_salida * OUTPUT_PRICING

    def costo_actual() -> float:
        return costo_de(*tokens_acumulados())

    limitador = _crear_limitador(max_llamadas, tope_usd, costo_actual)
    almacen = AlmacenHistoriales()
    # sin streaming: con herramientas, ChatAnthropic haría streaming con output como lista de bloques, y esa lista rompe el historial
    # aprobador None = denegar_siempre: sin un humano que apruebe, la acción de riesgo alto no se ejecuta
    registro_aprobaciones = RegistroAprobaciones()
    herramientas = construir_herramientas(aprobador, registro_aprobaciones)
    ejecutor = construir_ejecutor(modelo, herramientas=herramientas, max_iterations=max_iteraciones, stream_runnable=False)
    agente = construir_agente_con_memoria(ejecutor, almacen)
    salida = {
        "proveedor": "claude",
        "nota_costo": ejecutar_cadenas.NOTA_COSTO,
        "resultados": [],
        "llamadas_totales": 0,
        "costo_total_estimado": 0.0,
        "cortado_por_tope": False,
        "cortado_por_llamadas": False,
        "cortado_por_falta_de_uso": False,
        "aprobaciones": [],
    }
    for turno in casos["turnos"]:
        if costo_actual() >= tope_usd:
            salida["cortado_por_tope"] = True
            break
        llamadas_antes = limitador.llamadas
        aprobaciones_antes = len(registro_aprobaciones.registros)
        entrada_antes, salida_antes = tokens_acumulados()
        detector = ejecutar_cadenas._crear_detector_de_corte()
        sesion = turno["sesion"]
        registro = {
            "id": turno["id"],
            "sesion": sesion,
            "mensaje": turno["mensaje"],
            "herramientas_esperadas": turno["herramientas_esperadas"],
        }
        config = {"callbacks": [medidor, limitador, detector], "configurable": {"session_id": sesion}}
        resultado = None
        try:
            resultado = agente.invoke({"entrada": turno["mensaje"]}, config=config)
            pasos = [
                {"herramienta": accion.tool, "entrada": accion.tool_input, "observacion": str(observacion)}
                for accion, observacion in resultado["intermediate_steps"]
            ]
            usadas = [paso["herramienta"] for paso in pasos]
            registro.update(
                respuesta=resultado["output"],
                herramientas_usadas=usadas,
                coincide_herramientas=usadas == turno["herramientas_esperadas"],
                pasos=pasos,
                hubo_fallo=resultado["output"] == SALIDA_MAX_ITERACIONES,
            )
        except LimiteExcedido as error:
            registro.update(error=str(error), hubo_fallo=True)
            salida["cortado_por_llamadas" if limitador.motivo == MOTIVO_LLAMADAS else "cortado_por_tope"] = True
        except Exception as error:
            # un turno fallido no corta los siguientes
            logger.exception("Error procesando turno")
            registro.update(error=str(error), hubo_fallo=True)
        todas = registro_aprobaciones.registros
        registro["aprobaciones"] = todas[aprobaciones_antes:]
        salida["aprobaciones"] = todas
        if ruta_trazas is not None and resultado is not None:
            try:
                registrar_traza(sesion, turno["mensaje"], resultado, ruta=ruta_trazas, aprobaciones=registro["aprobaciones"])
            except Exception:
                # un fallo al escribir la traza no marca el turno como error ni corta la corrida
                logger.exception("No se pudo escribir la traza del turno")
        entrada_ahora, salida_ahora = tokens_acumulados()
        tokens_entrada, tokens_salida = entrada_ahora - entrada_antes, salida_ahora - salida_antes
        llamadas = limitador.llamadas - llamadas_antes
        if llamadas and not tokens_entrada and not tokens_salida:
            logger.warning("Turno sin datos de uso: el costo estimado puede quedar subestimado")
            # sin datos de uso el tope de gasto no se puede controlar; un turno con error no activa este corte
            if "error" not in registro:
                salida["cortado_por_falta_de_uso"] = True
        registro.update(
            llamadas=llamadas,
            tokens_entrada=tokens_entrada,
            tokens_salida=tokens_salida,
            costo_estimado=costo_de(tokens_entrada, tokens_salida),
            truncado=detector.truncado,
            mensajes_en_sesion=almacen.mensajes(sesion),
        )
        salida["resultados"].append(registro)
        salida["llamadas_totales"] = limitador.llamadas
        salida["costo_total_estimado"] = costo_actual()
        # guardado incremental: si algo falla a mitad no se pierde lo ya pagado
        ejecutar_cadenas.guardar_resultados(salida, ruta_salida)
        if salida["cortado_por_llamadas"] or salida["cortado_por_tope"] or salida["cortado_por_falta_de_uso"]:
            break
    ejecutar_cadenas.guardar_resultados(salida, ruta_salida)
    return salida


def informar(salida: dict) -> None:
    """Muestra el resultado de cada turno sin ocultar diferencias, fallos ni cortes."""
    for n, resultado in enumerate(salida["resultados"], start=1):
        print(f"[Turno {n}] {resultado['id']}: {resultado.get('respuesta', 'sin respuesta')}")
        if "error" in resultado:
            print(f"  Error: {resultado['error']}")
        # un turno con error no trae traza: no se sabe qué herramientas alcanzó a usar
        if "herramientas_usadas" in resultado:
            usadas, esperadas = resultado["herramientas_usadas"], resultado["herramientas_esperadas"]
            print(f"  Herramientas: usadas {usadas}, esperadas {esperadas}")
            if usadas != esperadas:
                print("  DIFERENCIA de herramientas.")
        if resultado.get("truncado"):
            print("  AVISO: respuesta cortada por max_tokens.")
        if resultado.get("hubo_fallo") and "error" not in resultado:
            print("  Fallo: el agente se detuvo por el máximo de iteraciones.")
    if salida["cortado_por_tope"]:
        print("La ejecución se detuvo al alcanzar el tope de gasto.")
    if salida["cortado_por_llamadas"]:
        print("La ejecución se detuvo al alcanzar el límite de llamadas.")
    if salida.get("cortado_por_falta_de_uso"):
        print("La ejecución se detuvo: un turno no devolvió datos de uso y el tope de gasto no se puede controlar sin ellos.")
    aprobaciones = salida.get("aprobaciones")
    if aprobaciones:
        denegadas = sum(1 for decision in aprobaciones if not decision["aprobado"])
        print(f"Aprobaciones de riesgo alto: {len(aprobaciones)} ({denegadas} denegadas)")
    print(f"Llamadas totales: {salida['llamadas_totales']}")
    print(f"Costo total estimado: ${salida['costo_total_estimado']:.5f}")
    print(salida["nota_costo"])


def main(argv: list[str] | None = None) -> int:
    """Ejecuta el agente contra la API real solo con --confirmar-api-real."""
    parser = argparse.ArgumentParser(description="Agente de LangChain sobre respuestas REALES de Claude (con costo).")
    parser.add_argument("--casos", default="JSON/casos_agente.json")
    parser.add_argument("--salida", default="docs/resultados_agente.json")
    parser.add_argument("--tope-usd", type=float, default=TOPE_USD_MAXIMO)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_POR_DEFECTO)
    parser.add_argument("--aprobacion", choices=MODOS_APROBACION, default="denegar")
    parser.add_argument("--tiempo-aprobacion", type=float, default=TIEMPO_APROBACION_POR_DEFECTO)
    parser.add_argument("--trazas", default=RUTA_TRAZAS_POR_DEFECTO)
    parser.add_argument("--sin-trazas", action="store_true")
    parser.add_argument("--confirmar-api-real", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirmar_api_real:
        print("Este script hace llamadas REALES a la API de Claude, con costo. Para ejecutarlo pasa --confirmar-api-real.", file=sys.stderr)
        return 2
    if not args.salida:
        print("Error: --salida no puede estar vacía.", file=sys.stderr)
        return 1
    directorio = os.path.dirname(os.path.abspath(args.salida))
    if not os.path.isdir(directorio):
        print(f"Error: el directorio de --salida no existe: {directorio}", file=sys.stderr)
        return 1
    if os.path.isdir(args.salida):
        print(f"Error: --salida es un directorio: {args.salida}", file=sys.stderr)
        return 1
    if ejecutar_cadenas.es_el_mismo_archivo(args.salida, args.casos):
        print("Error: --salida no puede ser el mismo archivo que --casos.", file=sys.stderr)
        return 1
    if not os.access(args.salida if os.path.exists(args.salida) else directorio, os.W_OK):
        print(f"Error: no se puede escribir en --salida: {args.salida}", file=sys.stderr)
        return 1
    if not 0 < args.tope_usd <= TOPE_USD_MAXIMO:
        print(f"Error: --tope-usd debe ser mayor que 0 y como máximo {TOPE_USD_MAXIMO}.", file=sys.stderr)
        return 1
    if not 0 < args.max_tokens <= MAX_TOKENS_MAXIMO:
        print(f"Error: --max-tokens debe ser mayor que 0 y como máximo {MAX_TOKENS_MAXIMO}.", file=sys.stderr)
        return 1
    if not 0 < args.tiempo_aprobacion <= TIEMPO_APROBACION_MAXIMO:
        print(f"Error: --tiempo-aprobacion debe ser mayor que 0 y como máximo {TIEMPO_APROBACION_MAXIMO}.", file=sys.stderr)
        return 1
    casos = ejecutar_cadenas.cargar_casos(args.casos)
    problema = validar_casos_agente(casos)
    if problema:
        print(f"Error: {problema}", file=sys.stderr)
        return 1
    if len(casos["turnos"]) > MAX_TURNOS:
        print(f"Error: los casos tienen {len(casos['turnos'])} turnos; el máximo es {MAX_TURNOS}.", file=sys.stderr)
        return 1

    # import diferido: sin el flag no se carga langchain ni la API key
    from modelo_langchain import crear_modelo

    try:
        modelo = crear_modelo(max_tokens=args.max_tokens)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    aprobador = None
    if args.aprobacion == "interactiva":
        from supervision import crear_aprobador_interactivo

        aprobador = crear_aprobador_interactivo(args.tiempo_aprobacion)
    ruta_trazas = None if args.sin_trazas else args.trazas
    salida = ejecutar_turnos(casos, modelo, args.tope_usd, args.salida, aprobador=aprobador, ruta_trazas=ruta_trazas)
    informar(salida)
    if not any("error" not in r and not r.get("hubo_fallo") for r in salida["resultados"]):
        print("Error: ningún turno dio un resultado útil.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
