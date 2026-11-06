"""Chain de tres pasos y enrutamiento sobre respuestas REALES de Claude (Bloques 18 a 20).

Procesa los casos de JSON/casos_cadenas.json con LangChain (llamadas REALES a la
API, con costo) y escribe los resultados en docs/resultados_cadenas.json después
de cada caso.

Cómo correrlo (desde la raíz del proyecto):

    python scripts/ejecutar_cadenas.py --confirmar-api-real
"""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

logger = logging.getLogger(__name__)

TOPE_USD_MAXIMO = 0.10
MAX_TOKENS_MAXIMO = 2048
MAX_LLAMADAS = 12
LLAMADAS_POR_AVISO = 3
LLAMADAS_POR_MENSAJE = 2
NOTA_COSTO = "El costo es una estimación calculada con los tokens informados y los precios del proyecto; no es el cobro real."


def cargar_casos(ruta: str) -> dict:
    """Lee los avisos y mensajes desde un JSON UTF-8."""
    with open(ruta, encoding="utf-8") as archivo:
        return json.load(archivo)


def _es_texto_no_vacio(valor) -> bool:
    """Indica si el valor es una cadena con algo más que espacios."""
    return isinstance(valor, str) and bool(valor.strip())


def validar_casos(casos) -> str | None:
    """Devuelve el primer problema de estructura de los casos, o None si no hay ninguno."""
    # import diferido: enrutamiento carga langchain_core, y aquí ya se pasó el flag
    from enrutamiento import CATEGORIAS

    if not isinstance(casos, dict):
        return "el archivo de casos no es un objeto JSON."
    for clave in ("avisos", "mensajes"):
        if not isinstance(casos.get(clave), list):
            return f"el archivo de casos no tiene la lista {clave}."
    for n, aviso in enumerate(casos["avisos"]):
        if not isinstance(aviso, dict) or not _es_texto_no_vacio(aviso.get("texto")):
            return f"avisos[{n}] no tiene el campo texto válido."
    for n, mensaje in enumerate(casos["mensajes"]):
        if not isinstance(mensaje, dict) or not _es_texto_no_vacio(mensaje.get("mensaje")):
            return f"mensajes[{n}] no tiene el campo mensaje válido."
        if mensaje.get("esperada") not in CATEGORIAS:
            return f"mensajes[{n}] no tiene el campo esperada válido."
    return None


def guardar_resultados(salida: dict, ruta_salida: str) -> None:
    """Reescribe el archivo de salida completo con los resultados."""
    with open(ruta_salida, "w", encoding="utf-8") as archivo:
        json.dump(salida, archivo, indent=2, ensure_ascii=False)


def procesar_aviso(modelo, caso: dict, config: dict) -> dict:
    """Pasa un aviso por los tres pasos, uno a uno para conservar los puntos."""
    from cadena_tres_pasos import construir_pasos

    extraer, simplificar, formatear = construir_pasos(modelo)
    puntos = extraer.invoke({"texto": caso["texto"]}, config=config)
    texto_simple = simplificar.invoke({"puntos": puntos}, config=config)
    respuesta = formatear.invoke({"texto_simple": texto_simple}, config=config)
    return {"aviso": caso["texto"], "puntos": puntos, "texto_simple": texto_simple, "respuesta": respuesta}


def procesar_mensaje(modelo, caso: dict, config: dict) -> dict:
    """Pasa un mensaje por el flujo con enrutamiento y anota la categoría obtenida."""
    from enrutamiento import (
        MENSAJE_ERROR,
        RESPUESTA_GENERICA,
        construir_cadena_con_enrutamiento,
        construir_paso_clasificacion,
        construir_rutas_seguras,
        procesar_mensaje_seguro,
    )

    categorias = []

    def anotar(categoria: str) -> str:
        categorias.append(categoria)
        return categoria

    rutas = construir_rutas_seguras(modelo)
    # el flujo solo devuelve el texto de la ruta; la categoría se anota de paso, sin una llamada extra
    cadena = construir_cadena_con_enrutamiento(
        construir_paso_clasificacion(modelo) | anotar,
        cadena_incidente=rutas["incidente"],
        cadena_vulnerabilidad=rutas["vulnerabilidad"],
        cadena_consulta=rutas["consulta"],
    )
    respuesta = procesar_mensaje_seguro(cadena, caso["mensaje"], config)
    return {
        "mensaje": caso["mensaje"],
        "categoria_esperada": caso["esperada"],
        "categoria_obtenida": categorias[0] if categorias else None,
        "respuesta": respuesta,
        "hubo_fallo": respuesta in (RESPUESTA_GENERICA, MENSAJE_ERROR),
    }


def ejecutar_casos(casos: dict, modelo, tope_usd: float, ruta_salida: str) -> dict:
    """Procesa cada caso, mide su uso y corta entre casos al llegar al tope de gasto."""
    from api_claude_refactor import INPUT_PRICING, OUTPUT_PRICING
    from langchain_core.callbacks import UsageMetadataCallbackHandler

    salida = {"nota_costo": NOTA_COSTO, "resultados": [], "costo_total_estimado": 0.0, "cortado_por_tope": False}
    trabajos = [(procesar_aviso, caso) for caso in casos["avisos"]]
    trabajos += [(procesar_mensaje, caso) for caso in casos["mensajes"]]
    for procesar, caso in trabajos:
        if salida["costo_total_estimado"] >= tope_usd:
            salida["cortado_por_tope"] = True
            break
        medidor = UsageMetadataCallbackHandler()
        try:
            resultado = procesar(modelo, caso, {"callbacks": [medidor]})
        except Exception as error:
            # un caso fallido no corta los siguientes
            logger.exception("Error procesando caso")
            resultado = {"caso": caso, "error": str(error)}
        if not medidor.usage_metadata:
            logger.warning("Caso sin datos de uso: el costo estimado puede quedar subestimado")
        tokens_entrada = sum(uso["input_tokens"] for uso in medidor.usage_metadata.values())
        tokens_salida = sum(uso["output_tokens"] for uso in medidor.usage_metadata.values())
        costo = tokens_entrada * INPUT_PRICING + tokens_salida * OUTPUT_PRICING
        salida["resultados"].append(
            {**resultado, "tokens_entrada": tokens_entrada, "tokens_salida": tokens_salida, "costo_estimado": costo}
        )
        salida["costo_total_estimado"] += costo
        # guardado incremental: si algo falla a mitad no se pierde lo ya pagado
        guardar_resultados(salida, ruta_salida)
    guardar_resultados(salida, ruta_salida)
    return salida


def informar(salida: dict) -> None:
    """Muestra el resultado de cada caso sin ocultar diferencias ni fallos."""
    for n, resultado in enumerate(salida["resultados"], start=1):
        print(f"[Caso {n}] {resultado.get('respuesta', 'sin respuesta')}")
        if "error" in resultado:
            print(f"  Error: {resultado['error']}")
        if resultado.get("hubo_fallo"):
            print("  Fallo: la respuesta es un texto de respaldo, no del modelo.")
        if "categoria_esperada" in resultado:
            esperada, obtenida = resultado["categoria_esperada"], resultado["categoria_obtenida"]
            if obtenida != esperada:
                print(f"  DIFERENCIA de categoría: esperada {esperada}, obtenida {obtenida}.")
            else:
                print(f"  Categoría: {obtenida} (coincide).")
    if salida["cortado_por_tope"]:
        print("La ejecución se detuvo al alcanzar el tope de gasto.")
    print(f"Costo total estimado: ${salida['costo_total_estimado']:.5f}")
    print(NOTA_COSTO)


def main(argv: list[str] | None = None) -> int:
    """Ejecuta las cadenas contra la API real solo con --confirmar-api-real."""
    parser = argparse.ArgumentParser(description="Cadenas de LangChain sobre respuestas REALES de Claude (con costo).")
    parser.add_argument("--casos", default="JSON/casos_cadenas.json")
    parser.add_argument("--salida", default="docs/resultados_cadenas.json")
    parser.add_argument("--tope-usd", type=float, default=TOPE_USD_MAXIMO)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_MAXIMO)
    parser.add_argument("--confirmar-api-real", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirmar_api_real:
        print(
            "Este script hace llamadas REALES a la API de Claude, con costo. "
            "Para ejecutarlo pasa --confirmar-api-real.",
            file=sys.stderr,
        )
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
    if os.path.realpath(args.salida) == os.path.realpath(args.casos):
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
    casos = cargar_casos(args.casos)
    problema = validar_casos(casos)
    if problema:
        print(f"Error: {problema}", file=sys.stderr)
        return 1
    llamadas = LLAMADAS_POR_AVISO * len(casos["avisos"]) + LLAMADAS_POR_MENSAJE * len(casos["mensajes"])
    if llamadas > MAX_LLAMADAS:
        print(f"Error: los casos requieren {llamadas} llamadas; el máximo es {MAX_LLAMADAS}.", file=sys.stderr)
        return 1

    # import diferido: sin el flag no se carga langchain ni la API key
    from modelo_langchain import crear_modelo

    try:
        modelo = crear_modelo(max_tokens=args.max_tokens)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    salida = ejecutar_casos(casos, modelo, args.tope_usd, args.salida)
    informar(salida)
    if not any("error" not in r and not r.get("hubo_fallo") for r in salida["resultados"]):
        print("Error: ningún caso dio un resultado útil.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
