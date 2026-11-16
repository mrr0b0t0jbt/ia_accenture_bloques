"""Mide la latencia por etapa y el costo del flujo RAG completo con APIs REALES (Bloque 28).

Para cada pregunta de JSON/casos_latencia_costo.json y cada k pedido hace UNA petición de embedding con Gemini
(embed_query), busca en el índice local y hace UNA llamada a Claude. Anexa un registro por llamada a
docs/registros_latencia_costo.jsonl (archivo de solo anexar). Hace llamadas REALES, con costo: los embeddings de Gemini
no se estiman en dólares (se cuentan peticiones) y el costo de Claude es una estimación con los precios del proyecto.

Los topes están en el código: MAX_CASOS, MAX_KS, MAX_LLAMADAS, MAX_CONSULTAS_EMBEDDING (se fuerza en ejecución,
los reintentos de un 429 cuentan) y el tope de gasto, que se comprueba antes de cada llamada a Claude (el gasto puede
pasarse como máximo en lo que cuesta UNA llamada). Con Ctrl-C lo ya anexado queda y sale con 130.

Requiere el índice creado antes con scripts/indexar_corpus.py --confirmar-api-real.

Cómo correrlo (desde cualquier directorio):

    python scripts/medir_latencia_costo.py --confirmar-api-real
    python scripts/medir_latencia_costo.py --confirmar-api-real --ks 3,6 --max-tokens 512 --tope-usd 0.08
"""
import argparse
import datetime
import importlib.util
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

_spec = importlib.util.spec_from_file_location("ejecutar_cadenas", RAIZ / "scripts" / "ejecutar_cadenas.py")
ejecutar_cadenas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ejecutar_cadenas)  # solo biblioteca estándar

MAX_CASOS = 6
MAX_KS = 2
MAX_LLAMADAS = 12
MAX_CONSULTAS_EMBEDDING = 20
TOPE_USD_MAXIMO = 0.08
MAX_TOKENS_MAXIMO = 1024
MAX_TOKENS_POR_DEFECTO = 512
NOTA_COSTO = (
    "Los embeddings de Gemini no se estiman en dólares (el proyecto no tiene el precio de gemini-embedding-001): "
    "se cuentan peticiones. El costo de Claude es una estimación con los precios del proyecto (USD por token); no es el cobro real."
)


def _error(mensaje: str, codigo: int = 1) -> int:
    print(f"Error: {mensaje}", file=sys.stderr)
    return codigo


def _linea(texto) -> str:
    """Texto externo para consola: sin controles ni saltos de línea."""
    from recuperacion import limpiar_para_consola

    return limpiar_para_consola(str(texto)).replace("\n", " ")


def _ahora() -> datetime.datetime:
    """Hora actual en UTC (se parchea en las pruebas)."""
    return datetime.datetime.now(datetime.timezone.utc)


def _parsear_ks(texto: str, k_maximo: int) -> list[int] | None:
    """Convierte '3,6' en [3, 6]; None si no son 1 o 2 enteros distintos entre 1 y k_maximo."""
    partes = [parte.strip() for parte in texto.split(",")]
    if not 1 <= len(partes) <= MAX_KS or not all(p.isascii() and p.isdigit() for p in partes):
        return None
    ks = [int(p) for p in partes]
    if len(set(ks)) != len(ks) or not all(1 <= k <= k_maximo for k in ks):
        return None
    return ks


def _problema_de_casos(casos, maximo_pregunta: int) -> str | None:
    """Primer problema de estructura de los casos, o None."""
    if not isinstance(casos, list) or not 1 <= len(casos) <= MAX_CASOS:
        return f"el archivo de casos debe ser una lista de 1 a {MAX_CASOS} casos."
    vistos = set()
    for n, caso in enumerate(casos):
        if not isinstance(caso, dict):
            return f"casos[{n}] no es un objeto."
        for campo in ("id", "pregunta"):
            if not ejecutar_cadenas._es_texto_no_vacio(caso.get(campo)):
                return f"casos[{n}] no tiene el campo {campo} como texto no vacío."
        if caso["id"] in vistos:
            return f"casos[{n}] repite el id {caso['id']!r}."
        vistos.add(caso["id"])
        if len(caso["pregunta"]) > maximo_pregunta:
            return f"casos[{n}] tiene una pregunta de más de {maximo_pregunta} caracteres."
    return None


def _es_registro_previo(ruta: str) -> bool:
    """Indica si el archivo está vacío o es un JSONL de registros de este script."""
    import medicion_rag

    try:
        if os.path.getsize(ruta) > 0:
            medicion_rag.leer_registros(ruta)  # lanza ValueError si alguna línea no es un registro
    except (OSError, ValueError):
        return False
    return True


def _problema_de_salida(salida: str, casos: str, directorio_indice: str) -> str | None:
    """Motivo por el que --salida no es seguro (se corrompería un archivo ajeno o no se podría escribir), o None."""
    if not salida:
        return "--salida no puede estar vacío."
    carpeta = os.path.dirname(os.path.abspath(salida))
    if not os.path.isdir(carpeta) or os.path.isdir(salida):
        return f"--salida no es un archivo en un directorio existente: {salida}"
    if ejecutar_cadenas.es_el_mismo_archivo(salida, casos):
        return "--salida no puede ser el mismo archivo que --casos."
    indice = os.path.realpath(directorio_indice)
    if os.path.commonpath([indice, os.path.realpath(salida)]) == indice:
        return "--salida no puede quedar dentro del directorio del índice (--directorio)."
    if not os.access(carpeta, os.W_OK | os.X_OK) or (os.path.exists(salida) and not os.access(salida, os.W_OK)):
        return f"no se puede escribir en --salida: {salida}"
    if os.path.exists(salida) and not (os.path.isfile(salida) and _es_registro_previo(salida)):
        return "--salida ya existe y no es un archivo vacío ni un registro de una corrida anterior; anexar ahí lo corrompería."
    return None


def _ms(valor) -> str:
    return "n/d" if valor is None else f"{valor:.1f}"


def _linea_de_registro(registro: dict) -> str:
    """Una línea del informe por registro (los textos externos pasan por _linea)."""
    t = registro["tiempos_ms"]
    texto = (
        f"{_linea(registro['id_caso'])} k={registro['k']} embed {_ms(t['embed_query'])} ms | busqueda {_ms(t['busqueda_vectorial'])} ms"
        f" | generacion {_ms(t['generacion'])} ms | total {_ms(registro['tiempo_total_ms'])} ms | "
    )
    if registro["error"] is not None:
        return texto + f"ERROR: {_linea(registro['error'])}"
    if registro["tokens"] is None:
        return texto + "tokens sin datos de uso"
    texto += f"tokens in {registro['tokens']['input_tokens']} out {registro['tokens']['output_tokens']} | USD {registro['costo_estimado']:.5f}"
    return texto + (" (cortada por max_tokens)" if registro["truncado"] else "")


def informar(registros: list[dict], consultas: int, costo_total: float) -> None:
    """Muestra cada registro, los promedios por k y los totales."""
    import medicion_rag

    for registro in registros:
        print(_linea_de_registro(registro))
    resumen = medicion_rag.resumir(registros)
    print(f"\nLlamadas registradas: {resumen['llamadas']} (con error: {resumen['con_error']}; los errores no entran en los promedios)")
    for k, datos in resumen["por_k"].items():
        p = datos["promedio_ms"]
        print(
            f"  k={k} ({datos['registros']} registros): embed {p['embed_query']} ms | busqueda {p['busqueda_vectorial']} ms | "
            f"generacion {p['generacion']} ms | total {p['total']} ms | tokens in {datos['promedio_input_tokens']} "
            f"out {datos['promedio_output_tokens']} | USD {datos['costo_total']:.5f}"
        )
    crecen = resumen["tokens_crecen_con_k"]
    veredicto = {True: "si", False: "no", None: "sin datos"}[crecen["si"]]
    extra = f" (no crece en: {_linea(', '.join(crecen['casos_que_no_crecen']))})" if crecen["casos_que_no_crecen"] else ""
    print(f"input_tokens crece con k: {veredicto}{extra}")
    lenta = resumen["generacion_mas_lenta"]
    print(f"generacion es la etapa mas lenta en {lenta['si']} de {lenta['total']} registros")
    print(f"Consultas de embedding (incluidos reintentos): {consultas}")
    print(f"Llamadas a Claude: {len(registros)}")
    print(f"Costo total estimado de Claude: USD {costo_total:.5f}")
    print(NOTA_COSTO)


def _correr(casos: list[dict], ks: list[int], args, recursos: tuple, contador: dict, registros: list[dict], corrida: str) -> int:
    """Una llamada por caso y k, anexando cada registro al instante; devuelve el código de salida (Ctrl-C se propaga)."""
    import embeddings_gemini
    import medicion_rag
    from api_claude_refactor import INPUT_PRICING, OUTPUT_PRICING

    vectorstore, embeddings, modelo = recursos
    gasto = 0.0
    for caso in casos:
        for k in ks:
            # tolerancia: el costo en punto flotante puede quedar un epsilon por debajo del tope exacto
            if gasto >= args.tope_usd - 1e-9:
                print(f"\nLa ejecución se detuvo al alcanzar el tope de gasto (USD {args.tope_usd}).")
                return 0
            try:
                registro = medicion_rag.registrar_consulta(
                    caso["id"], caso["pregunta"], k, vectorstore, embeddings, modelo, contador,
                    MAX_CONSULTAS_EMBEDDING, (INPUT_PRICING, OUTPUT_PRICING), corrida, args.max_tokens, ahora=_ahora,
                )
            except Exception as error:
                # error de la etapa de embedding: no hay datos que registrar
                topado = isinstance(error, medicion_rag.TopeDePeticiones)
                return _error(str(error) if topado else embeddings_gemini.describir_error_api(error)[1])
            medicion_rag.anexar_registro(args.salida, registro)
            registros.append(registro)
            gasto += registro["costo_estimado"] or 0.0
            if registro["sin_datos_de_uso"]:
                return _error("Claude no informó los tokens usados: sin uso no se controla el gasto; se detiene la corrida.")
    return 0


def _crear_parser() -> argparse.ArgumentParser:
    """Arma el parser; allow_abbrev=False evita que '--conf' active --confirmar-api-real."""
    parser = argparse.ArgumentParser(
        description="Mide la latencia por etapa y el costo del flujo RAG con Gemini y Claude (con costo).",
        allow_abbrev=False,
    )
    parser.add_argument("--casos", default=str(RAIZ / "JSON" / "casos_latencia_costo.json"))
    parser.add_argument("--directorio", default=str(RAIZ / "vectorstore"))
    parser.add_argument("--coleccion", default=None, help="Por defecto, la de indexacion.")
    parser.add_argument("--salida", default=str(RAIZ / "docs" / "registros_latencia_costo.jsonl"))
    parser.add_argument("--ks", default="3,6", help="1 o 2 enteros separados por coma.")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_POR_DEFECTO)
    parser.add_argument("--tope-usd", type=float, default=TOPE_USD_MAXIMO)
    parser.add_argument("--confirmar-api-real", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Mide el flujo RAG contra las APIs reales solo con --confirmar-api-real."""
    args = _crear_parser().parse_args(argv)
    if not args.confirmar_api_real:
        print(
            "Este script hace llamadas REALES a las APIs de Gemini (embeddings) y de Claude, con costo. "
            "Para ejecutarlo pasa --confirmar-api-real.",
            file=sys.stderr,
        )
        return 2
    # imports diferidos: sin el flag no se carga langchain ni chroma ni se leen las claves
    import embeddings_gemini
    import indexacion
    import medicion_rag
    import modelo_langchain
    import recuperacion

    ks = _parsear_ks(args.ks, medicion_rag.K_MAXIMO_MEDICION)
    if ks is None:
        return _error(f"--ks debe ser 1 o 2 enteros distintos entre 1 y {medicion_rag.K_MAXIMO_MEDICION}, separados por coma.", 2)
    if not 1 <= args.max_tokens <= MAX_TOKENS_MAXIMO:
        return _error(f"--max-tokens debe estar entre 1 y {MAX_TOKENS_MAXIMO}.", 2)
    if not 0 < args.tope_usd <= TOPE_USD_MAXIMO:
        return _error(f"--tope-usd debe ser mayor que 0 y como máximo {TOPE_USD_MAXIMO}.", 2)
    args.casos, args.directorio, args.salida = (os.path.expanduser(ruta) for ruta in (args.casos, args.directorio, args.salida))

    try:
        casos = ejecutar_cadenas.cargar_casos(args.casos)
    except (OSError, ValueError) as error:  # JSONDecodeError y UnicodeDecodeError son ValueError
        return _error(f"no se pudieron leer los casos: {error}")
    problema = _problema_de_casos(casos, recuperacion.MAX_LONGITUD_PREGUNTA)
    if problema:
        return _error(problema)
    if len(casos) * len(ks) > MAX_LLAMADAS:
        return _error(f"{len(casos)} casos x {len(ks)} valores de k son {len(casos) * len(ks)} llamadas; el máximo es {MAX_LLAMADAS}.")
    problema = _problema_de_salida(args.salida, args.casos, args.directorio)
    if problema:
        return _error(problema)

    coleccion = indexacion.COLECCION_POR_DEFECTO if args.coleccion is None else args.coleccion
    ruta_indice = indexacion.ruta_absoluta(args.directorio)
    # Chroma crea el directorio vacío en silencio, por eso se comprueba antes de abrirlo
    if not os.path.isdir(ruta_indice):
        return _error(f"no existe el índice en {ruta_indice}: indexa primero con scripts/indexar_corpus.py --confirmar-api-real.")
    try:
        embeddings = embeddings_gemini.crear_embeddings()  # solo crea el cliente: no hace ninguna petición
    except ValueError as error:
        return _error(str(error))
    except Exception as error:
        return _error(embeddings_gemini.describir_error_api(error)[1])
    try:
        vectorstore = indexacion.cargar_vectorstore(embeddings, ruta_indice, coleccion)
    except Exception as error:
        return _error(str(error)[:300])
    try:
        modelo = modelo_langchain.crear_modelo(max_tokens=args.max_tokens)
    except ValueError as error:
        return _error(str(error))

    corrida = _ahora().strftime("%Y%m%dT%H%M%SZ")
    contador, registros = {"consultas": 0}, []
    try:
        codigo = _correr(casos, ks, args, (vectorstore, embeddings, modelo), contador, registros, corrida)
    except KeyboardInterrupt:
        codigo = 130
    informar(registros, contador["consultas"], sum(r["costo_estimado"] or 0.0 for r in registros))
    if codigo == 130:
        print("Interrumpido por la persona. Lo ya anexado se conserva.", file=sys.stderr)
    return codigo


if __name__ == "__main__":
    sys.exit(main())
