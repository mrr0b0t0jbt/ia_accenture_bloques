"""Evalúa la recuperación por CHUNK con embeddings REALES de Gemini (Bloque 27).

Para cada caso de JSON/evaluacion_recuperacion.json recupera los 5 chunks más parecidos del índice y calcula
precisión, exhaustividad y F1 (ver src/metricas_recuperacion.py). NO llama a Claude: lo único que gasta son los
embeddings de las consultas con Gemini (una petición por caso y repetición). Escribe los resultados en
docs/resultados_evaluacion_recuperacion.json después de cada caso de la primera repetición.

Con --repeticiones 2 (por defecto) repite todas las consultas para comprobar si la recuperación es determinista.
Los topes (casos, repeticiones y consultas de embeddings) están en el código: sin ellos el script no gasta.
Con Ctrl-C guarda lo hecho y sale con 130.

Requiere el índice creado antes con scripts/indexar_corpus.py --confirmar-api-real (mismo corpus y tamaños).

Cómo correrlo (desde cualquier directorio):

    python scripts/evaluar_recuperacion.py --confirmar-api-real
    python scripts/evaluar_recuperacion.py --confirmar-api-real --k 5 --repeticiones 1
"""
import argparse
import datetime
import importlib.util
import json
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import metricas_recuperacion  # noqa: E402  (Python puro: no carga langchain)
from metricas_recuperacion import K_MAXIMO  # noqa: E402

_spec = importlib.util.spec_from_file_location("ejecutar_cadenas", RAIZ / "scripts" / "ejecutar_cadenas.py")
ejecutar_cadenas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ejecutar_cadenas)  # solo biblioteca estándar

MAX_CASOS = metricas_recuperacion.MAX_CASOS
MAX_REPETICIONES = 2
MAX_CONSULTAS_EMBEDDING = 40
MODELO_EMBEDDINGS = "gemini-embedding-001"
NOTA_COSTO = (
    "Solo se gastan embeddings de consultas de Gemini (una petición por consulta); el proyecto no tiene el precio de "
    "gemini-embedding-001, por eso no se estima un costo en dólares. No se llama a Claude."
)
NOTA_METODO = (
    "Las etiquetas se hicieron leyendo el corpus antes de ver lo que devuelve el retriever; los promedios de precisión, "
    "exhaustividad y F1 son solo sobre los casos respondibles y los casos sin respuesta se reportan aparte "
    "(ver el docstring de metricas_recuperacion)."
)


def _error(mensaje: str, codigo: int = 1) -> int:
    print(f"Error: {mensaje}", file=sys.stderr)
    return codigo


def _linea(texto) -> str:
    """Texto externo para consola: sin controles ni saltos de línea."""
    from recuperacion import limpiar_para_consola

    return limpiar_para_consola(str(texto)).replace("\n", " ")


def _es_resultado_previo(ruta: str) -> bool:
    """Indica si el archivo es un JSON objeto con 'por_caso' (lista), 'promedios_respondibles' (objeto) y 'nota_costo' (texto)."""
    try:
        with open(ruta, encoding="utf-8") as archivo:
            contenido = json.load(archivo)
    except (OSError, ValueError):
        return False
    return (
        isinstance(contenido, dict)
        and isinstance(contenido.get("por_caso"), list)
        and isinstance(contenido.get("promedios_respondibles"), dict)
        and isinstance(contenido.get("nota_costo"), str)
    )


def _problema_de_salida(salida: str, casos: str, corpus: str, directorio_indice: str) -> str | None:
    """Motivo por el que --salida no es seguro (se pisaría un archivo ajeno o no se podría escribir), o None."""
    if not salida:
        return "--salida no puede estar vacío."
    carpeta = os.path.dirname(os.path.abspath(salida))
    if not os.path.isdir(carpeta) or os.path.isdir(salida):
        return f"--salida no es un archivo en un directorio existente: {salida}"
    if ejecutar_cadenas.es_el_mismo_archivo(salida, casos):
        return "--salida no puede ser el mismo archivo que --casos."
    if ejecutar_cadenas.es_el_mismo_archivo(salida, corpus):
        return "--salida no puede ser el mismo archivo que --corpus."
    indice = os.path.realpath(directorio_indice)
    if os.path.commonpath([indice, os.path.realpath(salida)]) == indice:
        return "--salida no puede quedar dentro del directorio del índice (--directorio)."
    if not os.access(carpeta, os.W_OK | os.X_OK) or (os.path.exists(salida) and not os.access(salida, os.W_OK)):
        return f"no se puede escribir en --salida: {salida}"
    if os.path.exists(salida) and not (os.path.isfile(salida) and _es_resultado_previo(salida)):
        return "--salida ya existe y no es un resultado de una corrida anterior; usa otra ruta o borralo."
    return None


def _id_de_chunk(documento) -> str:
    """Id documento_fragmento construido desde la metadata (no desde Document.id)."""
    metadata = documento.metadata
    id_documento, fragmento = metadata.get("documento"), metadata.get("fragmento")
    if not isinstance(id_documento, str) or not id_documento or not isinstance(fragmento, int) or isinstance(fragmento, bool):
        raise ValueError(f"un chunk recuperado no tiene la metadata documento (texto) y fragmento (entero): {metadata!r}"[:300])
    return f"{id_documento}_{fragmento}"


def _consultar(vectorstore, pregunta: str, corrida: dict) -> list[tuple[str, float]]:
    """Una recuperación de K_MAXIMO chunks; cada intento (también los de un 429) cuenta como una consulta de embedding."""
    import indexacion
    import recuperacion

    def buscar():
        corrida["consultas"] += 1
        return recuperacion.recuperar(vectorstore, pregunta, k=K_MAXIMO)

    return [(_id_de_chunk(documento), distancia) for documento, distancia in indexacion.con_reintentos(buscar)]


def _resultado_de(casos_hechos: list[dict], pares_hechos: list, k: int) -> dict:
    """Evalúa los casos hechos reutilizando los rankings ya recuperados (no vuelve a consultar)."""
    if not casos_hechos:
        return {"por_caso": [], "promedios_respondibles": {}, "sin_respuesta": []}
    rankings = iter(pares_hechos)
    return metricas_recuperacion.evaluar_recuperacion(lambda pregunta, k_max: next(rankings), casos_hechos, k)


def _documento(corrida: dict, casos: list[dict], k: int, configuracion: dict) -> dict:
    """Arma el JSON de salida con lo hecho hasta ahora."""
    resultado = _resultado_de(casos[: len(corrida["pares"])], corrida["pares"], k)
    return {
        "nota_costo": NOTA_COSTO,
        "fecha": datetime.date.today().isoformat(),
        "configuracion": configuracion,
        "consultas_de_embeddings": corrida["consultas"],
        "interrumpido": corrida["interrumpido"],
        "determinista": corrida["determinista"],
        "max_dif_distancia": corrida["max_dif_distancia"],
        "nota_metodo": NOTA_METODO,
        "promedios_respondibles": resultado["promedios_respondibles"],
        "sin_respuesta": resultado["sin_respuesta"],
        "por_caso": resultado["por_caso"],
    }


def _redondear(distancia):
    return None if distancia is None else round(distancia, 6)


def _guardar(documento: dict, ruta: str) -> None:
    """Guarda de forma atómica el documento con las distancias redondeadas a 6 decimales."""
    guardado = dict(documento)
    guardado["max_dif_distancia"] = _redondear(documento["max_dif_distancia"])
    guardado["sin_respuesta"] = [{**s, "distancia_minima": _redondear(s["distancia_minima"])} for s in documento["sin_respuesta"]]
    guardado["por_caso"] = [
        {**c, "recuperados": [{**r, "distancia": _redondear(r["distancia"])} for r in c["recuperados"]]} for c in documento["por_caso"]
    ]
    ejecutar_cadenas.guardar_resultados(guardado, ruta)


def _correr(casos: list[dict], vectorstore, args, configuracion: dict) -> dict:
    """Primera pasada con guardado tras cada caso y, si se pide, segunda pasada para medir el determinismo."""
    corrida = {"pares": [], "consultas": 0, "interrumpido": False, "determinista": None, "max_dif_distancia": None}
    try:
        for caso in casos:
            corrida["pares"].append(_consultar(vectorstore, caso["pregunta"], corrida))
            _guardar(_documento(corrida, casos, args.k, configuracion), args.salida)
        if args.repeticiones == 2:
            iguales, diferencias = True, []
            for caso, primeros in zip(casos, corrida["pares"]):
                segundos = _consultar(vectorstore, caso["pregunta"], corrida)
                iguales = iguales and [i for i, _ in primeros] == [i for i, _ in segundos]
                diferencias += [abs(a[1] - b[1]) for a, b in zip(primeros, segundos)]
            corrida["determinista"], corrida["max_dif_distancia"] = iguales, max(diferencias, default=0.0)
    except KeyboardInterrupt:
        corrida["interrumpido"] = True
    documento = _documento(corrida, casos, args.k, configuracion)
    _guardar(documento, args.salida)
    return documento


def informar(documento: dict, k: int) -> None:
    """Muestra las métricas por caso, los promedios al k pedido, la tabla por k y el determinismo."""
    for caso in documento["por_caso"]:
        if caso["respondible"]:
            m = caso["metricas"][k]
            print(f"P={m['precision']:.2f} R={m['exhaustividad']:.2f} F1={m['f1']:.2f}  {_linea(caso['id'])}")
    for caso in documento["sin_respuesta"]:
        minima = "n/a" if caso["distancia_minima"] is None else f"{caso['distancia_minima']:.4f}"
        print(f"sin respuesta: recupero [{_linea(', '.join(caso['ids_recuperados']))}] (distancia minima {minima})  {_linea(caso['id'])}")
    promedios = documento["promedios_respondibles"]
    if promedios:
        print(f"\nPromedios de los casos respondibles con k={k}: P={promedios[k]['precision']:.2f} R={promedios[k]['exhaustividad']:.2f} F1={promedios[k]['f1']:.2f}")
        for n, m in promedios.items():
            print(f"  k={n}: P={m['precision']:.2f} R={m['exhaustividad']:.2f} F1={m['f1']:.2f}")
    print(f"\nConsultas de embeddings: {documento['consultas_de_embeddings']}")
    if documento["determinista"] is None:
        print("Determinismo: no medido")
    else:
        print(f"Determinismo: {'sí' if documento['determinista'] else 'NO'} (mayor diferencia de distancia {documento['max_dif_distancia']:.6f})")
    print(documento["nota_costo"])


def _crear_parser() -> argparse.ArgumentParser:
    """Arma el parser; allow_abbrev=False evita que '--conf' active --confirmar-api-real."""
    parser = argparse.ArgumentParser(
        description="Evalúa la recuperación por chunk con embeddings de Gemini (con costo; no llama a Claude).",
        allow_abbrev=False,
    )
    parser.add_argument("--casos", default=str(RAIZ / "JSON" / "evaluacion_recuperacion.json"))
    parser.add_argument("--corpus", default=str(RAIZ / "JSON" / "corpus_miniapp.json"))
    parser.add_argument("--directorio", default=str(RAIZ / "vectorstore"))
    parser.add_argument("--coleccion", default=None, help="Por defecto, la de indexacion.")
    parser.add_argument("--salida", default=str(RAIZ / "docs" / "resultados_evaluacion_recuperacion.json"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--repeticiones", type=int, default=MAX_REPETICIONES)
    parser.add_argument("--confirmar-api-real", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Evalúa la recuperación contra la API real de embeddings solo con --confirmar-api-real."""
    args = _crear_parser().parse_args(argv)
    if not args.confirmar_api_real:
        print(
            "Este script hace llamadas REALES a la API de embeddings de Gemini, con costo. Para ejecutarlo pasa --confirmar-api-real.",
            file=sys.stderr,
        )
        return 2
    if not 1 <= args.k <= K_MAXIMO:
        return _error(f"--k debe estar entre 1 y {K_MAXIMO}.", 2)
    if not 1 <= args.repeticiones <= MAX_REPETICIONES:
        return _error(f"--repeticiones debe estar entre 1 y {MAX_REPETICIONES}.", 2)
    args.casos, args.corpus, args.directorio, args.salida = (
        os.path.expanduser(ruta) for ruta in (args.casos, args.corpus, args.directorio, args.salida)
    )

    # imports diferidos: sin el flag no se carga langchain ni chroma ni se leen las claves
    import embeddings_gemini
    import indexacion
    import recuperacion

    try:
        casos = ejecutar_cadenas.cargar_casos(args.casos)
    except (OSError, ValueError) as error:  # JSONDecodeError y UnicodeDecodeError son ValueError
        return _error(f"no se pudieron leer los casos: {error}")
    try:
        documentos = indexacion.cargar_corpus(args.corpus)
        _, ids_corpus = indexacion.dividir_corpus(documentos, indexacion.CHUNK_SIZE_POR_DEFECTO, indexacion.CHUNK_OVERLAP_POR_DEFECTO)
    except ValueError as error:
        return _error(str(error))
    if isinstance(casos, list) and len(casos) > MAX_CASOS:
        return _error(f"hay {len(casos)} casos; el máximo es {MAX_CASOS}.")
    problema = metricas_recuperacion.validar_casos_evaluacion(casos, set(ids_corpus))
    if problema:
        return _error(problema)
    larga = next((c["id"] for c in casos if len(c["pregunta"]) > recuperacion.MAX_LONGITUD_PREGUNTA), None)
    if larga:
        return _error(f"el caso {larga!r} tiene una pregunta de más de {recuperacion.MAX_LONGITUD_PREGUNTA} caracteres.")
    consultas = len(casos) * args.repeticiones
    if consultas > MAX_CONSULTAS_EMBEDDING:
        return _error(f"{len(casos)} casos x {args.repeticiones} repeticiones son {consultas} consultas de embedding; el máximo es {MAX_CONSULTAS_EMBEDDING}.")
    problema = _problema_de_salida(args.salida, args.casos, args.corpus, args.directorio)
    if problema:
        return _error(problema)

    coleccion = indexacion.COLECCION_POR_DEFECTO if args.coleccion is None else args.coleccion
    ruta_indice = indexacion.ruta_absoluta(args.directorio)
    # Chroma crea el directorio vacío en silencio, por eso se comprueba antes de abrirlo
    if not os.path.isdir(ruta_indice):
        return _error(f"no existe el índice en {ruta_indice}: reindexa con scripts/indexar_corpus.py --confirmar-api-real.")
    try:
        embeddings = embeddings_gemini.crear_embeddings()  # solo crea el cliente: no hace ninguna petición
    except ValueError as error:
        return _error(str(error))
    except Exception as error:
        return _error(embeddings_gemini.describir_error_api(error)[1])
    try:
        vectorstore = indexacion.cargar_vectorstore(embeddings, ruta_indice, coleccion)
        ids_indice = vectorstore.get()["ids"]
    except Exception as error:
        return _error(str(error)[:300])
    if len(ids_indice) != len(ids_corpus) or set(ids_indice) != set(ids_corpus):
        return _error(
            f"el índice tiene {len(ids_indice)} chunks y el corpus produce {len(ids_corpus)}, o sus ids difieren: "
            "reindexa con scripts/indexar_corpus.py --confirmar-api-real."
        )

    configuracion = {
        "k": args.k,
        "k_maximo": K_MAXIMO,
        "repeticiones": args.repeticiones,
        "chunk_size": indexacion.CHUNK_SIZE_POR_DEFECTO,
        "chunk_overlap": indexacion.CHUNK_OVERLAP_POR_DEFECTO,
        "coleccion": coleccion,
        "modelo_embeddings": MODELO_EMBEDDINGS,
        "casos": len(casos),
        "chunks_en_el_indice": len(ids_indice),
    }
    try:
        documento = _correr(casos, vectorstore, args, configuracion)
    except ValueError as error:  # metadata sin la forma esperada
        return _error(str(error))
    except Exception as error:
        return _error(embeddings_gemini.describir_error_api(error)[1])
    informar(documento, args.k)
    if documento["interrumpido"]:
        print("Interrumpido por la persona. Resultados parciales guardados.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
