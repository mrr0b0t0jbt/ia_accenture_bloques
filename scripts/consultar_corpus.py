"""Consulta el corpus indexado con recuperación semántica y respuestas REALES de Claude con citas (Bloque 26).

Para cada pregunta de JSON/casos_recuperacion.json recupera los fragmentos más parecidos con
embeddings REALES de Gemini, le pide a Claude que responda SOLO con ese contexto citando [n]
y verifica cada cita. Escribe los resultados en docs/resultados_recuperacion.json después de
cada caso. Hace llamadas REALES a las APIs, con costo.

Requiere el índice creado antes con scripts/indexar_corpus.py --confirmar-api-real.

Cómo correrlo (desde cualquier directorio):

    python scripts/consultar_corpus.py --confirmar-api-real
    python scripts/consultar_corpus.py --confirmar-api-real --k 4 --tope-usd 0.02
"""
import argparse
import importlib.util
import logging
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import embeddings_gemini  # noqa: E402  (no carga langchain_google_genai hasta crear_embeddings)

_spec = importlib.util.spec_from_file_location("ejecutar_cadenas", RAIZ / "scripts" / "ejecutar_cadenas.py")
ejecutar_cadenas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ejecutar_cadenas)

logger = logging.getLogger(__name__)

TOPE_USD_MAXIMO = 0.05
MAX_TOKENS_MAXIMO = 1024
MAX_TOKENS_POR_DEFECTO = 512
MAX_PREGUNTAS = 10
MAX_K_SCRIPT = 5
NOTA_COSTO = (
    f"{ejecutar_cadenas.NOTA_COSTO} Los embeddings de Gemini no se estiman en dólares "
    "porque el proyecto no tiene su precio."
)


def validar_casos_recuperacion(casos) -> str | None:
    """Devuelve el primer problema de estructura de las preguntas, o None si no hay ninguno."""
    # import diferido: recuperacion carga langchain_core, y aquí ya se pasó el flag
    from recuperacion import MAX_LONGITUD_PREGUNTA

    if not isinstance(casos, list) or not 1 <= len(casos) <= MAX_PREGUNTAS:
        return f"el archivo de casos debe ser una lista de 1 a {MAX_PREGUNTAS} preguntas."
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
        if len(caso["pregunta"]) > MAX_LONGITUD_PREGUNTA:
            return f"casos[{n}] tiene una pregunta de más de {MAX_LONGITUD_PREGUNTA} caracteres."
        fuentes = caso.get("fuentes_esperadas")
        if not isinstance(fuentes, list) or not all(isinstance(fuente, str) for fuente in fuentes):
            return f"casos[{n}] no tiene fuentes_esperadas como lista de textos."
        if not isinstance(caso.get("respondible"), bool):
            return f"casos[{n}] no tiene respondible como booleano."
        if caso["respondible"] and not fuentes:
            return f"casos[{n}] es respondible y no tiene fuentes_esperadas."
        if not caso["respondible"] and fuentes:
            return f"casos[{n}] no es respondible y tiene fuentes_esperadas."
        if "nota" in caso and not isinstance(caso["nota"], str):
            return f"casos[{n}] tiene una nota que no es texto."
    return None


def consultar_casos(casos: list[dict], vectorstore, modelo, k: int, tope_usd: float, ruta_salida: str) -> dict:
    """Recupera, pregunta a Claude y verifica las citas de cada caso; corta entre casos al llegar al tope."""
    import indexacion
    import recuperacion
    from api_claude_refactor import INPUT_PRICING, OUTPUT_PRICING

    salida = {
        "nota_costo": NOTA_COSTO,
        "k": k,
        "preguntas_procesadas": 0,
        "consultas_de_embeddings": 0,
        "llamadas_a_claude": 0,
        "costo_total_estimado": 0.0,
        "cortado_por_tope": False,
        "resultados": [],
    }
    for caso in casos:
        if salida["costo_total_estimado"] >= tope_usd:
            salida["cortado_por_tope"] = True
            break
        resultado = {
            "id": caso["id"],
            "pregunta": caso["pregunta"],
            "respondible": caso["respondible"],
            "fuentes_esperadas": caso["fuentes_esperadas"],
            "recuperados": [],
            "acierto_recuperacion": None,
            "respuesta": "",
            "citas": recuperacion.extraer_citas("", {}),
            "admite_falta_de_contexto": False,
            "llamadas_modelo": 0,
            "tokens_entrada": 0,
            "tokens_salida": 0,
            "costo_estimado": 0.0,
            "truncado": False,
        }

        def buscar():
            salida["consultas_de_embeddings"] += 1
            return recuperacion.construir_prompt(caso["pregunta"], vectorstore, k)

        try:
            prompt, mapa_citas, encontrados = indexacion.con_reintentos(buscar)
        except Exception as error:
            resultado["error"] = embeddings_gemini.describir_error_api(error)[1]
        else:
            resultado["recuperados"] = [
                {
                    "rango": rango,
                    "documento": documento.metadata.get("documento"),
                    "fragmento": documento.metadata.get("fragmento"),
                    "fuente": documento.metadata.get("fuente"),
                    "distancia": distancia,
                }
                for rango, (documento, distancia) in enumerate(encontrados, start=1)
            ]
            if caso["respondible"]:
                recuperados = {r["documento"] for r in resultado["recuperados"]}
                resultado["acierto_recuperacion"] = set(caso["fuentes_esperadas"]) <= recuperados
            resultado["llamadas_modelo"] = 1
            try:
                mensaje = modelo.invoke(prompt)
            except Exception as error:
                # un caso fallido no corta los siguientes
                logger.exception("Error consultando al modelo")
                resultado["error"] = f"{type(error).__name__}: {str(error)[:300]}"
            else:
                uso = getattr(mensaje, "usage_metadata", None) or {}
                if not uso:
                    logger.warning("Caso sin datos de uso: el costo estimado puede quedar subestimado")
                resultado["tokens_entrada"] = uso.get("input_tokens", 0)
                resultado["tokens_salida"] = uso.get("output_tokens", 0)
                resultado["costo_estimado"] = resultado["tokens_entrada"] * INPUT_PRICING + resultado["tokens_salida"] * OUTPUT_PRICING
                resultado["truncado"] = getattr(mensaje, "response_metadata", {}).get("stop_reason") == "max_tokens"
                resultado["respuesta"] = recuperacion.texto_de_respuesta(mensaje)
                resultado["citas"] = recuperacion.extraer_citas(resultado["respuesta"], mapa_citas)
                resultado["admite_falta_de_contexto"] = recuperacion.admite_falta_de_contexto(resultado["respuesta"])
        salida["resultados"].append(resultado)
        salida["preguntas_procesadas"] += 1
        salida["llamadas_a_claude"] += resultado["llamadas_modelo"]
        salida["costo_total_estimado"] += resultado["costo_estimado"]
        # guardado incremental: si algo falla a mitad no se pierde lo ya pagado
        ejecutar_cadenas.guardar_resultados(salida, ruta_salida)
    ejecutar_cadenas.guardar_resultados(salida, ruta_salida)
    return salida


def informar(salida: dict) -> None:
    """Muestra cada caso con lo recuperado, la respuesta, las citas y si admitió la falta de contexto."""
    from recuperacion import limpiar_para_consola as limpio

    for resultado in salida["resultados"]:
        print(f"\n[{limpio(resultado['id'])}] {limpio(resultado['pregunta'])}")
        for r in resultado["recuperados"]:
            print(f"  {r['rango']}. {limpio(str(r['fuente']))} (documento {limpio(str(r['documento']))}, fragmento {r['fragmento']}) distancia {r['distancia']:.3f}")
        if "error" in resultado:
            print(f"  Error: {limpio(resultado['error'])}")
        if resultado["llamadas_modelo"] and "error" not in resultado:
            print(f"  Respuesta: {limpio(resultado['respuesta'])}")
        citas = resultado["citas"]
        for numero, fuente in citas["validas"]:
            print(f"  Cita [{numero}] -> {limpio(fuente)}")
        for numero in citas["invalidas"]:
            print(f"  Cita INVALIDA [{numero}]")
        if citas["sin_citas"] and "error" not in resultado:
            print("  SIN CITAS")
        if resultado["truncado"]:
            print("  AVISO: respuesta cortada por max_tokens.")
        admite = resultado["admite_falta_de_contexto"]
        if not resultado["respondible"]:
            print("  ADMITIO la falta de contexto" if admite else "  NO ADMITIO la falta de contexto (posible alucinacion)")
        elif admite:
            print("  ADMITIO falta de contexto aunque la pregunta era respondible")
    if salida["cortado_por_tope"]:
        print("\nLa ejecución se detuvo al alcanzar el tope de gasto.")
    print(f"\nPreguntas procesadas: {salida['preguntas_procesadas']}")
    print(f"Consultas de embeddings: {salida['consultas_de_embeddings']}")
    print(f"Llamadas a Claude: {salida['llamadas_a_claude']}")
    print(f"Costo total estimado: ${salida['costo_total_estimado']:.5f}")
    print(salida["nota_costo"])


def _error(mensaje: str) -> int:
    print(f"Error: {mensaje}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Consulta el corpus contra las APIs reales solo con --confirmar-api-real."""
    parser = argparse.ArgumentParser(description="Responde preguntas con recuperación semántica (Gemini) y Claude, citando fuentes (con costo).")
    parser.add_argument("--casos", default=str(RAIZ / "JSON" / "casos_recuperacion.json"))
    parser.add_argument("--directorio", default=str(RAIZ / "vectorstore"))
    parser.add_argument("--coleccion", default="corpus_miniapp")
    parser.add_argument("--salida", default=str(RAIZ / "docs" / "resultados_recuperacion.json"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--tope-usd", type=float, default=TOPE_USD_MAXIMO)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_POR_DEFECTO)
    parser.add_argument("--confirmar-api-real", action="store_true")
    args = parser.parse_args(argv)
    if not args.confirmar_api_real:
        print(
            "Este script hace llamadas REALES a las APIs de Gemini (embeddings) y de Claude, con costo. "
            "Para ejecutarlo pasa --confirmar-api-real.",
            file=sys.stderr,
        )
        return 2
    if not 1 <= args.k <= MAX_K_SCRIPT:
        return _error(f"--k debe estar entre 1 y {MAX_K_SCRIPT}.")
    if not 0 < args.tope_usd <= TOPE_USD_MAXIMO:
        return _error(f"--tope-usd debe ser mayor que 0 y como máximo {TOPE_USD_MAXIMO}.")
    if not 1 <= args.max_tokens <= MAX_TOKENS_MAXIMO:
        return _error(f"--max-tokens debe estar entre 1 y {MAX_TOKENS_MAXIMO}.")
    carpeta_salida = os.path.dirname(os.path.abspath(args.salida))
    if not os.path.isdir(carpeta_salida) or os.path.isdir(args.salida):
        return _error(f"--salida no es un archivo en un directorio existente: {args.salida}")
    if ejecutar_cadenas.es_el_mismo_archivo(args.salida, args.casos):
        return _error("--salida no puede ser el mismo archivo que --casos.")
    try:
        casos = ejecutar_cadenas.cargar_casos(args.casos)
    except (OSError, ValueError) as error:  # JSONDecodeError y UnicodeDecodeError son ValueError
        return _error(f"no se pudieron leer los casos: {error}")
    problema = validar_casos_recuperacion(casos)
    if problema:
        return _error(problema)

    # import diferido: sin el flag no se carga langchain ni se leen las claves
    import indexacion
    import modelo_langchain

    try:
        vectorstore = indexacion.cargar_vectorstore(embeddings_gemini.crear_embeddings(), args.directorio, args.coleccion)
        en_indice = {metadata.get("documento") for metadata in vectorstore.get(include=["metadatas"])["metadatas"]}
    except Exception as error:
        return _error(str(error)[:300])
    faltan = sorted({fuente for caso in casos for fuente in caso["fuentes_esperadas"]} - en_indice)
    if faltan:
        return _error(f"las fuentes esperadas no están en el índice: {', '.join(faltan)}. Reindexa el corpus.")
    try:
        modelo = modelo_langchain.crear_modelo(max_tokens=args.max_tokens)
    except ValueError as error:
        return _error(str(error))
    salida = consultar_casos(casos, vectorstore, modelo, args.k, args.tope_usd, args.salida)
    informar(salida)
    if all("error" in resultado for resultado in salida["resultados"]):
        return _error("ningun caso dio un resultado util.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
