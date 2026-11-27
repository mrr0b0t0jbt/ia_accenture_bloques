"""Responde UNA pregunta con el servicio RAG del Gate 2 (servicio_rag.consultar) y muestra la respuesta con sus fuentes.

Recupera con embeddings REALES de Gemini desde el índice ya creado y responde con Claude REAL, con costo.
La pregunta no se imprime ni se registra; con --registro-json los logs JSON (sin contenido) van a stderr.
Sin --confirmar-api-real sale con código 2 sin cargar langchain ni leer las claves.
Si el modelo corta la respuesta por --max-tokens, avisa en stderr (sin la pregunta ni la respuesta) y sale con 0.

Requiere el índice creado antes con scripts/indexar_corpus.py --confirmar-api-real.

Cómo correrlo (desde cualquier directorio):

    python scripts/consultar_servicio.py --confirmar-api-real --pregunta "¿Cuál es el plazo de parcheo?"
"""
import argparse
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

MAX_K = 10
K_POR_DEFECTO = 3
MAX_TOKENS_MAXIMO = 1024
MAX_TOKENS_POR_DEFECTO = 512
MAX_LARGO_ERROR = 300


def _error(mensaje: str) -> int:
    print(f"Error: {mensaje}"[:MAX_LARGO_ERROR], file=sys.stderr)
    return 1


def _crear_parser() -> argparse.ArgumentParser:
    """Arma el parser; allow_abbrev=False evita que '--conf' active --confirmar-api-real."""
    parser = argparse.ArgumentParser(
        description="Responde una pregunta con el servicio RAG (Gemini + Claude), citando fuentes (con costo).",
        allow_abbrev=False,
    )
    parser.add_argument("--pregunta", required=True)
    parser.add_argument("--k", type=int, default=K_POR_DEFECTO)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_POR_DEFECTO)
    parser.add_argument("--directorio", default=str(RAIZ / "vectorstore"))
    parser.add_argument("--coleccion", default="corpus_miniapp")
    parser.add_argument("--registro-json", action="store_true", help="escribe en stderr un log JSON por evento, sin contenido")
    parser.add_argument("--confirmar-api-real", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Consulta el servicio contra las APIs reales solo con --confirmar-api-real."""
    args = _crear_parser().parse_args(argv)
    if not args.confirmar_api_real:
        print(
            "Este script hace llamadas REALES a las APIs de Gemini (embeddings) y de Claude, con costo. "
            "Para ejecutarlo pasa --confirmar-api-real.",
            file=sys.stderr,
        )
        return 2
    if not 1 <= args.k <= MAX_K:
        return _error(f"--k debe estar entre 1 y {MAX_K}.")
    if not 1 <= args.max_tokens <= MAX_TOKENS_MAXIMO:
        return _error(f"--max-tokens debe estar entre 1 y {MAX_TOKENS_MAXIMO}.")
    if not args.pregunta.strip():
        return _error("--pregunta no puede estar vacía.")
    directorio = os.path.expanduser(args.directorio)

    # import diferido: sin el flag no se carga langchain ni se leen las claves
    try:
        import embeddings_gemini
        import indexacion
        import modelo_langchain
        import recuperacion
        import registro_estructurado
        import servicio_rag
        from api_claude_refactor import INPUT_PRICING, OUTPUT_PRICING

        logger = registro_estructurado.crear_logger() if args.registro_json else None
        try:
            vectorstore = indexacion.cargar_vectorstore(embeddings_gemini.crear_embeddings(), directorio, args.coleccion)
        except Exception as error:
            return _error(f"no se pudo abrir el índice: {error}")
        try:
            modelo = modelo_langchain.crear_modelo(max_tokens=args.max_tokens)
        except Exception as error:
            return _error(f"no se pudo crear el modelo: {error}")
        try:
            salida = servicio_rag.consultar(args.pregunta, vectorstore, modelo, args.k, logger, (INPUT_PRICING, OUTPUT_PRICING))
        except ValueError:  # sin el texto de la excepción: podría traer parte de la pregunta
            return _error(f"pregunta o parámetros inválidos (la pregunta admite hasta {recuperacion.MAX_LONGITUD_PREGUNTA} caracteres).")
        except Exception as error:
            return _error(f"falló la consulta ({type(error).__name__}).")
    except KeyboardInterrupt:
        print("Interrumpido por la persona.", file=sys.stderr)
        return 130
    print(recuperacion.limpiar_para_consola(salida["respuesta"]))
    if salida["estado"] != "respondida":
        print(f"Aviso: la consulta terminó sin respuesta citada (estado {salida['estado']}).", file=sys.stderr)
    if salida["truncado"]:
        print(f"Aviso: la respuesta salio cortada por --max-tokens ({args.max_tokens}); "
              f"prueba subirlo (máximo {MAX_TOKENS_MAXIMO}).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
