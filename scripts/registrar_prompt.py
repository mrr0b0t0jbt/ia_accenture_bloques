"""Registro de versiones del prompt del RAG (Bloque 32): lista, verifica contra el código, registra una versión nueva y muestra el diff.

No llama a ninguna API ni lee claves. Escribe SOLO con --registrar, y solo agrega una línea a docs/registro_prompts.jsonl.

Cómo correrlo (desde cualquier directorio):

    python scripts/registrar_prompt.py --listar
    python scripts/registrar_prompt.py --verificar          # sale con 1 si el prompt del código no es el último registrado
    python scripts/registrar_prompt.py --diff
    python scripts/registrar_prompt.py --registrar --origen actual --autor "nombre" --nota "qué cambió y por qué"
"""
import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import registro_prompts  # noqa: E402

NOMBRE_PROMPT = "prompt_rag"
ORIGENES = ("actual", "plantilla_un_mensaje")


def textos_del_codigo() -> dict[str, str]:
    """Texto de cada prompt del código (importa langchain_core solo cuando se necesita)."""
    import defensa_inyeccion
    import recuperacion

    return {
        "actual": defensa_inyeccion.PROMPT_SISTEMA + "\n" + defensa_inyeccion.PLANTILLA_USUARIO,
        "plantilla_un_mensaje": recuperacion.PROMPT_TEMPLATE,
    }


def _argumentos(argv):
    parser = argparse.ArgumentParser(description="Registro de versiones del prompt del RAG (sin API).")
    accion = parser.add_mutually_exclusive_group(required=True)
    accion.add_argument("--listar", action="store_true", help="muestra las versiones registradas")
    accion.add_argument("--verificar", action="store_true", help="compara el prompt del código con el último registrado")
    accion.add_argument("--diff", action="store_true", help="diff entre las dos últimas versiones")
    accion.add_argument("--registrar", action="store_true", help="agrega una versión (exige --origen, --autor y --nota)")
    parser.add_argument("--registro", default=str(RAIZ / registro_prompts.RUTA_REGISTRO_POR_DEFECTO), help="ruta del JSONL")
    parser.add_argument("--origen", choices=ORIGENES)
    parser.add_argument("--autor")
    parser.add_argument("--nota")
    parser.add_argument("--modelo")
    parser.add_argument("--parametros-json", help="objeto JSON con los parámetros del modelo")
    argumentos = parser.parse_args(argv)
    if argumentos.registrar and not (argumentos.origen and argumentos.autor and argumentos.nota):
        parser.error("--registrar exige --origen, --autor y --nota")
    return argumentos


def main(argv=None) -> int:
    argumentos = _argumentos(argv)
    try:
        entradas = registro_prompts.leer_registro(argumentos.registro)
    except ValueError as error:
        print(f"Registro invalido: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"No se pudo leer el registro: {type(error).__name__}.", file=sys.stderr)
        return 1
    if argumentos.listar:
        for e in entradas:
            print(f"{e['nombre']} v{e['version']} {e['huella']} {e['fecha_utc']} {e['autor']} - {e['nota']}")
        return 0
    if argumentos.diff:
        versiones = [e for e in entradas if e["nombre"] == NOMBRE_PROMPT]
        if len(versiones) < 2:
            print("Hay menos de dos versiones registradas: no hay nada que comparar.", file=sys.stderr)
            return 1
        print("\n".join(registro_prompts.diff_prompts(versiones[-2]["texto"], versiones[-1]["texto"])))
        return 0
    if argumentos.verificar:
        problemas = registro_prompts.problemas_del_registro(entradas)
        for problema in problemas:
            print(f"Problema en el registro: {problema}", file=sys.stderr)
        resultado = registro_prompts.comparar_con_codigo(entradas, NOMBRE_PROMPT, textos_del_codigo()["actual"])
        if resultado["estado"] == "igual" and not problemas:
            print(f"OK: el prompt del codigo es la version {resultado['version']} ({resultado['huella_codigo']}).")
            return 0
        if resultado["estado"] == "sin_registro":
            print("El prompt del codigo no tiene ninguna version registrada.", file=sys.stderr)
        elif resultado["estado"] == "cambiado":
            print(f"El prompt del codigo ({resultado['huella_codigo']}) no es el ultimo registrado ({resultado['huella_registrada']}).", file=sys.stderr)
        return 1
    parametros = None
    if argumentos.parametros_json is not None:
        try:
            parametros = json.loads(argumentos.parametros_json)
        except ValueError:
            print("--parametros-json no es JSON valido.", file=sys.stderr)
            return 1
    try:
        entrada = registro_prompts.registrar_version(
            argumentos.registro, NOMBRE_PROMPT, textos_del_codigo()[argumentos.origen], argumentos.autor, argumentos.nota, argumentos.modelo, parametros
        )
    except ValueError as error:
        print(f"No se registro: {error}", file=sys.stderr)
        return 1
    except OSError as error:
        print(f"No se registro: no se pudo escribir el registro ({type(error).__name__}).", file=sys.stderr)
        return 1
    print(f"Registrada {entrada['nombre']} v{entrada['version']} ({entrada['huella']}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
