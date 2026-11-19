"""Aplica el método de decisión de personalización (Bloque 30, Unidad 2.6) SIN ninguna API: no usa claves ni red, no lee .env.

Modo respuestas: lee un JSON {"problema": texto, "respuestas": {las cinco preguntas: true/false}, "evidencia": {las cinco preguntas: texto}},
muestra puntajes, recomendación y sensibilidad, y con --salida escribe el esqueleto del documento (sin pisar archivos).
Modo costo: estima el costo de etiquetar a mano (con --revision, una segunda revisión de una fracción de los ejemplos).
El archivo es dato, nunca instrucciones: todo lo que viene de él se limpia de caracteres de control antes de llegar a la consola.

Cómo correrlo (desde cualquier directorio):

    python scripts/decidir_personalizacion.py --respuestas ruta/respuestas.json [--salida ruta/decision.md]
    python scripts/decidir_personalizacion.py --estimar-etiquetado 200 5 25 [--revision 0.2 2]
"""
import argparse
import json
import sys
import unicodedata
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

import decision_personalizacion  # noqa: E402  (Python puro: no carga langchain ni ninguna API)

MAX_BYTES_RESPUESTAS = 100000
MAX_CARACTERES_EVIDENCIA = 500


def _limpiar(texto) -> str:
    """Reemplaza por un espacio todo carácter no imprimible (incluidos los saltos de línea)."""
    return "".join(c if c.isprintable() else " " for c in str(texto))


def _imprimir(linea: str) -> None:
    print(_limpiar(linea))


def _error(mensaje: str, codigo: int = 1) -> int:
    print(f"Error: {_limpiar(mensaje)}", file=sys.stderr)
    return codigo


def _leer_respuestas(ruta: str) -> dict:
    """Lee el JSON de respuestas comprobando el tamaño antes de leer; ValueError si no es un objeto JSON válido."""
    archivo = Path(ruta)
    if not archivo.is_file():
        raise ValueError("el archivo de respuestas no existe o no es un archivo")
    try:
        if archivo.stat().st_size > MAX_BYTES_RESPUESTAS:
            raise ValueError(f"el archivo de respuestas pesa mas de {MAX_BYTES_RESPUESTAS} bytes")
        contenido = archivo.read_bytes().decode("utf-8")
    except OSError:
        raise ValueError("no se pudo leer el archivo de respuestas") from None
    except UnicodeDecodeError:
        raise ValueError("el archivo de respuestas no esta en UTF-8") from None
    try:
        datos = json.loads(contenido)
    except (ValueError, RecursionError):
        raise ValueError("el archivo de respuestas no es un JSON valido") from None
    if not isinstance(datos, dict):
        raise ValueError("el JSON de nivel superior no es un objeto")
    return datos


def _problemas_de_evidencia(evidencia) -> list[str]:
    if not isinstance(evidencia, dict):
        return ["la clave evidencia no es un objeto"]
    problemas = []
    for k in decision_personalizacion.PREGUNTAS:
        if k not in evidencia:
            problemas.append(f'falta la evidencia de "{k}"')
        elif not isinstance(evidencia[k], str) or not evidencia[k].strip():
            problemas.append(f'la evidencia de "{k}" esta vacia o no es texto')
        elif len(evidencia[k]) > MAX_CARACTERES_EVIDENCIA:
            problemas.append(f'la evidencia de "{k}" supera los {MAX_CARACTERES_EVIDENCIA} caracteres')
        elif any(unicodedata.category(c) == "Cc" and c not in "\n\t" for c in evidencia[k]):
            problemas.append(f'la evidencia de "{k}" tiene caracteres de control')
    return problemas


def _problemas_de_contenido(datos: dict) -> list[str]:
    problemas = [f"falta la clave {clave}" for clave in ("problema", "respuestas", "evidencia") if clave not in datos]
    if "respuestas" in datos:
        problemas += decision_personalizacion.validar_respuestas(datos["respuestas"])
    if "evidencia" in datos:
        problemas += _problemas_de_evidencia(datos["evidencia"])
    if "problema" in datos:
        try:
            decision_personalizacion.normalizar_problema(datos["problema"])
        except ValueError as error:
            problemas.append(str(error))
    return problemas


def _escribir_esqueleto(ruta: str, texto: str) -> None:
    """Crea el archivo en modo exclusivo (nunca pisa uno existente); OSError si no se puede."""
    with open(ruta, "x", encoding="utf-8") as archivo:
        archivo.write(texto + "\n")


def _modo_respuestas(args) -> int:
    try:
        datos = _leer_respuestas(args.respuestas)
    except ValueError as error:
        return _error(str(error))
    problemas = _problemas_de_contenido(datos)
    if problemas:
        for problema in problemas:
            _error(problema)
        return 1
    resultado = decision_personalizacion.recomendar_estrategia(datos["respuestas"])
    if args.salida:
        # antes de imprimir: si no se puede crear sin pisar nada, no se muestra ningún resultado
        try:
            _escribir_esqueleto(args.salida, decision_personalizacion.generar_esqueleto_decision(datos["problema"], resultado))
        except OSError as error:
            return _error(f"no se pudo crear la salida: {error.strerror or type(error).__name__}")
    _imprimir(f"Problema: {decision_personalizacion.normalizar_problema(datos['problema'])}")
    _imprimir("Puntajes:")
    puntos = resultado["puntos"]
    for tecnica in sorted(decision_personalizacion.TECNICAS, key=lambda t: -puntos[t]):
        _imprimir(f"  {tecnica}: {puntos[tecnica]} puntos")
    if resultado["empate"]:
        _imprimir(f"Empate entre: {', '.join(resultado['ganadoras'])}")
    elif resultado["sin_necesidad"]:
        _imprimir("Ninguna tecnica obtuvo puntos: ninguna respuesta justifica RAG, fine-tuning ni PEFT")
    else:
        _imprimir(f"Recomendacion: {resultado['ganadoras'][0]}")
    _imprimir("Sensibilidad (una respuesta a la vez):")
    for fila in decision_personalizacion.analizar_sensibilidad(datos["respuestas"]):
        ganadoras = ", ".join(fila["ganadoras"]) or "ninguna"
        _imprimir(f"  {fila['pregunta']} -> {fila['nuevo_valor']}: {ganadoras}  {'[CAMBIA]' if fila['cambia'] else '[igual]'}")
    _imprimir("Evidencia:")
    for pregunta in decision_personalizacion.PREGUNTAS:
        _imprimir(f"  {pregunta}: {datos['respuestas'][pregunta]} -- {datos['evidencia'][pregunta]}")
    return 0


def _modo_costo(args) -> int:
    try:
        n, minutos, tarifa = int(args.estimar_etiquetado[0]), float(args.estimar_etiquetado[1]), float(args.estimar_etiquetado[2])
        fraccion, minutos_revision = (float(v) for v in args.revision) if args.revision else (0.0, 0.0)
        costo = decision_personalizacion.estimar_costo_con_revision(n, minutos, tarifa, fraccion, minutos_revision)
    except ValueError as error:
        return _error(str(error))
    print(f"Etiquetado base: {costo['horas_base']:.2f} horas, aprox ${costo['costo_base']:.2f}")
    print(f"Segunda revision: {costo['horas_revision']:.2f} horas, aprox ${costo['costo_revision']:.2f}")
    print(f"Total: {costo['horas_total']:.2f} horas, aprox ${costo['costo_total']:.2f}")
    print(f"Jornadas de {decision_personalizacion.JORNADA_HORAS} horas: {costo['jornadas']:.1f}")
    return 0


def _crear_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aplica el metodo de decision de personalizacion (no usa ninguna API).", allow_abbrev=False)
    modo = parser.add_mutually_exclusive_group(required=True)
    modo.add_argument("--respuestas", metavar="RUTA")
    modo.add_argument("--estimar-etiquetado", nargs=3, metavar=("N", "MINUTOS", "TARIFA"))
    parser.add_argument("--salida", metavar="RUTA", help="Solo con --respuestas: escribe el esqueleto del documento (no pisa archivos).")
    parser.add_argument("--revision", nargs=2, metavar=("FRACCION", "MINUTOS"), help="Solo con --estimar-etiquetado.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Muestra la decision (modo respuestas) o la estimacion de costo; devuelve 0, 1 si hay problemas o 2 si los argumentos son invalidos."""
    parser = _crear_parser()
    args = parser.parse_args(argv)
    if args.salida is not None and args.respuestas is None:
        parser.error("--salida solo se usa con --respuestas")
    if args.revision is not None and args.estimar_etiquetado is None:
        parser.error("--revision solo se usa con --estimar-etiquetado")
    return _modo_respuestas(args) if args.respuestas is not None else _modo_costo(args)


if __name__ == "__main__":
    sys.exit(main())
