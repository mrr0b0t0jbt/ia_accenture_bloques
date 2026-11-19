"""Mide con la API REAL cuántas respuestas del clasificador del router vienen mal formateadas (Bloque 30, Unidad 2.6).

Para cada ejemplo de JSON/dataset_tuning_clasificacion.json (y cada repetición) hace UNA llamada a Claude con
enrutamiento.PLANTILLA_CLASIFICACION (solo esa plantilla y el modelo, no las rutas del router) y comprueba si la respuesta CRUDA
es exactamente una categoría (formato estricto), si queda válida tras la normalización del router (formato efectivo) y si acierta.
La regla de la clase: si el 15 % o más incumple el formato, el problema es de formato o tono. Hace llamadas REALES, con costo:
el costo es una estimación con los precios del proyecto. La respuesta del modelo es dato: se sanea antes de guardarla o mostrarla.

Los topes están en el código: MAX_LLAMADAS, el rango de --repeticiones, --max-tokens y --tope-usd (se comprueba antes de cada llamada).
El JSON de --salida se reemplaza de forma atómica; se escribe al final y en cualquier salida anticipada una vez creado el modelo.

Cómo correrlo (desde cualquier directorio):

    python scripts/medir_clasificador_router.py --confirmar-api-real
    python scripts/medir_clasificador_router.py --confirmar-api-real --repeticiones 2 --max-tokens 32 --tope-usd 0.02
"""
import argparse
import datetime
import json
import os
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

MAX_LLAMADAS = 75
MAX_REPETICIONES = 3
MAX_TOKENS_MAXIMO = 256
MAX_TOKENS_POR_DEFECTO = 64
TOPE_USD_MAXIMO = 0.05
TOPE_USD_POR_DEFECTO = 0.03
MAX_CARACTERES_ERROR = 300
MODELO = "claude-sonnet-5-5"
NOTA = (
    "El costo es una estimacion con los precios del proyecto, no el cobro real. Una sola pasada es una muestra y no estadistica. "
    "El router del SDK puede reintentar. La muestra son los ejemplos del dataset de tuning, que escribio la orquestadora y no "
    "mensajes reales de un usuario."
)


def _limpiar(texto) -> str:
    """Reemplaza por un espacio todo carácter no imprimible (incluidos los saltos de línea)."""
    return "".join(c if c.isprintable() else " " for c in str(texto))


def _imprimir(linea: str) -> None:
    print(_limpiar(linea))


def _error(mensaje: str, codigo: int = 1) -> int:
    print(f"Error: {_limpiar(mensaje)}", file=sys.stderr)
    return codigo


def _ahora() -> datetime.datetime:
    """Hora actual en UTC con zona (se parchea en las pruebas)."""
    return datetime.datetime.now(datetime.timezone.utc)


def _ruta_para_registro(ruta: str) -> str:
    """Ruta relativa a la raíz del repo; fuera de ella solo el nombre del archivo (el JSON se versiona)."""
    try:
        return str(Path(ruta).resolve().relative_to(RAIZ))
    except ValueError:
        return Path(ruta).name


def _problema_de_salida(salida: str, dataset: str) -> str | None:
    """Motivo por el que --salida no es seguro (pisaría el dataset o no se podría escribir), o None."""
    if not salida:
        return "--salida no puede estar vacío."
    carpeta = os.path.dirname(os.path.abspath(salida))
    if not os.path.isdir(carpeta) or os.path.isdir(salida):
        return f"--salida no es un archivo en un directorio existente: {salida}"
    if os.path.realpath(salida) == os.path.realpath(dataset) or (os.path.exists(salida) and os.path.samefile(salida, dataset)):
        return "--salida no puede ser el mismo archivo que --dataset."
    if not os.access(carpeta, os.W_OK | os.X_OK) or (os.path.exists(salida) and not os.access(salida, os.W_OK)):
        return f"no se puede escribir en --salida: {salida}"
    return None


def _escribir_json(ruta: str, contenido: dict) -> None:
    """Escribe un temporal en la misma carpeta y lo reemplaza con os.replace; no deja .tmp si falla."""
    carpeta = os.path.dirname(os.path.abspath(ruta))
    descriptor, temporal = tempfile.mkstemp(dir=carpeta, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as archivo:
            json.dump(contenido, archivo, ensure_ascii=False, indent=2)
            archivo.write("\n")
        os.replace(temporal, ruta)
    except BaseException:
        if os.path.exists(temporal):
            os.remove(temporal)
        raise


def _es_entero(valor) -> bool:
    return isinstance(valor, int) and not isinstance(valor, bool)


def _llamar(modelo, prompt: str, esperada: str) -> tuple[dict, dict | None]:
    """Una llamada: devuelve (campos del resultado, uso o None); un error del modelo queda en el campo error."""
    import medicion_clasificador
    import recuperacion

    campos = {"respuesta": None, "exacta": False, "normalizada": None, "valida": False, "correcta": False, "tokens": None,
              "costo_estimado": None, "truncado": False, "error": None}
    try:
        mensaje = modelo.invoke(prompt)
    except Exception as fallo:
        campos["error"] = f"{type(fallo).__name__}: {str(fallo)}"[:MAX_CARACTERES_ERROR]
        return campos, None
    campos.update(medicion_clasificador.clasificar_respuesta(recuperacion.texto_de_respuesta(mensaje), esperada))
    campos["truncado"] = (getattr(mensaje, "response_metadata", None) or {}).get("stop_reason") == "max_tokens"
    uso = getattr(mensaje, "usage_metadata", None) or {}
    entrada, salida = uso.get("input_tokens"), uso.get("output_tokens")
    if not _es_entero(entrada) or entrada == 0:
        return campos, None  # el cliente real deja el uso en ceros: sin uso el gasto no se controla
    salida = salida if _es_entero(salida) else 0
    campos["tokens"] = {"input_tokens": entrada, "output_tokens": salida, "total_tokens": entrada + salida}
    campos["costo_estimado"] = medicion_clasificador.costo_de_llamada(entrada, salida)
    return campos, campos["tokens"]


def _correr(ejemplos: list[dict], args, modelo, resultados: list[dict], estado: dict) -> int:
    """Una llamada por ejemplo y repetición; devuelve el código de salida (Ctrl-C se propaga)."""
    from enrutamiento import PLANTILLA_CLASIFICACION

    gasto = 0.0
    for repeticion in range(1, args.repeticiones + 1):
        for ejemplo in ejemplos:
            # tolerancia: el costo en punto flotante puede quedar un epsilon por debajo del tope exacto
            if gasto >= args.tope_usd - 1e-9:
                print(f"\nLa ejecución se detuvo al alcanzar el tope de gasto (USD {args.tope_usd}).")
                estado["incompleto"] = True
                return 0
            campos, tokens = _llamar(modelo, PLANTILLA_CLASIFICACION.format(mensaje=ejemplo["texto"]), ejemplo["categoria"])
            resultados.append({
                "id": ejemplo["id"], "repeticion": repeticion, "categoria_esperada": ejemplo["categoria"],
                "prioridad_esperada": ejemplo["prioridad"], **campos,
            })
            gasto += campos["costo_estimado"] or 0.0
            if campos["error"] is None and tokens is None:
                estado["incompleto"] = True
                estado["error"] = "Claude no informó los tokens usados: sin uso no se controla el gasto; se detiene la corrida."
                return _error(estado["error"])
    return 0


def _porcentaje(fallos: int, evaluadas: int) -> str:
    return f"{100 * fallos / evaluadas:.1f}" if evaluadas else "n/d"


def _informar(resumen: dict, costo_total: float) -> None:
    n = resumen["evaluadas"]
    _imprimir(f"Llamadas: {resumen['llamadas']} (con error: {resumen['con_error']})")
    _imprimir(f"Respuestas evaluadas: {n}")
    _imprimir(f"Formato estricto (la respuesta es exactamente la categoria): {resumen['fallos_formato_estricto']} fallos de {n} ({_porcentaje(resumen['fallos_formato_estricto'], n)} %)")
    _imprimir(f"Formato efectivo (la categoria queda valida tras normalizar): {resumen['fallos_formato_efectivo']} fallos de {n} ({_porcentaje(resumen['fallos_formato_efectivo'], n)} %)")
    veredicto = {True: "si", False: "no", None: "sin datos"}[resumen["supera_umbral_estricto"]]
    _imprimir(f"Umbral de la clase: {resumen['umbral'] * 100:.0f} %. Supera el umbral (estricto): {veredicto}")
    _imprimir(f"Exactitud de la categoria: {resumen['aciertos']} de {n}")
    for categoria, datos in resumen["por_categoria"].items():
        _imprimir(f"  {categoria}: {datos['aciertos']} de {datos['esperadas']}")
    _imprimir(f"Costo total estimado de Claude: USD {costo_total:.5f}")
    _imprimir(NOTA)


def _crear_parser() -> argparse.ArgumentParser:
    """Arma el parser; allow_abbrev=False evita que '--conf' active --confirmar-api-real."""
    parser = argparse.ArgumentParser(description="Mide el formato y la exactitud del clasificador del router (con costo).", allow_abbrev=False)
    parser.add_argument("--dataset", default=str(RAIZ / "JSON" / "dataset_tuning_clasificacion.json"))
    parser.add_argument("--repeticiones", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS_POR_DEFECTO)
    parser.add_argument("--tope-usd", type=float, default=TOPE_USD_POR_DEFECTO)
    parser.add_argument("--salida", default=str(RAIZ / "docs" / "resultados_clasificador_router.json"))
    parser.add_argument("--confirmar-api-real", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Mide el clasificador contra la API real solo con --confirmar-api-real; 0 si termina, 1 si falla, 2 si los argumentos son inválidos."""
    args = _crear_parser().parse_args(argv)
    if not args.confirmar_api_real:
        print("Este script hace llamadas REALES a la API de Claude, con costo. Para ejecutarlo pasa --confirmar-api-real.", file=sys.stderr)
        return 2
    if not 1 <= args.repeticiones <= MAX_REPETICIONES:
        return _error(f"--repeticiones debe estar entre 1 y {MAX_REPETICIONES}.", 2)
    if not 1 <= args.max_tokens <= MAX_TOKENS_MAXIMO:
        return _error(f"--max-tokens debe estar entre 1 y {MAX_TOKENS_MAXIMO}.", 2)
    if not 0 < args.tope_usd <= TOPE_USD_MAXIMO:
        return _error(f"--tope-usd debe ser mayor que 0 y como máximo {TOPE_USD_MAXIMO}.", 2)
    args.dataset, args.salida = os.path.expanduser(args.dataset), os.path.expanduser(args.salida)
    # imports diferidos: sin la bandera no se carga langchain, no se leen las claves ni el dataset
    import dataset_tuning
    import medicion_clasificador
    import modelo_langchain

    try:
        datos = dataset_tuning.cargar_dataset(args.dataset)
        problemas = dataset_tuning.validar_datos(datos)
        if not problemas:
            ejemplos_tuning = dataset_tuning.construir_ejemplos(datos)
            problemas = dataset_tuning.validar_dataset(ejemplos_tuning)
    except ValueError as error:
        return _error(str(error))
    if problemas:
        return _error(f"el dataset tiene {len(problemas)} problemas; el primero: {problemas[0]}")
    ejemplos = [
        {"id": e["id"], "texto": t.text_input, "categoria": e["salida"]["categoria"], "prioridad": e["salida"]["prioridad"]}
        for e, t in zip(datos["ejemplos"], ejemplos_tuning)
    ]
    if len(ejemplos) * args.repeticiones > MAX_LLAMADAS:
        return _error(f"{len(ejemplos)} ejemplos x {args.repeticiones} repeticiones son {len(ejemplos) * args.repeticiones} llamadas; el máximo es {MAX_LLAMADAS}.")
    problema = _problema_de_salida(args.salida, args.dataset)
    if problema:
        return _error(problema)
    try:
        modelo = modelo_langchain.crear_modelo(max_tokens=args.max_tokens)
    except ValueError as error:
        return _error(str(error))

    resultados, estado = [], {"incompleto": False, "error": None}
    try:
        codigo = _correr(ejemplos, args, modelo, resultados, estado)
    except KeyboardInterrupt:
        codigo, estado["incompleto"] = 130, True
        print("Interrumpido por la persona. Lo ya medido se guarda.", file=sys.stderr)
    if codigo == 0 and resultados and all(r["error"] for r in resultados):
        estado["error"] = "todas las llamadas a Claude fallaron (ver el campo error de cada resultado)."
        codigo = _error(estado["error"])
    resumen = medicion_clasificador.resumir(resultados)
    costo_total = sum(r["costo_estimado"] or 0.0 for r in resultados)
    _informar(resumen, costo_total)
    try:
        _escribir_json(args.salida, {
            "fecha": _ahora().isoformat(),
            "modelo": MODELO,
            "parametros": {
                "repeticiones": args.repeticiones, "max_tokens": args.max_tokens, "tope_usd": args.tope_usd,
                "dataset": _ruta_para_registro(args.dataset),
            },
            "incompleto": estado["incompleto"],
            "error": estado["error"],
            "resultados": resultados,
            "resumen": resumen,
            "costo_total": costo_total,
            "nota": NOTA,
        })
    except OSError as error:
        return _error(f"no se pudo escribir la salida: {error.strerror or type(error).__name__}")
    return codigo


if __name__ == "__main__":
    sys.exit(main())
