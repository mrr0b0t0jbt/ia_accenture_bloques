"""Registro de versiones de prompts (Bloque 32, Unidad 2.7): JSONL de solo anexar con huella, autor, fecha UTC, nota, modelo y parametros.

Cada línea es una versión de un prompt: se agrega al final, nunca se reescribe. La huella (sha256 de 12 caracteres, la misma de
registro_estructurado) permite saber si el prompt que está en el código es el último registrado. Sin red ni claves.

Límites conocidos: no hay bloqueo de archivo (dos escrituras simultáneas pueden mezclarse); la huella sin sal no impide que
alguien edite el texto Y la huella a mano (la cadena de huella_anterior y el control de versiones de git lo muestran, no lo evitan).
"""
import difflib
import json
import math
import os
from datetime import datetime, timezone

from registro_estructurado import huella

RUTA_REGISTRO_POR_DEFECTO = "docs/registro_prompts.jsonl"
MAX_TEXTO = 100_000
MAX_NOMBRE = 200
MAX_AUTOR = 200
MAX_NOTA = 1000
MAX_MODELO = 200
MAX_PARAMETROS = 20
CLAVES = ("nombre", "version", "huella", "huella_anterior", "texto", "autor", "fecha_utc", "nota", "modelo", "parametros")

huella_prompt = huella


def _texto(valor, campo: str, maximo: int) -> str:
    if not isinstance(valor, str) or not valor.strip() or len(valor) > maximo:
        raise ValueError(f"{campo} invalido: debe ser un texto no vacio de hasta {maximo} caracteres")
    return valor


def _validar_parametros(parametros) -> dict | None:
    if parametros is None:
        return None
    if not isinstance(parametros, dict) or len(parametros) > MAX_PARAMETROS:
        raise ValueError(f"parametros invalidos: debe ser un diccionario de hasta {MAX_PARAMETROS} claves")
    for clave, valor in parametros.items():
        if not isinstance(clave, str) or not clave or type(valor) not in (str, int, float, bool, type(None)):
            raise ValueError("parametros invalidos: claves de texto y valores simples (texto, numero, booleano o null)")
        if isinstance(valor, float) and not math.isfinite(valor):
            raise ValueError("parametros invalidos: numeros finitos")
    return dict(parametros)


def leer_registro(ruta: str) -> list[dict]:
    """Lee el registro completo y lo valida línea por línea; un archivo que no existe es un registro vacío."""
    if not os.path.exists(ruta):
        return []
    with open(ruta, encoding="utf-8", newline="") as archivo:
        lineas = archivo.read().split("\n")
    if lineas and lineas[-1] == "":
        lineas.pop()
    entradas = []
    for numero, linea in enumerate(lineas, start=1):
        try:
            entrada = json.loads(linea)
        except ValueError:
            raise ValueError(f"linea {numero}: no es JSON valido") from None
        if not isinstance(entrada, dict) or tuple(entrada) != CLAVES:
            raise ValueError(f"linea {numero}: las claves no son las esperadas")
        if not (type(entrada["version"]) is int and entrada["version"] >= 1):
            raise ValueError(f"linea {numero}: version invalida")
        if not all(isinstance(entrada[c], str) for c in ("nombre", "huella", "texto", "autor", "fecha_utc", "nota")):
            raise ValueError(f"linea {numero}: campos de texto invalidos")
        if entrada["huella_anterior"] is not None and not isinstance(entrada["huella_anterior"], str):
            raise ValueError(f"linea {numero}: huella_anterior invalida")
        if entrada["huella"] != huella_prompt(entrada["texto"]):
            raise ValueError(f"linea {numero}: la huella no coincide con el texto")
        try:
            datetime.strptime(entrada["fecha_utc"], "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            raise ValueError(f"linea {numero}: fecha_utc invalida") from None
        entradas.append(entrada)
    return entradas


def ultima(entradas: list[dict], nombre: str) -> dict | None:
    """Devuelve la versión más reciente del prompt con ese nombre, o None."""
    candidatas = [e for e in entradas if e["nombre"] == nombre]
    return candidatas[-1] if candidatas else None


def registrar_version(
    ruta: str, nombre: str, texto: str, autor: str, nota: str, modelo: str | None = None, parametros: dict | None = None, ahora: datetime | None = None
) -> dict:
    """Agrega UNA línea al final del registro y devuelve la entrada; rechaza una versión idéntica a la última del mismo nombre."""
    _texto(nombre, "nombre", MAX_NOMBRE)
    _texto(texto, "texto", MAX_TEXTO)
    _texto(autor, "autor", MAX_AUTOR)
    _texto(nota, "nota", MAX_NOTA)
    if modelo is not None:
        _texto(modelo, "modelo", MAX_MODELO)
    parametros = _validar_parametros(parametros)
    if ahora is None:
        ahora = datetime.now(timezone.utc)
    if not isinstance(ahora, datetime) or ahora.utcoffset() is None:
        raise ValueError("ahora invalido: debe ser un datetime con zona horaria")
    if not os.path.isdir(os.path.dirname(os.path.abspath(ruta))):
        raise ValueError("el directorio del registro no existe")
    entradas = leer_registro(ruta)
    problemas = problemas_del_registro(entradas)
    if problemas:
        raise ValueError(f"el registro tiene problemas: {problemas[0]}")
    anterior = ultima(entradas, nombre)
    nueva_huella = huella_prompt(texto)
    if anterior is not None and anterior["huella"] == nueva_huella:
        raise ValueError("sin cambios: la huella ya es la de la ultima version")
    entrada = {
        "nombre": nombre,
        "version": 1 if anterior is None else anterior["version"] + 1,
        "huella": nueva_huella,
        "huella_anterior": None if anterior is None else anterior["huella"],
        "texto": texto,
        "autor": autor,
        "fecha_utc": ahora.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nota": nota,
        "modelo": modelo,
        "parametros": parametros,
    }
    with open(ruta, "a", encoding="utf-8", newline="") as archivo:
        archivo.write(json.dumps(entrada, ensure_ascii=True) + "\n")
    return entrada


def problemas_del_registro(entradas: list[dict]) -> list[str]:
    """Lista los problemas de la cadena de versiones de cada nombre (huecos, huella_anterior que no encadena, fechas que retroceden)."""
    problemas = []
    ultimas: dict[str, dict] = {}
    for numero, entrada in enumerate(entradas, start=1):
        previa = ultimas.get(entrada["nombre"])
        esperada = 1 if previa is None else previa["version"] + 1
        if entrada["version"] != esperada:
            problemas.append(f"linea {numero}: version {entrada['version']} en lugar de {esperada}")
        if entrada["huella_anterior"] != (None if previa is None else previa["huella"]):
            problemas.append(f"linea {numero}: huella_anterior no encadena con la version previa")
        if previa is not None and entrada["fecha_utc"] < previa["fecha_utc"]:
            problemas.append(f"linea {numero}: la fecha retrocede respecto de la version previa")
        ultimas[entrada["nombre"]] = entrada
    return problemas


def comparar_con_codigo(entradas: list[dict], nombre: str, texto_actual: str) -> dict:
    """Compara la huella del prompt que está en el código con la de la última versión registrada."""
    registrada = ultima(entradas, nombre)
    huella_codigo = huella_prompt(texto_actual)
    if registrada is None:
        return {"estado": "sin_registro", "huella_codigo": huella_codigo, "huella_registrada": None, "version": None}
    estado = "igual" if registrada["huella"] == huella_codigo else "cambiado"
    return {"estado": estado, "huella_codigo": huella_codigo, "huella_registrada": registrada["huella"], "version": registrada["version"]}


def diff_prompts(anterior: str, nueva: str) -> list[str]:
    """Diff por palabras entre dos versiones (el mismo mecanismo de la clase: difflib.unified_diff sobre split())."""
    return list(difflib.unified_diff(anterior.split(), nueva.split(), fromfile="version_anterior", tofile="version_nueva", lineterm=""))
