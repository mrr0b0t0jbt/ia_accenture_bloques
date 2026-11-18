"""Validación estricta y balance del dataset de tuning (Bloque 29, Unidad 2.6), en Python puro y SIN ninguna API.

El contenido del archivo es dato, nunca instrucciones: los mensajes que citan algo del archivo usan repr o no lo repiten.
google.genai se importa solo dentro de construir_ejemplos.
"""
import collections
import json
import unicodedata
from pathlib import Path

CLAVES = ("categoria", "prioridad")
VALORES_PERMITIDOS = {"categoria": ("incidente", "vulnerabilidad", "consulta"), "prioridad": ("alta", "media", "baja")}
MIN_EJEMPLOS = 10
MAX_EJEMPLOS = 500
MAX_CARACTERES_TEXTO = 1000
MAX_BYTES_ARCHIVO = 1000000
RAZON_AVISO_BALANCE = 2.0


def normalizar_texto(texto: str) -> str:
    """Une las palabras con un espacio y pasa a casefold (la forma con que se detectan duplicados)."""
    return " ".join(texto.split()).casefold()


def cargar_dataset(ruta) -> dict:
    """Lee el archivo del dataset (comprobando su tamaño antes) y devuelve el objeto JSON; ValueError si no es válido."""
    ruta = Path(ruta)
    if not ruta.is_file():
        raise ValueError(f"el dataset no existe o no es un archivo: {str(ruta)!r}")
    try:
        tamano = ruta.stat().st_size
        if tamano > MAX_BYTES_ARCHIVO:
            raise ValueError(f"el dataset pesa {tamano} bytes; el maximo es {MAX_BYTES_ARCHIVO}")
        contenido = ruta.read_bytes().decode("utf-8")
    except OSError as error:
        raise ValueError(f"no se pudo leer el dataset: {error.strerror or type(error).__name__}") from None
    except UnicodeDecodeError:
        raise ValueError("el dataset no esta en UTF-8") from None
    try:
        datos = json.loads(contenido)
    except (ValueError, RecursionError):
        raise ValueError("el dataset no es un JSON valido") from None
    if not isinstance(datos, dict):
        raise ValueError("el JSON de nivel superior del dataset no es un objeto")
    return datos


def validar_datos(datos) -> list[str]:
    """Devuelve los problemas de forma del archivo ya cargado (lista vacía si todo está en orden)."""
    if not isinstance(datos, dict):
        return ["datos no es un objeto"]
    problemas = []
    ejemplos = datos.get("ejemplos")
    if not isinstance(ejemplos, list):
        problemas.append("falta la clave ejemplos o no es una lista")
        ejemplos = []
    if datos.get("claves") != list(CLAVES):
        problemas.append("claves del archivo distintas de CLAVES")
    if datos.get("valores_permitidos") != {k: list(v) for k, v in VALORES_PERMITIDOS.items()}:
        problemas.append("valores_permitidos del archivo distintos de VALORES_PERMITIDOS")
    ids = set()
    for i, ejemplo in enumerate(ejemplos):
        if not isinstance(ejemplo, dict):
            problemas.append(f"ejemplos[{i}] no es un objeto")
            continue
        id_ = ejemplo.get("id")
        if not isinstance(id_, str) or not id_.strip():
            problemas.append(f"ejemplos[{i}] no tiene id de texto no vacio")
        elif id_ in ids:
            problemas.append(f"ejemplos[{i}] repite el id {id_!r}")
        else:
            ids.add(id_)
        if "texto" not in ejemplo:
            problemas.append(f"ejemplos[{i}] no tiene texto")
        if "salida" not in ejemplo:
            problemas.append(f"ejemplos[{i}] no tiene salida")
    return problemas


def construir_ejemplos(datos) -> list:
    """Convierte los ejemplos del archivo en TuningExample (output = JSON con acentos sin escapar); no modifica datos."""
    problemas = validar_datos(datos)
    if problemas:
        raise ValueError(problemas[0])
    from google.genai import types

    resultado = []
    for i, ejemplo in enumerate(datos["ejemplos"]):
        if not isinstance(ejemplo["texto"], str):
            raise ValueError(f"ejemplos[{i}] tiene un texto que no es de tipo texto")
        resultado.append(types.TuningExample(text_input=ejemplo["texto"], output=json.dumps(ejemplo["salida"], ensure_ascii=False)))
    return resultado


def _parsear(salida: str) -> tuple[object, list[str]]:
    """Parsea el JSON y devuelve (valor, claves repetidas del objeto de nivel superior)."""
    estado = {"repetidas": []}

    def pares(lista):
        objeto, repetidas = {}, []
        for clave, valor in lista:
            if clave in objeto and clave not in repetidas:
                repetidas.append(clave)
            objeto[clave] = valor
        estado["repetidas"] = repetidas  # el objeto de nivel superior es el último en cerrarse
        return objeto

    valor = json.loads(salida, object_pairs_hook=pares)
    return valor, estado["repetidas"]


def _problemas_del_texto(texto) -> list[str]:
    if not isinstance(texto, str):
        return ["text_input no es texto"]
    if not texto.strip():
        return ["text_input vacio"]
    problemas = []
    if len(texto) > MAX_CARACTERES_TEXTO:
        problemas.append(f"text_input supera los {MAX_CARACTERES_TEXTO} caracteres")
    if any(unicodedata.category(c) == "Cc" and c not in "\n\t" for c in texto):
        problemas.append("text_input tiene caracteres de control")
    if not _es_utf8(texto):
        problemas.append("text_input no se puede codificar en UTF-8")
    return problemas


def _es_utf8(texto: str) -> bool:
    """False si el texto trae un surrogate suelto (el SDK falla al serializarlo)."""
    try:
        texto.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _problemas_del_json(salida: str, claves, valores_permitidos) -> list[str]:
    try:
        valor, repetidas = _parsear(salida)
    except (ValueError, RecursionError):
        return ["output no es JSON valido"]
    if not isinstance(valor, dict):
        return ["output no es un objeto JSON"]
    problemas = [f"output repite la clave {clave!r}" for clave in repetidas]
    problemas += [f'output no tiene la clave "{k}"' for k in claves if k not in valor]
    sobran = sorted(set(valor) - set(claves))
    if sobran:
        problemas.append("output tiene claves de mas: " + ", ".join(repr(k) for k in sobran))
    if valores_permitidos is not None:
        for k in claves:
            if k in valor and k in valores_permitidos:
                if not isinstance(valor[k], str) or valor[k] not in valores_permitidos[k]:
                    problemas.append(f'output tiene un valor no permitido para "{k}"')
    if not repetidas and set(valor) == set(claves) and salida != json.dumps({k: valor[k] for k in claves}, ensure_ascii=False):
        problemas.append("output no esta en el formato canonico (claves en orden y separadores por defecto)")
    return problemas


def validar_dataset(examples, claves=CLAVES, valores_permitidos=VALORES_PERMITIDOS) -> list[str]:
    """Devuelve TODOS los problemas de los ejemplos (objetos con text_input y output): globales, luego por índice."""
    n = len(examples)
    problemas = []
    if n < MIN_EJEMPLOS:
        problemas.append(f"el dataset tiene {n} ejemplos; el minimo es {MIN_EJEMPLOS}")
    if n > MAX_EJEMPLOS:
        problemas.append(f"el dataset tiene {n} ejemplos; el maximo es {MAX_EJEMPLOS}")
    vistos = {}
    for i, ejemplo in enumerate(examples):
        texto, salida = ejemplo.text_input, ejemplo.output
        prefijo = f"ejemplo {i}: "
        problemas += [prefijo + p for p in _problemas_del_texto(texto)]
        if not isinstance(salida, str):
            problemas.append(prefijo + "output no es texto")
            continue
        if not salida.strip():
            problemas.append(prefijo + "output vacio")
            continue
        if not _es_utf8(salida):
            problemas.append(prefijo + "output no se puede codificar en UTF-8")
        normalizado =normalizar_texto(texto) if isinstance(texto, str) else ""
        if normalizado:
            j = vistos.setdefault(normalizado, i)
            if j != i:
                sufijo = " con otra salida" if salida != examples[j].output else ""
                problemas.append(f"{prefijo}text_input duplicado del ejemplo {j}{sufijo}")
        problemas += [prefijo + p for p in _problemas_del_json(salida, claves, valores_permitidos)]
    return problemas


def contar_por_clave(examples, clave: str) -> collections.Counter:
    """Cuenta los valores de texto de la clave en los outputs que sean un objeto JSON; ignora en silencio los demás."""
    conteo = collections.Counter()
    for ejemplo in examples:
        try:
            valor, _ = _parsear(ejemplo.output)
        except (ValueError, TypeError, RecursionError):
            continue
        if isinstance(valor, dict) and isinstance(valor.get(clave), str):
            conteo[valor[clave]] += 1
    return conteo


def resumir_balance(conteo, valores_permitidos=None) -> dict:
    """Resume el conteo: ordenado de mayor a menor, máximo, mínimo, razón max/min (None si el mínimo es 0) y ausentes."""
    cuentas = dict(conteo)
    orden = list(valores_permitidos) if valores_permitidos is not None else []
    for valor in orden:
        cuentas.setdefault(valor, 0)
    posicion = {valor: i for i, valor in enumerate(orden)}
    ordenado = sorted(cuentas, key=lambda v: (-cuentas[v], posicion.get(v, len(posicion)), v))
    base = [cuentas[v] for v in orden] if valores_permitidos is not None else list(cuentas.values())
    maximo, minimo = max(cuentas.values(), default=0), min(base, default=0)
    return {
        "conteo": {v: cuentas[v] for v in ordenado},
        "maximo": maximo,
        "minimo": minimo,
        "razon": round(maximo / minimo, 1) if minimo else None,
        "ausentes": [v for v in orden if cuentas[v] == 0],
    }


def avisos_de_balance(resumenes: dict[str, dict]) -> list[str]:
    """Devuelve avisos (no errores) por clave: valores sin ejemplos o razón max/min mayor que RAZON_AVISO_BALANCE."""
    avisos = []
    for clave, resumen in resumenes.items():
        razon = resumen["razon"]
        if razon is None and resumen["ausentes"]:
            avisos.append(f"aviso: {clave} no tiene ejemplos de: {', '.join(resumen['ausentes'])}")
        elif razon is not None and razon > RAZON_AVISO_BALANCE:
            avisos.append(
                f"aviso: la razon max/min de {clave} es {razon}x (mayor que {RAZON_AVISO_BALANCE}): "
                "conviene agregar ejemplos a los valores menos representados"
            )
    return avisos
