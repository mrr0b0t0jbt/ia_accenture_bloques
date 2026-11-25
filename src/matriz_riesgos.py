"""Matriz de riesgos del proyecto (Bloque 32, Unidad 2.7): puntaje, categoria y validacion contra el codigo.

La matriz vive en docs/matriz_riesgos.json (una fila por riesgo). Este modulo calcula el puntaje (probabilidad por impacto), la
categoria (cortes 15, 9 y 4, fijados de antemano) y si una fila esta al limite de otra categoria, y comprueba que cada evidencia
("ruta.py::simbolo") existe de verdad en el codigo, leyendo el archivo con ast (no importa nada). Sin red ni claves.

Limites: probabilidad e impacto son estimaciones, no mediciones; que el simbolo exista no prueba que la mitigacion funcione.
"""
import ast
import json
import os

RUTA_MATRIZ_POR_DEFECTO = "docs/matriz_riesgos.json"
CLAVES = ("riesgo", "probabilidad", "impacto", "mitigacion", "evidencias", "supervision", "nota")
CORTES = (("CRITICO", 15), ("ALTO", 9), ("MEDIO", 4))
MAX_TEXTO = 500
MAX_EVIDENCIAS = 10
MAX_FILAS = 100


def _es_entero(valor) -> bool:
    return isinstance(valor, int) and not isinstance(valor, bool)


def _validar_escala(valor, campo: str) -> None:
    if not _es_entero(valor) or not 1 <= valor <= 5:
        raise ValueError(f"{campo} debe ser un entero de 1 a 5")


def puntaje_riesgo(probabilidad, impacto) -> int:
    """Probabilidad por impacto (ambos enteros de 1 a 5)."""
    _validar_escala(probabilidad, "probabilidad")
    _validar_escala(impacto, "impacto")
    return probabilidad * impacto


def categoria_riesgo(puntaje) -> str:
    """CRITICO desde 15, ALTO desde 9, MEDIO desde 4 y BAJO por debajo."""
    if not _es_entero(puntaje) or not 1 <= puntaje <= 25:
        raise ValueError("puntaje debe ser un entero de 1 a 25")
    for nombre, corte in CORTES:
        if puntaje >= corte:
            return nombre
    return "BAJO"


def esta_en_el_limite(probabilidad, impacto) -> bool:
    """True si mover probabilidad o impacto en un punto (dentro de 1 a 5) cambia la categoria."""
    actual = categoria_riesgo(puntaje_riesgo(probabilidad, impacto))
    vecinos = ((probabilidad + 1, impacto), (probabilidad - 1, impacto), (probabilidad, impacto + 1), (probabilidad, impacto - 1))
    for p2, i2 in vecinos:
        if 1 <= p2 <= 5 and 1 <= i2 <= 5 and categoria_riesgo(puntaje_riesgo(p2, i2)) != actual:
            return True
    return False


def _texto_valido(valor, vacio_permitido: bool = False) -> bool:
    if not isinstance(valor, str) or len(valor) > MAX_TEXTO:
        return False
    return vacio_permitido or bool(valor.strip())


def problemas_de_fila(fila, numero: int) -> list[str]:
    """Problemas de forma de una fila; cada mensaje cita solo el numero de fila, nunca el contenido."""
    if not isinstance(fila, dict):
        return [f"fila {numero}: no es un objeto"]
    if tuple(fila) != CLAVES:
        return [f"fila {numero}: las claves no son las esperadas o no estan en orden"]
    problemas = []
    if not _texto_valido(fila["riesgo"]):
        problemas.append(f"fila {numero}: riesgo vacio, muy largo o no es texto")
    for campo in ("probabilidad", "impacto"):
        if not _es_entero(fila[campo]) or not 1 <= fila[campo] <= 5:
            problemas.append(f"fila {numero}: {campo} fuera de 1 a 5 o no es entero")
    if not _texto_valido(fila["mitigacion"]):
        problemas.append(f"fila {numero}: mitigacion vacia, muy larga o no es texto")
    evidencias = fila["evidencias"]
    if not isinstance(evidencias, list) or len(evidencias) > MAX_EVIDENCIAS:
        problemas.append(f"fila {numero}: evidencias no es una lista de hasta {MAX_EVIDENCIAS}")
    elif not all(isinstance(e, str) and len(e) <= MAX_TEXTO for e in evidencias):
        problemas.append(f"fila {numero}: una evidencia no es texto o es muy larga")
    if fila["supervision"] is not None and not _texto_valido(fila["supervision"]):
        problemas.append(f"fila {numero}: supervision debe ser null o texto no vacio")
    if not _texto_valido(fila["nota"], vacio_permitido=True):
        problemas.append(f"fila {numero}: nota muy larga o no es texto")
    return problemas


def problemas_de_matriz(filas) -> list[str]:
    """Problemas de forma de toda la matriz: lista no vacia, filas validas y riesgos sin repetir."""
    if not isinstance(filas, list) or not filas or len(filas) > MAX_FILAS:
        return [f"la matriz debe ser una lista de 1 a {MAX_FILAS} filas"]
    problemas = []
    vistos = {}
    for numero, fila in enumerate(filas, start=1):
        problemas.extend(problemas_de_fila(fila, numero))
        nombre = fila.get("riesgo") if isinstance(fila, dict) else None
        if isinstance(nombre, str) and nombre.strip():
            clave = " ".join(nombre.split()).casefold()
            if clave in vistos:
                problemas.append(f"fila {numero}: riesgo repetido (igual que la fila {vistos[clave]})")
            else:
                vistos[clave] = numero
    return problemas


def leer_matriz(ruta: str) -> list[dict]:
    """Lee y valida la forma de la matriz; el error nunca cita el contenido ni la ruta."""
    try:
        with open(ruta, encoding="utf-8") as archivo:
            filas = json.load(archivo)
    except OSError as error:
        raise ValueError(f"no se pudo leer la matriz ({type(error).__name__})") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"la matriz no es JSON valido (linea {error.lineno})") from None
    except UnicodeDecodeError:
        raise ValueError("la matriz no es UTF-8 valido") from None
    problemas = problemas_de_matriz(filas)
    if problemas:
        raise ValueError("; ".join(problemas))
    return filas


def _nombres_de_primer_nivel(arbol: ast.Module) -> set[str]:
    nombres = set()
    for nodo in arbol.body:
        if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nombres.add(nodo.name)
        elif isinstance(nodo, ast.Assign):
            nombres.update(t.id for t in nodo.targets if isinstance(t, ast.Name))
        elif isinstance(nodo, ast.AnnAssign) and isinstance(nodo.target, ast.Name):
            nombres.add(nodo.target.id)
    return nombres


def problema_de_evidencia(raiz: str, evidencia: str) -> str | None:
    """None si 'ruta.py::simbolo' existe como definicion de primer nivel dentro de raiz; si no, el motivo (sin la ruta)."""
    ruta, separador, simbolo = evidencia.partition("::")
    if not separador or not ruta or not simbolo.isidentifier() or not ruta.endswith(".py"):
        return "formato invalido (se espera ruta.py::simbolo)"
    if os.path.isabs(ruta) or ".." in ruta.replace("\\", "/").split("/"):
        return "la ruta debe ser relativa y no salir de la raiz"
    base = os.path.realpath(raiz)
    completa = os.path.realpath(os.path.join(base, ruta))
    if os.path.commonpath([base, completa]) != base:
        return "la ruta sale de la raiz"
    if not os.path.isfile(completa):
        return "el archivo no existe"
    try:
        with open(completa, encoding="utf-8") as archivo:
            arbol = ast.parse(archivo.read())
    except (OSError, SyntaxError, ValueError):
        return "el archivo no se pudo analizar"
    if simbolo not in _nombres_de_primer_nivel(arbol):
        return "el simbolo no existe en el archivo"
    return None


def problemas_de_evidencias(filas: list[dict], raiz: str) -> list[str]:
    """Una fila por evidencia que no se encuentra en el codigo; cita fila y posicion, no el contenido."""
    problemas = []
    for numero, fila in enumerate(filas, start=1):
        if not fila["evidencias"]:
            problemas.append(f"fila {numero}: sin evidencias")
        for posicion, evidencia in enumerate(fila["evidencias"], start=1):
            motivo = problema_de_evidencia(raiz, evidencia)
            if motivo:
                problemas.append(f"fila {numero}, evidencia {posicion}: {motivo}")
    return problemas


def problemas_de_supervision(filas: list[dict]) -> list[str]:
    """Una fila por riesgo ALTO o CRITICO sin punto de supervision; cita fila y categoria, nunca el contenido."""
    problemas = []
    for numero, fila in enumerate(filas, start=1):
        categoria = categoria_riesgo(puntaje_riesgo(fila["probabilidad"], fila["impacto"]))
        if categoria in ("CRITICO", "ALTO") and fila["supervision"] is None:
            problemas.append(f"fila {numero}: riesgo {categoria} sin punto de supervision")
    return problemas


def filas_calculadas(filas: list[dict]) -> list[dict]:
    """Copia de las filas con puntaje, categoria y al_limite, ordenadas de mayor a menor puntaje (estable)."""
    calculadas = []
    for fila in filas:
        puntaje = puntaje_riesgo(fila["probabilidad"], fila["impacto"])
        calculadas.append(
            {**fila, "puntaje": puntaje, "categoria": categoria_riesgo(puntaje), "al_limite": esta_en_el_limite(fila["probabilidad"], fila["impacto"])}
        )
    return sorted(calculadas, key=lambda f: -f["puntaje"])


def tabla(filas: list[dict]) -> list[str]:
    """Una linea por riesgo: categoria, puntaje, nombre y la marca [AL LIMITE] si corresponde."""
    lineas = []
    for fila in filas_calculadas(filas):
        marca = " [AL LIMITE]" if fila["al_limite"] else ""
        lineas.append(f"{fila['categoria']:8} ({fila['puntaje']:2}) - {fila['riesgo']}{marca}")
    return lineas
