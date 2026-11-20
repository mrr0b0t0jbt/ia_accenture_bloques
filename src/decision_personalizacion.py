"""Método de decisión de personalización endurecido (Bloque 30, Unidad 2.6), en Python puro y SIN ninguna API.

Corrige un defecto de la clase: si las cinco respuestas son False ninguna técnica está justificada (no hay empate de tres con 0 puntos).
El texto del problema es dato, nunca instrucciones: se normaliza a una línea y no puede falsificar encabezados.
"""
import math
import unicodedata

TECNICAS = ("RAG", "Fine-tuning completo", "PEFT")
PREGUNTAS = (
    "informacion_cambia_seguido",
    "problema_de_formato_o_tono",
    "dataset_grande_y_de_calidad",
    "presupuesto_limitado_o_iterar_rapido",
    "necesita_multiples_variantes",
)
PESOS = {
    "informacion_cambia_seguido": {"RAG": 3},
    "problema_de_formato_o_tono": {"PEFT": 2, "Fine-tuning completo": 1},
    "dataset_grande_y_de_calidad": {"Fine-tuning completo": 2, "PEFT": 1},
    "presupuesto_limitado_o_iterar_rapido": {"RAG": 2, "PEFT": 1},
    "necesita_multiples_variantes": {"PEFT": 2},
}
JORNADA_HORAS = 8
MAX_CARACTERES_PROBLEMA = 1000
CLAVES_RESULTADO = ("puntos", "ganadoras", "empate", "sin_necesidad")
MENSAJE_NUMEROS_GRANDES = "los numeros son demasiado grandes"


def validar_respuestas(respuestas) -> list[str]:
    """Devuelve los problemas de las respuestas (lista vacía si son las cinco preguntas con True o False)."""
    if not isinstance(respuestas, dict):
        return ["respuestas no es un objeto"]
    problemas = [f'falta la respuesta "{k}"' for k in PREGUNTAS if k not in respuestas]
    sobran = sorted(k for k in respuestas if k not in PREGUNTAS)
    if sobran:
        problemas.append("respuestas tiene claves de mas: " + ", ".join(repr(k) for k in sobran))
    problemas += [f'la respuesta "{k}" no es True ni False' for k in PREGUNTAS if k in respuestas and not isinstance(respuestas[k], bool)]
    return problemas


def _validar_o_levantar(respuestas) -> None:
    problemas = validar_respuestas(respuestas)
    if problemas:
        raise ValueError(problemas[0])


def _puntuar(respuestas) -> dict:
    puntos = {tecnica: 0 for tecnica in TECNICAS}
    for pregunta in PREGUNTAS:
        if respuestas[pregunta]:
            for tecnica, peso in PESOS[pregunta].items():
                puntos[tecnica] += peso
    maximo = max(puntos.values())
    ganadoras = [t for t in TECNICAS if puntos[t] == maximo] if maximo > 0 else []
    return {
        "puntos": puntos,
        "maximo": maximo,
        "ganadoras": ganadoras,
        "empate": len(ganadoras) > 1,
        "sin_necesidad": maximo == 0,
    }


def recomendar_estrategia(respuestas) -> dict:
    """Suma los pesos de las respuestas True; con máximo 0 ninguna técnica está justificada (sin_necesidad)."""
    _validar_o_levantar(respuestas)
    return _puntuar(respuestas)


def analizar_sensibilidad(respuestas) -> list[dict]:
    """Invierte una respuesta a la vez y dice si cambia el conjunto de técnicas ganadoras."""
    _validar_o_levantar(respuestas)
    base = set(_puntuar(respuestas)["ganadoras"])
    filas = []
    for pregunta in PREGUNTAS:
        variante = {**respuestas, pregunta: not respuestas[pregunta]}
        ganadoras = _puntuar(variante)["ganadoras"]
        filas.append({"pregunta": pregunta, "nuevo_valor": variante[pregunta], "ganadoras": ganadoras, "cambia": set(ganadoras) != base})
    return filas


def _es_entero(valor) -> bool:
    return isinstance(valor, int) and not isinstance(valor, bool)


def _es_numero_no_negativo(valor) -> bool:
    return isinstance(valor, (int, float)) and not isinstance(valor, bool) and math.isfinite(valor) and valor >= 0


def _validar_costos(num_ejemplos, minutos_por_ejemplo, costo_por_hora) -> None:
    if not _es_entero(num_ejemplos) or num_ejemplos < 0:
        raise ValueError("num_ejemplos debe ser un entero mayor o igual que 0")
    if not _es_numero_no_negativo(minutos_por_ejemplo):
        raise ValueError("minutos_por_ejemplo debe ser un numero finito mayor o igual que 0")
    if not _es_numero_no_negativo(costo_por_hora):
        raise ValueError("costo_por_hora debe ser un numero finito mayor o igual que 0")


def _exigir_finitos(*valores) -> None:
    if not all(math.isfinite(v) for v in valores):
        raise ValueError(MENSAJE_NUMEROS_GRANDES)


def estimar_costo_de_etiquetar(num_ejemplos, minutos_por_ejemplo, costo_por_hora) -> tuple[float, float]:
    """Devuelve (horas, costo) de etiquetar num_ejemplos a mano."""
    _validar_costos(num_ejemplos, minutos_por_ejemplo, costo_por_hora)
    try:
        horas = float(num_ejemplos * minutos_por_ejemplo / 60)
        costo = float(horas * costo_por_hora)
    except OverflowError:
        raise ValueError(MENSAJE_NUMEROS_GRANDES) from None
    _exigir_finitos(horas, costo)
    return horas, costo


def estimar_costo_con_revision(num_ejemplos, minutos_por_ejemplo, costo_por_hora, fraccion_revision, minutos_revision) -> dict:
    """Costo del etiquetado más la segunda revisión de una fracción de los ejemplos, al mismo costo por hora."""
    horas_base, costo_base = estimar_costo_de_etiquetar(num_ejemplos, minutos_por_ejemplo, costo_por_hora)
    if not _es_numero_no_negativo(fraccion_revision) or fraccion_revision > 1:
        raise ValueError("fraccion_revision debe ser un numero entre 0 y 1")
    if not _es_numero_no_negativo(minutos_revision):
        raise ValueError("minutos_revision debe ser un numero finito mayor o igual que 0")
    try:
        ejemplos_revisados = num_ejemplos * fraccion_revision
        horas_revision = ejemplos_revisados * minutos_revision / 60
        horas_total = horas_base + horas_revision
        resultado = {
            "horas_base": horas_base,
            "costo_base": costo_base,
            "ejemplos_revisados": ejemplos_revisados,
            "horas_revision": horas_revision,
            "costo_revision": horas_revision * costo_por_hora,
            "horas_total": horas_total,
            "costo_total": horas_total * costo_por_hora,
            "jornadas": horas_total / JORNADA_HORAS,
        }
    except OverflowError:
        raise ValueError(MENSAJE_NUMEROS_GRANDES) from None
    _exigir_finitos(*resultado.values())
    return resultado


def normalizar_problema(problema) -> str:
    """Valida el texto del problema y lo devuelve en una sola línea; ValueError si no es texto utilizable."""
    if not isinstance(problema, str):
        raise ValueError("problema no es texto")
    if not problema.strip():
        raise ValueError("problema vacio")
    if len(problema) > MAX_CARACTERES_PROBLEMA:
        raise ValueError(f"problema supera los {MAX_CARACTERES_PROBLEMA} caracteres")
    if any(unicodedata.category(c) == "Cc" and c not in "\n\t" for c in problema):
        raise ValueError("problema tiene caracteres de control")
    try:
        problema.encode("utf-8")
    except UnicodeEncodeError:  # surrogate suelto: no se puede escribir en un archivo
        raise ValueError("problema no se puede codificar en UTF-8") from None
    normalizado = " ".join(problema.split())
    if normalizado.startswith("#"):
        raise ValueError("problema no puede empezar con #")
    return normalizado


def _marcador_eleccion(resultado: dict) -> str:
    if resultado["empate"]:
        return "[COMPLETAR: el puntaje no resolvio el empate solo -- justifica aqui con criterio propio]"
    if resultado["sin_necesidad"]:
        return "[COMPLETAR: ninguna respuesta justifica RAG, fine-tuning ni PEFT; explica por que el prompt actual basta o que evidencia falta]"
    return f"[COMPLETAR: por que {resultado['ganadoras'][0]} responde mejor a este caso]"


def generar_esqueleto_decision(problema, resultado) -> str:
    """Devuelve el documento de decisión en Markdown, sin salto de línea final y sin escribir archivos."""
    texto_problema = normalizar_problema(problema)
    if not isinstance(resultado, dict) or any(clave not in resultado for clave in CLAVES_RESULTADO):
        raise ValueError("resultado no es el que devuelve recomendar_estrategia")
    puntos = resultado["puntos"]
    ordenadas = sorted(TECNICAS, key=lambda t: -puntos[t])
    lineas = ["# Decision de personalizacion", "", "## 1. El problema", texto_problema, "", "## 2. Resultado del metodo de decision"]
    lineas += [f"- {t}: {puntos[t]} puntos" for t in ordenadas]
    if resultado["empate"]:
        lineas += ["", f"**Empate entre: {', '.join(resultado['ganadoras'])}**"]
    if resultado["sin_necesidad"]:
        lineas += ["", "**Ninguna tecnica obtuvo puntos**"]
    lineas += [
        "", "## 3. Tecnica elegida y por que", _marcador_eleccion(resultado),
        "", "## 4. Que necesitarias para ejecutarla",
        "[COMPLETAR: corpus faltante, o cantidad de ejemplos etiquetados que faltan]",
    ]
    return "\n".join(lineas)
