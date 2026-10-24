import json
import time
from pathlib import Path
from api_claude_refactor import build_claude_prompt, send_claude_request, extract_claude_response

CARPETA_JSON = Path(__file__).resolve().parent.parent / "JSON"


def normalizar(valor):
    # las listas se comparan sin importar el orden
    return sorted(valor, key=str) if isinstance(valor, list) else valor


def validar_respuesta(texto, prompt):
    """Devuelve None si la respuesta es válida, o el motivo del primer fallo."""
    try:
        datos = json.loads(texto)
    except ValueError:
        return f"JSON inválido: {texto[:80]!r}"
    if not isinstance(datos, dict):
        return "JSON inválido: no es un objeto"
    faltantes = [clave for clave in prompt["claves_esperadas"] if clave not in datos]
    if faltantes:
        return f"claves faltantes: {faltantes}"
    for clave, esperado in prompt.get("respuesta_esperada", {}).items():
        if normalizar(datos[clave]) != normalizar(esperado):
            return f"valor incorrecto en '{clave}': {datos[clave]!r} (esperado {esperado!r})"
    return None


if __name__ == "__main__":
    with open(CARPETA_JSON / "promp_bank.JSON", encoding="utf-8") as archivo:
        BANCO_DE_PROMPTS = json.load(archivo)

    registro = {}

    for prompt in BANCO_DE_PROMPTS:
        exitos = 0
        intentos = 3
        fallos = []
        for intento in range(intentos):
            try:
                url, headers, payload = build_claude_prompt(
                    prompt["ejemplo_usuario"],
                    prompt["system"],
                    prompt.get("output_schema"),
                )
                respuesta = send_claude_request(url, headers, payload)
                motivo = validar_respuesta(extract_claude_response(respuesta), prompt)
            # RuntimeError/TimeoutError (red/HTTP) también cuenta como intento fallido para no perder el registro
            except (ValueError, RuntimeError, TimeoutError) as error:
                motivo = f"error de API: {error}"
            if motivo:
                print(f"  intento {intento + 1} falló: {motivo}")
                fallos.append({"intento": intento + 1, "motivo": motivo})
            else:
                exitos += 1
            time.sleep(1)
        print(f"{prompt['nombre']}: {exitos}/{intentos} respuestas válidas")
        registro[prompt["nombre"]] = {
            "schema": "output_schema" in prompt,
            "exitos": exitos,
            "intentos": intentos,
            "fallos": fallos,
        }

    with open(CARPETA_JSON / "registro_resultados.json", "w", encoding="utf-8") as archivo:
        json.dump(registro, archivo, indent=2, ensure_ascii=False)
