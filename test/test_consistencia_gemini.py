import json
import time
from api_gemini import build_gemini_prompt, send_gemini_request, extract_gemini_response
from test_consistencia import CARPETA_JSON, validar_respuesta


if __name__ == "__main__":
    with open(CARPETA_JSON / "promp_bank_gemini.JSON", encoding="utf-8") as archivo:
        BANCO_DE_PROMPTS = json.load(archivo)

    registro = {}

    for prompt in BANCO_DE_PROMPTS:
        exitos = 0
        intentos = 3
        fallos = []
        for intento in range(intentos):
            try:
                url, headers, payload = build_gemini_prompt(
                    prompt["ejemplo_usuario"],
                    prompt["system"],
                    prompt.get("output_schema"),
                )
                respuesta = send_gemini_request(url, headers, payload)
                motivo = validar_respuesta(extract_gemini_response(respuesta), prompt)
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

    with open(CARPETA_JSON / "registro_resultados_gemini.json", "w", encoding="utf-8") as archivo:
        json.dump(registro, archivo, indent=2, ensure_ascii=False)
