import os
from api_gemini_refactor import get_gemini_response

#Llamar a la función get_gemini_response desde gemini.py
prompt = "Confirma en una frase corta que esta conexion funciona correctamente."
try:
    response = get_gemini_response(prompt)
    print(response)
except (ValueError, RuntimeError) as ve:
    print(f"Error: {ve}")