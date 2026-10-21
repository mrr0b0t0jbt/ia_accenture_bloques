import os
from api_claude_refactor import get_claude_response

#Llamar a la función get_claude_response desde claude.py
prompt = "Confirma en una frase corta que esta conexion funciona correctamente."
try:
    response = get_claude_response(prompt)
    print(response)
except (ValueError, RuntimeError) as ve:
    print(f"Error: {ve}")