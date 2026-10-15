import os
from api_gemini import get_gemini_response

#Llamar a la función get_gemini_response desde gemini.py

response = get_gemini_response("Confirma en una frase corta que esta conexion funciona correctamente.")
print(response)