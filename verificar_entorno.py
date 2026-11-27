"""Verifica el entorno antes de la corrida final del Gate 2: Python, paquetes, claves e índice.

Las claves se informan solo como "definida" o "FALTA"; su valor nunca se imprime. El .env se lee SOLO del
directorio actual (Path.cwd() / ".env"), nunca de un directorio padre. No llama a ninguna API.

Cómo correrlo (desde la raíz del repositorio):

    python verificar_entorno.py
"""
import os
import sys
from importlib import metadata
from pathlib import Path

VERSION_MINIMA = (3, 12)
PAQUETES = (
    "langchain-core",
    "langchain-chroma",
    "chromadb",
    "langgraph",
    "langgraph-checkpoint-sqlite",
    "python-dotenv",
    "langchain-anthropic",
    "langchain-google-genai",
)
VARIABLES = ("CLAUDE_API_KEY", "GEMINI_API_KEY")
DIRECTORIO_INDICE = "vectorstore"


def _cargar_env_local(pendientes: list[str]) -> None:
    """Carga el .env del directorio actual si existe, sin pisar variables ya definidas ni buscar hacia arriba."""
    ruta = Path.cwd() / ".env"
    if not ruta.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        pendientes.append("python-dotenv para leer el .env")
        return
    load_dotenv(dotenv_path=ruta, override=False)


def main() -> int:
    pendientes: list[str] = []

    version = ".".join(str(parte) for parte in sys.version_info[:3])
    if sys.version_info[:2] >= VERSION_MINIMA:
        print(f"Python {version}: OK")
    else:
        print(f"Python {version}: se requiere {VERSION_MINIMA[0]}.{VERSION_MINIMA[1]} o más")
        pendientes.append(f"Python {VERSION_MINIMA[0]}.{VERSION_MINIMA[1]}+")

    for paquete in PAQUETES:
        try:
            print(f"{paquete}: {metadata.version(paquete)}")
        except metadata.PackageNotFoundError:
            print(f"{paquete}: FALTA")
            pendientes.append(paquete)

    _cargar_env_local(pendientes)
    for variable in VARIABLES:
        if os.environ.get(variable, "").strip():
            print(f"{variable}: definida")
        else:
            print(f"{variable}: FALTA")
            pendientes.append(variable)

    indice = Path.cwd() / DIRECTORIO_INDICE
    if indice.is_dir() and any(indice.iterdir()):
        print(f"Índice {DIRECTORIO_INDICE}/: OK")
    else:
        print(f"Índice {DIRECTORIO_INDICE}/: FALTA o está vacío")
        pendientes.append(f"índice en {DIRECTORIO_INDICE}/")

    if pendientes:
        print("Pendiente: " + ", ".join(pendientes))
        return 1
    print("Entorno listo para la corrida final")
    return 0


if __name__ == "__main__":
    sys.exit(main())
