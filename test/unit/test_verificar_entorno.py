import os
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
CANARIO_CLAUDE = "canario-claude-0001"
CANARIO_GEMINI = "canario-gemini-0002"


def _correr(tmp_path, **extra_env):
    """Copia el script a tmp_path y lo corre ahí con un entorno mínimo sin claves (más las variables dadas)."""
    shutil.copy(RAIZ / "verificar_entorno.py", tmp_path / "verificar_entorno.py")
    env = {clave: os.environ[clave] for clave in ("PATH", "SYSTEMROOT", "LANG") if clave in os.environ}
    env.update(extra_env)
    return subprocess.run([sys.executable, "verificar_entorno.py"], cwd=tmp_path, env=env,
                          capture_output=True, text=True, timeout=60)


def test_sin_claves_ni_indice_sale_con_1(tmp_path):
    proceso = _correr(tmp_path)
    assert proceso.returncode == 1
    assert "CLAUDE_API_KEY: FALTA" in proceso.stdout
    assert "GEMINI_API_KEY: FALTA" in proceso.stdout
    assert "Pendiente:" in proceso.stdout


def test_indice_vacio_cuenta_como_faltante(tmp_path):
    (tmp_path / "vectorstore").mkdir()
    proceso = _correr(tmp_path, CLAUDE_API_KEY=CANARIO_CLAUDE, GEMINI_API_KEY=CANARIO_GEMINI)
    assert proceso.returncode == 1
    assert "vectorstore/" in proceso.stdout.split("Pendiente:")[1]


def test_con_claves_e_indice_sale_con_0_sin_mostrar_valores(tmp_path):
    (tmp_path / "vectorstore").mkdir()
    (tmp_path / "vectorstore" / "chroma.sqlite3").write_bytes(b"x")
    proceso = _correr(tmp_path, CLAUDE_API_KEY=CANARIO_CLAUDE, GEMINI_API_KEY=CANARIO_GEMINI)
    salida = proceso.stdout + proceso.stderr
    assert proceso.returncode == 0, salida
    assert "Entorno listo para la corrida final" in proceso.stdout
    assert "CLAUDE_API_KEY: definida" in proceso.stdout
    assert CANARIO_CLAUDE not in salida
    assert CANARIO_GEMINI not in salida


def test_lee_el_env_del_directorio_actual_sin_mostrar_valores(tmp_path):
    (tmp_path / "vectorstore").mkdir()
    (tmp_path / "vectorstore" / "chroma.sqlite3").write_bytes(b"x")
    (tmp_path / ".env").write_text(f"CLAUDE_API_KEY={CANARIO_CLAUDE}\nGEMINI_API_KEY={CANARIO_GEMINI}\n", encoding="utf-8")
    proceso = _correr(tmp_path)
    salida = proceso.stdout + proceso.stderr
    assert proceso.returncode == 0, salida
    assert CANARIO_CLAUDE not in salida
    assert CANARIO_GEMINI not in salida


def test_no_lee_el_env_de_un_directorio_padre(tmp_path):
    (tmp_path / ".env").write_text(f"CLAUDE_API_KEY={CANARIO_CLAUDE}\n", encoding="utf-8")
    hijo = tmp_path / "hijo"
    hijo.mkdir()
    proceso = _correr(hijo)
    assert proceso.returncode == 1
    assert "CLAUDE_API_KEY: FALTA" in proceso.stdout
