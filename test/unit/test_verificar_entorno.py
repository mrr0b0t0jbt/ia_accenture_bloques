import os
import shutil
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent.parent
CANARIO_CLAUDE = "canario-claude-0001"
CANARIO_GEMINI = "canario-gemini-0002"


def _correr_en(carpeta_script, cwd, **extra_env):
    """Copia el script a carpeta_script y lo corre con cwd dado y un entorno mínimo sin claves (más las variables dadas)."""
    shutil.copy(RAIZ / "verificar_entorno.py", carpeta_script / "verificar_entorno.py")
    env = {clave: os.environ[clave] for clave in ("PATH", "SYSTEMROOT", "LANG") if clave in os.environ}
    env.update(extra_env)
    return subprocess.run([sys.executable, str(carpeta_script / "verificar_entorno.py")], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=60)


def _correr_con_version(tmp_path, version):
    """Corre verificar_entorno.py en una carpeta vacia haciendo que sys.version_info valga la version dada."""
    version_falsa = tuple(version) + ("final", 0)
    script = str(RAIZ / "verificar_entorno.py")
    corredor = tmp_path / "corredor.py"
    corredor.write_text(
        "import runpy, sys\n"
        f"sys.version_info = {version_falsa!r}\n"
        f"runpy.run_path({script!r}, run_name=\"__main__\")\n",
        encoding="utf-8",
    )
    env = {clave: os.environ[clave] for clave in ("PATH", "SYSTEMROOT", "LANG") if clave in os.environ}
    return subprocess.run([sys.executable, str(corredor)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)


def _correr(tmp_path, **extra_env):
    """Copia el script a tmp_path y lo corre ahí con un entorno mínimo sin claves (más las variables dadas)."""
    return _correr_en(tmp_path, tmp_path, **extra_env)


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


def _escribir_env_con_canarios(tmp_path):
    (tmp_path / ".env").write_text(f"CLAUDE_API_KEY={CANARIO_CLAUDE}\nGEMINI_API_KEY={CANARIO_GEMINI}\n", encoding="utf-8")


def test_variable_vacia_en_el_entorno_no_se_pisa_con_el_env(tmp_path):
    # override=False: una variable ya definida, aunque vacía, gana sobre el .env (con override=True diría "definida")
    _escribir_env_con_canarios(tmp_path)
    proceso = _correr(tmp_path, CLAUDE_API_KEY="")
    salida = proceso.stdout + proceso.stderr
    assert "CLAUDE_API_KEY: FALTA" in proceso.stdout, salida
    assert CANARIO_CLAUDE not in salida
    assert CANARIO_GEMINI not in salida


def test_variable_del_entorno_gana_sobre_el_env_sin_mostrar_valores(tmp_path):
    canario_entorno = "canario-entorno-0006"
    _escribir_env_con_canarios(tmp_path)
    proceso = _correr(tmp_path, CLAUDE_API_KEY=canario_entorno)
    salida = proceso.stdout + proceso.stderr
    assert "CLAUDE_API_KEY: definida" in proceso.stdout, salida
    assert canario_entorno not in salida
    assert CANARIO_CLAUDE not in salida
    assert CANARIO_GEMINI not in salida


def test_no_lee_el_env_de_un_directorio_padre(tmp_path):
    (tmp_path / ".env").write_text(f"CLAUDE_API_KEY={CANARIO_CLAUDE}\n", encoding="utf-8")
    hijo = tmp_path / "hijo"
    hijo.mkdir()
    proceso = _correr(hijo)
    assert proceso.returncode == 1
    assert "CLAUDE_API_KEY: FALTA" in proceso.stdout


def test_el_env_se_lee_del_cwd_y_no_de_la_carpeta_del_script(tmp_path):
    canario_padre = "canario-padre-0004"
    canario_cwd = "canario-cwd-0005"
    padre = tmp_path / "padre"
    carpeta_script = padre / "scripts_copia"
    carpeta_script.mkdir(parents=True)
    (padre / ".env").write_text(f"GEMINI_API_KEY={canario_padre}\n", encoding="utf-8")
    trabajo = tmp_path / "trabajo"
    (trabajo / "vectorstore").mkdir(parents=True)
    (trabajo / "vectorstore" / "chroma.sqlite3").write_bytes(b"x")
    (trabajo / ".env").write_text(f"CLAUDE_API_KEY={canario_cwd}\n", encoding="utf-8")
    proceso = _correr_en(carpeta_script, trabajo)
    salida = proceso.stdout + proceso.stderr
    assert proceso.returncode == 1, salida
    assert "CLAUDE_API_KEY: definida" in proceso.stdout
    assert "GEMINI_API_KEY: FALTA" in proceso.stdout
    assert "Pendiente:" in proceso.stdout
    assert "GEMINI_API_KEY" in proceso.stdout.split("Pendiente:")[1]
    assert canario_padre not in salida
    assert canario_cwd not in salida


def test_python_3_11_no_cumple_la_version_minima(tmp_path):
    proceso = _correr_con_version(tmp_path, (3, 11, 9))
    assert proceso.returncode == 1
    assert "Python 3.11.9: se requiere 3.12 o más" in proceso.stdout
    assert "Python 3.12+" in proceso.stdout.split("Pendiente:")[-1]


def test_python_3_12_es_la_version_minima_aceptada(tmp_path):
    proceso = _correr_con_version(tmp_path, (3, 12, 0))
    assert "Python 3.12.0: OK" in proceso.stdout
    assert "se requiere" not in proceso.stdout


def test_python_3_14_cumple_la_version_minima(tmp_path):
    proceso = _correr_con_version(tmp_path, (3, 14, 9))
    assert "Python 3.14.9: OK" in proceso.stdout
    assert "se requiere" not in proceso.stdout
