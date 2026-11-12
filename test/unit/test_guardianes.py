import shutil
import subprocess
import sys
from pathlib import Path

CONFTEST = Path(__file__).resolve().parent / "conftest.py"

PRUEBAS_AJENAS = '''
import os


def test_escribe_el_indice():
    os.makedirs("vectorstore")


def test_escribe_las_trazas():
    os.makedirs("trazas")
    with open("trazas/trazas_agente.jsonl", "w") as archivo:
        archivo.write("{}\\n")


def test_inocua():
    assert True
'''


def test_los_dos_guardianes_autouse_fallan_cuando_una_prueba_escribe_donde_no_debe(tmp_path):
    shutil.copy(CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "test_ajenas.py").write_text(PRUEBAS_AJENAS, encoding="utf-8")
    resultado = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-c", "pytest.ini", f"--rootdir={tmp_path}", "-rA", "test_ajenas.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    salida = resultado.stdout
    assert resultado.returncode != 0, salida
    assert "ERROR test_ajenas.py::test_escribe_el_indice" in salida
    assert "ERROR test_ajenas.py::test_escribe_las_trazas" in salida
    assert "La prueba escribió en vectorstore/" in salida
    assert "La prueba escribió en trazas/trazas_agente.jsonl" in salida
    assert "PASSED test_ajenas.py::test_inocua" in salida
    assert "ERROR test_ajenas.py::test_inocua" not in salida
