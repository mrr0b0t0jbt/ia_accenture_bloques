import os

from conftest import RUTA_INDICE_REAL, RUTA_TRAZAS_REALES, estado_de_las_trazas_reales, estado_del_indice_real


def test_el_guardian_detecta_crear_crecer_y_modificar_las_trazas_y_no_se_queja_si_no_se_tocan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inicial = estado_de_las_trazas_reales()
    assert inicial is None and estado_de_las_trazas_reales() == inicial
    os.makedirs(os.path.dirname(RUTA_TRAZAS_REALES))
    with open(RUTA_TRAZAS_REALES, "w") as archivo:
        archivo.write("{}\n")
    creado = estado_de_las_trazas_reales()
    assert creado != inicial and estado_de_las_trazas_reales() == creado
    with open(RUTA_TRAZAS_REALES, "a") as archivo:
        archivo.write("{}\n")
    crecido = estado_de_las_trazas_reales()
    assert crecido != creado
    # mismo tamaño, otra fecha: una modificación que no cambia el tamaño
    os.utime(RUTA_TRAZAS_REALES, ns=(1, crecido[1] + 1_000_000_000))
    assert estado_de_las_trazas_reales() != crecido


def test_el_guardian_del_indice_detecta_que_aparece_vectorstore_y_no_se_queja_si_no_se_toca(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    inicial = estado_del_indice_real()
    assert inicial is None and estado_del_indice_real() == inicial
    os.makedirs(RUTA_INDICE_REAL)
    creado = estado_del_indice_real()
    assert creado != inicial and estado_del_indice_real() == creado
    # mismo directorio, otra fecha: un cambio que no lo crea
    os.utime(RUTA_INDICE_REAL, ns=(1, creado + 1_000_000_000))
    assert estado_del_indice_real() != creado
