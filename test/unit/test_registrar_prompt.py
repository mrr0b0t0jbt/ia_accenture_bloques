import importlib.util
import json
import os
import socket
import sys
from pathlib import Path

import pytest

import defensa_inyeccion
import recuperacion
import registro_prompts
from registro_estructurado import huella

RAIZ = Path(__file__).resolve().parent.parent.parent
_spec = importlib.util.spec_from_file_location("registrar_prompt", RAIZ / "scripts" / "registrar_prompt.py")
registrar_prompt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(registrar_prompt)

TEXTO_ACTUAL = defensa_inyeccion.PROMPT_SISTEMA + "\n" + defensa_inyeccion.PLANTILLA_USUARIO


@pytest.fixture
def ruta(tmp_path):
    return str(tmp_path / "registro.jsonl")


def _registrar(ruta, origen="actual", nota="nota de prueba", *extra):
    return registrar_prompt.main(["--registro", ruta, "--registrar", "--origen", origen, "--autor", "tester", "--nota", nota, *extra])


def test_l2_w_el_texto_actual_del_codigo_tiene_la_huella_que_ya_usan_los_resultados():
    assert registrar_prompt.textos_del_codigo() == {"actual": TEXTO_ACTUAL, "plantilla_un_mensaje": recuperacion.PROMPT_TEMPLATE}
    assert huella(TEXTO_ACTUAL) == defensa_inyeccion.HUELLA_PROMPT


def test_l2_x_la_ruta_por_defecto_es_docs_registro_prompts_de_la_raiz_del_proyecto():
    argumentos = registrar_prompt._argumentos(["--listar"])
    assert argumentos.registro == str(RAIZ / "docs" / "registro_prompts.jsonl")


@pytest.mark.parametrize(
    "argv",
    [[], ["--listar", "--verificar"], ["--registrar"], ["--registrar", "--origen", "actual", "--autor", "a"], ["--registrar", "--origen", "otro", "--autor", "a", "--nota", "n"]],
    ids=["sin_accion", "dos_acciones", "registrar_sin_datos", "registrar_sin_nota", "origen_desconocido"],
)
def test_l2_y_los_usos_incorrectos_salen_con_2_sin_tocar_nada(ruta, argv):
    with pytest.raises(SystemExit) as salida:
        registrar_prompt.main(["--registro", ruta, *argv])
    assert salida.value.code == 2 and not os.path.exists(ruta)


def test_l2_z_registrar_agrega_la_version_con_la_huella_del_codigo_y_una_repeticion_sale_con_1(ruta, capsys):
    assert _registrar(ruta, "actual", "primera", "--modelo", "claude-sonnet-5-5", "--parametros-json", '{"max_tokens": 1024}') == 0
    (entrada,) = registro_prompts.leer_registro(ruta)
    assert entrada["huella"] == defensa_inyeccion.HUELLA_PROMPT and entrada["version"] == 1 and entrada["nombre"] == "prompt_rag"
    assert entrada["modelo"] == "claude-sonnet-5-5" and entrada["parametros"] == {"max_tokens": 1024} and entrada["autor"] == "tester"
    assert f"Registrada prompt_rag v1 ({defensa_inyeccion.HUELLA_PROMPT})." in capsys.readouterr().out
    antes = Path(ruta).read_bytes()
    assert _registrar(ruta, "actual", "otra vez") == 1
    assert "No se registro: sin cambios" in capsys.readouterr().err and Path(ruta).read_bytes() == antes


def test_l2_aa_las_dos_versiones_del_historial_se_registran_en_orden_y_la_verificacion_da_ok(ruta, capsys):
    assert _registrar(ruta, "plantilla_un_mensaje", "un solo mensaje") == 0
    assert _registrar(ruta, "actual", "sistema y humano separados") == 0
    entradas = registro_prompts.leer_registro(ruta)
    assert [(e["version"], e["huella"]) for e in entradas] == [(1, huella(recuperacion.PROMPT_TEMPLATE)), (2, defensa_inyeccion.HUELLA_PROMPT)]
    capsys.readouterr()
    assert registrar_prompt.main(["--registro", ruta, "--verificar"]) == 0
    assert f"OK: el prompt del codigo es la version 2 ({defensa_inyeccion.HUELLA_PROMPT})." in capsys.readouterr().out


def test_l2_ab_parametros_json_que_no_es_json_sale_con_1_sin_escribir(ruta, capsys):
    assert _registrar(ruta, "actual", "n", "--parametros-json", "{no") == 1
    assert "--parametros-json no es JSON valido." in capsys.readouterr().err and not os.path.exists(ruta)


def test_l2_ac_verificar_sin_registro_sale_con_1(ruta, capsys):
    assert registrar_prompt.main(["--registro", ruta, "--verificar"]) == 1
    assert "no tiene ninguna version registrada" in capsys.readouterr().err


def test_l2_ad_verificar_detecta_un_cambio_silencioso_del_prompt(ruta, capsys, monkeypatch):
    assert _registrar(ruta, "actual", "n") == 0
    capsys.readouterr()
    monkeypatch.setattr(defensa_inyeccion, "PROMPT_SISTEMA", defensa_inyeccion.PROMPT_SISTEMA + " Cambio silencioso.")
    assert registrar_prompt.main(["--registro", ruta, "--verificar"]) == 1
    error = capsys.readouterr().err
    assert defensa_inyeccion.HUELLA_PROMPT in error and huella(defensa_inyeccion.PROMPT_SISTEMA + "\n" + defensa_inyeccion.PLANTILLA_USUARIO) in error
    assert "no es el ultimo registrado" in error


def test_l2_ae_verificar_con_un_registro_alterado_sale_con_1_y_el_numero_de_linea(ruta, capsys):
    assert _registrar(ruta, "actual", "n") == 0
    lineas = Path(ruta).read_text(encoding="utf-8").split("\n")
    entrada = json.loads(lineas[0])
    entrada["texto"] = entrada["texto"] + " editado a mano"
    Path(ruta).write_text(json.dumps(entrada) + "\n", encoding="utf-8")
    capsys.readouterr()
    assert registrar_prompt.main(["--registro", ruta, "--verificar"]) == 1
    assert "Registro invalido: linea 1: la huella no coincide con el texto" in capsys.readouterr().err


def test_l2_af_verificar_con_la_cadena_rota_sale_con_1_aunque_la_ultima_huella_coincida(ruta, capsys):
    assert _registrar(ruta, "plantilla_un_mensaje", "a") == 0
    assert _registrar(ruta, "actual", "b") == 0
    lineas = Path(ruta).read_text(encoding="utf-8").split("\n")[:-1]
    segunda = json.loads(lineas[1])
    segunda["huella_anterior"] = "000000000000"
    Path(ruta).write_text(lineas[0] + "\n" + json.dumps(segunda) + "\n", encoding="utf-8")
    capsys.readouterr()
    assert registrar_prompt.main(["--registro", ruta, "--verificar"]) == 1
    assert "Problema en el registro: linea 2: huella_anterior no encadena con la version previa" in capsys.readouterr().err


def test_l2_ag_listar_muestra_una_linea_por_version_y_nunca_el_texto_del_prompt(ruta, capsys):
    assert _registrar(ruta, "plantilla_un_mensaje", "un mensaje") == 0
    assert _registrar(ruta, "actual", "separados") == 0
    capsys.readouterr()
    assert registrar_prompt.main(["--registro", ruta, "--listar"]) == 0
    salida = capsys.readouterr().out.splitlines()
    assert len(salida) == 2 and salida[0].startswith(f"prompt_rag v1 {huella(recuperacion.PROMPT_TEMPLATE)} ") and salida[1].endswith(" - separados")
    assert "Responde la pregunta" not in "\n".join(salida) and "Eres un asistente" not in "\n".join(salida)


def test_l2_ah_listar_un_registro_que_no_existe_no_imprime_nada_y_sale_con_0(ruta, capsys):
    assert registrar_prompt.main(["--registro", ruta, "--listar"]) == 0
    assert capsys.readouterr().out == "" and not os.path.exists(ruta)


def test_l2_ai_diff_entre_las_dos_ultimas_versiones_y_error_con_menos_de_dos(ruta, capsys):
    assert registrar_prompt.main(["--registro", ruta, "--diff"]) == 1
    assert "menos de dos versiones" in capsys.readouterr().err
    assert _registrar(ruta, "plantilla_un_mensaje", "a") == 0
    capsys.readouterr()
    assert registrar_prompt.main(["--registro", ruta, "--diff"]) == 1
    assert "menos de dos versiones" in capsys.readouterr().err
    assert _registrar(ruta, "actual", "b") == 0
    capsys.readouterr()
    assert registrar_prompt.main(["--registro", ruta, "--diff"]) == 0
    esperado = registro_prompts.diff_prompts(recuperacion.PROMPT_TEMPLATE, TEXTO_ACTUAL)
    assert capsys.readouterr().out.splitlines() == "\n".join(esperado).splitlines() and esperado[:2] == ["--- version_anterior", "+++ version_nueva"]


def test_l2_aj_un_registro_en_un_directorio_que_no_existe_sale_con_1_sin_crear_nada(tmp_path, capsys):
    ruta = str(tmp_path / "no_existe" / "r.jsonl")
    assert _registrar(ruta) == 1
    assert "No se registro: el directorio del registro no existe" in capsys.readouterr().err and not (tmp_path / "no_existe").exists()


def test_l2_ak_el_script_no_abre_red_ni_importa_el_cliente_de_la_api(ruta, monkeypatch):
    def sin_red(*args, **kwargs):
        raise AssertionError("el script abrio red")

    monkeypatch.setattr(socket.socket, "connect", sin_red)
    monkeypatch.setitem(sys.modules, "api_claude_refactor", None)
    monkeypatch.setitem(sys.modules, "embeddings_gemini", None)
    assert _registrar(ruta) == 0 and registrar_prompt.main(["--registro", ruta, "--verificar"]) == 0


def test_l2_al_el_prompt_del_codigo_coincide_con_el_registro_commiteado():
    ruta = RAIZ / registro_prompts.RUTA_REGISTRO_POR_DEFECTO
    if not ruta.exists():
        pytest.skip("docs/registro_prompts.jsonl aun no esta en el repo: se siembra con scripts/registrar_prompt.py y lo commitea el usuario")
    entradas = registro_prompts.leer_registro(str(ruta))
    assert registro_prompts.problemas_del_registro(entradas) == []
    assert registro_prompts.comparar_con_codigo(entradas, registrar_prompt.NOMBRE_PROMPT, TEXTO_ACTUAL)["estado"] == "igual"


def test_l2b_a_un_registro_que_es_una_carpeta_sale_con_1_y_sin_traceback(tmp_path, capsys):
    carpeta = str(tmp_path)
    for argv in (["--listar"], ["--verificar"], ["--diff"], ["--registrar", "--origen", "actual", "--autor", "a", "--nota", "n"]):
        assert registrar_prompt.main(["--registro", carpeta, *argv]) == 1
        assert "No se pudo leer el registro: IsADirectoryError." in capsys.readouterr().err


def test_l2b_b_un_registro_ilegible_sale_con_1_y_nombra_solo_el_tipo_del_error(ruta, capsys, monkeypatch):
    def ilegible(ruta_del_registro):
        raise PermissionError("CANARIO-RUTA-PRIVADA")

    monkeypatch.setattr(registro_prompts, "leer_registro", ilegible)
    assert registrar_prompt.main(["--registro", ruta, "--listar"]) == 1
    error = capsys.readouterr().err
    assert error == "No se pudo leer el registro: PermissionError.\n" and "CANARIO" not in error


def test_l2b_c_un_registro_que_no_se_puede_escribir_sale_con_1_sin_traceback(ruta, capsys, monkeypatch):
    def sin_permiso(*args, **kwargs):
        raise PermissionError("CANARIO-RUTA-PRIVADA")

    monkeypatch.setattr(registro_prompts, "registrar_version", sin_permiso)
    assert _registrar(ruta) == 1
    error = capsys.readouterr().err
    assert error == "No se registro: no se pudo escribir el registro (PermissionError).\n" and "CANARIO" not in error


def test_l2b_d_registrar_se_niega_a_anexar_a_una_cadena_rota_y_no_escribe(ruta, capsys):
    assert _registrar(ruta, "plantilla_un_mensaje", "a") == 0
    assert _registrar(ruta, "actual", "b") == 0
    lineas = Path(ruta).read_text(encoding="utf-8").split("\n")[:-1]
    segunda = json.loads(lineas[1])
    segunda["huella_anterior"] = "000000000000"
    Path(ruta).write_text(lineas[0] + "\n" + json.dumps(segunda) + "\n", encoding="utf-8")
    antes = Path(ruta).read_bytes()
    capsys.readouterr()
    assert _registrar(ruta, "plantilla_un_mensaje", "c") == 1
    assert "No se registro: el registro tiene problemas: linea 2: huella_anterior no encadena con la version previa" in capsys.readouterr().err
    assert Path(ruta).read_bytes() == antes
