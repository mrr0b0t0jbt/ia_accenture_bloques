from unittest.mock import patch, Mock

import pytest

import api_claude_refactor
import api_gemini


@pytest.fixture(autouse=True)
def clave_falsa(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "clave-falsa-de-prueba")


def test_get_gemini_response_credencial_invalida_lanza_value_error_sin_unexpected_error(capsys):
    with patch("api_claude_refactor.requests.post", return_value=Mock(status_code=401, ok=False)):
        with pytest.raises(ValueError):
            api_gemini.get_gemini_response("prompt de prueba")
    assert "unexpected error" not in capsys.readouterr().out


def test_get_gemini_response_error_http_400_lanza_runtime_error_sin_unexpected_error(capsys):
    with patch("api_claude_refactor.requests.post", return_value=Mock(status_code=400, ok=False)):
        with pytest.raises(RuntimeError, match="Ocurrio un error de conexión: 400"):
            api_gemini.get_gemini_response("prompt de prueba")
    assert "unexpected error" not in capsys.readouterr().out


def test_get_gemini_response_error_inesperado_imprime_y_relanza(capsys):
    with patch("api_gemini.with_retry", side_effect=KeyError("simulado")):
        with pytest.raises(KeyError):
            api_gemini.get_gemini_response("prompt de prueba")
    assert "unexpected error" in capsys.readouterr().out
