from unittest.mock import patch, Mock

import pytest

import api_claude_refactor

URL = "https://api.anthropic.test/v1/messages"
HEADERS = {"x-api-key": "clave-falsa-de-prueba"}
PAYLOAD = '{"messages": [{"role": "user", "content": "prompt de prueba"}]}'
RESPUESTA_OK = {
    "content": [{"type": "text", "text": "hola de prueba"}],
    "usage": {"input_tokens": 1000, "output_tokens": 500},
}


@pytest.fixture(autouse=True)
def reiniciar_costo_total(monkeypatch):
    monkeypatch.setattr(api_claude_refactor, "costo_total_sesion", 0.0)


def respuesta_falsa(status_code, json_data=None):
    respuesta = Mock(status_code=status_code, ok=status_code < 400)
    if json_data is not None:
        respuesta.json.return_value = json_data
    return respuesta


# --- send_claude_request ---

@pytest.mark.parametrize("status_code", [401, 403], ids=["401", "403"])
def test_send_claude_request_credencial_invalida_lanza_value_error(status_code):
    with patch("api_claude_refactor.requests.post", return_value=respuesta_falsa(status_code)):
        with pytest.raises(ValueError):
            api_claude_refactor.send_claude_request(URL, HEADERS, PAYLOAD)


def test_send_claude_request_429_lanza_timeout_error():
    with patch("api_claude_refactor.requests.post", return_value=respuesta_falsa(429)):
        with pytest.raises(TimeoutError):
            api_claude_refactor.send_claude_request(URL, HEADERS, PAYLOAD)


@pytest.mark.parametrize("status_code", [500, 503], ids=["500", "503"])
def test_send_claude_request_5xx_lanza_timeout_error(status_code):
    with patch("api_claude_refactor.requests.post", return_value=respuesta_falsa(status_code)):
        with pytest.raises(TimeoutError):
            api_claude_refactor.send_claude_request(URL, HEADERS, PAYLOAD)


def test_send_claude_request_200_devuelve_json():
    respuesta = respuesta_falsa(200, RESPUESTA_OK)
    with patch("api_claude_refactor.requests.post", return_value=respuesta):
        assert api_claude_refactor.send_claude_request(URL, HEADERS, PAYLOAD) == RESPUESTA_OK


# --- with_retry ---

def test_with_retry_reintenta_ante_timeout_error():
    func = Mock(side_effect=[TimeoutError("simulado"), "ok"])
    with patch("api_claude_refactor.time.sleep") as mock_sleep:
        resultado = api_claude_refactor.with_retry(func)
    assert resultado == "ok"
    mock_sleep.assert_called_once()
    assert func.call_count == 2


def test_with_retry_no_reintenta_ante_value_error():
    func = Mock(side_effect=ValueError("simulado"))
    with patch("api_claude_refactor.time.sleep") as mock_sleep:
        with pytest.raises(ValueError):
            api_claude_refactor.with_retry(func)
    assert func.call_count == 1
    mock_sleep.assert_not_called()


def test_with_retry_agotado_relanza_timeout_error_original():
    error = TimeoutError("simulado")
    func = Mock(side_effect=error)
    with patch("api_claude_refactor.time.sleep") as mock_sleep:
        with pytest.raises(TimeoutError) as excinfo:
            api_claude_refactor.with_retry(func, attempts=3)
    assert excinfo.value is error
    assert mock_sleep.call_count == 2


def test_with_retry_espera_acotada_y_creciente():
    func = Mock(side_effect=TimeoutError("simulado"))
    with patch("api_claude_refactor.time.sleep") as mock_sleep:
        with pytest.raises(TimeoutError):
            api_claude_refactor.with_retry(func, attempts=6, base_wait=2, max_wait=5)
    esperas = [llamada.args[0] for llamada in mock_sleep.call_args_list]
    assert len(esperas) == 5
    for n, espera in enumerate(esperas, start=1):
        esperado = min(2 * 2 ** (n - 1), 5)
        assert esperado <= espera <= esperado * 1.1


def test_with_retry_attempts_cero_lanza_value_error():
    func = Mock()
    with pytest.raises(ValueError):
        api_claude_refactor.with_retry(func, attempts=0)
    func.assert_not_called()


# --- registrar_costo ---

def test_registrar_costo_calcula_y_acumula():
    response = {"usage": {"input_tokens": 1000, "output_tokens": 500}}
    assert api_claude_refactor.registrar_costo(response) == pytest.approx(0.007)
    assert api_claude_refactor.costo_total_sesion == pytest.approx(0.007)


def test_registrar_costo_dos_llamadas_acumulan():
    costo_1 = api_claude_refactor.registrar_costo({"usage": {"input_tokens": 1000, "output_tokens": 500}})
    costo_2 = api_claude_refactor.registrar_costo({"usage": {"input_tokens": 300, "output_tokens": 200}})
    assert api_claude_refactor.costo_total_sesion == pytest.approx(costo_1 + costo_2)


# --- get_claude_response ---

def test_get_claude_response_timeout_agotado_lanza_runtime_error(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "clave-falsa-de-prueba")
    with patch("api_claude_refactor.with_retry", side_effect=TimeoutError("simulado")):
        with pytest.raises(RuntimeError, match="El servicio no respondió tras varios intentos. Intenta más tarde."):
            api_claude_refactor.get_claude_response("prompt de prueba")
    assert api_claude_refactor.costo_total_sesion == 0.0


def test_get_claude_response_propaga_value_error_sin_cambios(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "clave-falsa-de-prueba")
    error = ValueError("simulado")
    with patch("api_claude_refactor.with_retry", side_effect=error):
        with pytest.raises(ValueError) as excinfo:
            api_claude_refactor.get_claude_response("prompt de prueba")
    assert excinfo.value is error


def test_get_claude_response_camino_feliz_devuelve_texto_y_costo(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "clave-falsa-de-prueba")
    with patch("api_claude_refactor.with_retry", return_value=RESPUESTA_OK):
        texto, costo = api_claude_refactor.get_claude_response("prompt de prueba")
    assert texto == "hola de prueba"
    assert costo == pytest.approx(0.007)
    assert api_claude_refactor.costo_total_sesion == pytest.approx(0.007)


def test_get_claude_response_429_y_luego_200_registra_costo_una_vez(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "clave-falsa-de-prueba")
    respuestas = [respuesta_falsa(429), respuesta_falsa(200, RESPUESTA_OK)]
    with patch("api_claude_refactor.requests.post", side_effect=respuestas), \
            patch("api_claude_refactor.time.sleep") as mock_sleep:
        texto, costo = api_claude_refactor.get_claude_response("prompt de prueba")
    assert texto == "hola de prueba"
    assert costo == pytest.approx(0.007)
    assert api_claude_refactor.costo_total_sesion == pytest.approx(0.007)
    mock_sleep.assert_called_once()
