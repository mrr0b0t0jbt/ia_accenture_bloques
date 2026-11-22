import socket

import pytest


def _conectar_con_socket():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conexion:
        conexion.settimeout(1)
        conexion.connect(("192.0.2.1", 9))


def _conectar_ex_con_socket():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conexion:
        conexion.settimeout(1)
        conexion.connect_ex(("192.0.2.1", 9))


@pytest.mark.parametrize(
    "intento",
    [
        pytest.param(lambda: socket.create_connection(("192.0.2.1", 9), timeout=1), id="create_connection"),
        pytest.param(lambda: socket.getaddrinfo("servidor.test", 443), id="getaddrinfo"),
        pytest.param(lambda: socket.gethostbyname("servidor.test"), id="gethostbyname"),
        pytest.param(_conectar_con_socket, id="socket_connect"),
        pytest.param(_conectar_ex_con_socket, id="socket_connect_ex"),
    ],
)
def test_l1_a_la_guarda_de_red_bloquea_toda_conexion_y_resolucion_de_nombres(intento):
    with pytest.raises(RuntimeError, match="RED BLOQUEADA"):
        intento()


def test_l1_b_la_guarda_de_red_no_impide_los_sockets_locales_que_usa_asyncio():
    a, b = socket.socketpair()
    try:
        a.sendall(b"x")
        assert b.recv(1) == b"x"
    finally:
        a.close()
        b.close()
