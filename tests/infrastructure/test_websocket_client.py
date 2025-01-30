"""
Testes unitários para o WebSocketClient utilizando pytest.
"""

from unittest.mock import MagicMock, patch

import pytest

from infrastructure.websocket.websocket_client import WebSocketClient


@pytest.fixture
def mock_sio():
    """Fixture que cria um mock para o objeto socketio.Client."""
    mock = MagicMock()
    return mock


@pytest.fixture
def websocket_client(mock_sio):
    """Fixture para inicializar o WebSocketClient com um mock de socketio."""
    with patch("infrastructure.websocket.websocket_client.socketio.Client", return_value=mock_sio):
        client = WebSocketClient(server_url="http://localhost:3000", max_retries=3, retry_delay=0.1)
    return client


class TestWebSocketClient:
    """Testes para a classe WebSocketClient."""

    def test_initialization(self, websocket_client):
        """Deve inicializar corretamente com os parâmetros fornecidos."""
        assert websocket_client.server_url == "http://localhost:3000"
        assert websocket_client.max_retries == 3
        assert websocket_client.retry_delay == 0.1

    def test_connect_success(self, websocket_client, mock_sio):
        """Deve conectar com sucesso na primeira tentativa."""
        mock_sio.connect.return_value = None  # Simula conexão bem-sucedida

        assert websocket_client.connect() is True
        mock_sio.connect.assert_called_once_with("http://localhost:3000", transports=["websocket"])

    def test_connect_with_retries(self, websocket_client, mock_sio):
        """Deve tentar conectar várias vezes antes de falhar."""
        mock_sio.connect.side_effect = Exception("Conexão falhou")  # Simula falha

        assert websocket_client.connect() is False
        assert mock_sio.connect.call_count == 3  # Deve tentar conectar 3 vezes

    def test_disconnect(self, websocket_client, mock_sio):
        """Deve chamar o método de desconexão corretamente."""
        websocket_client.disconnect()
        mock_sio.disconnect.assert_called_once()

    def test_send_detection_success(self, websocket_client, mock_sio):
        """Deve enviar dados de detecção corretamente via WebSocket."""
        detection_data = {"class": "fire", "confidence": 0.98}
        frame_bytes = b"image_data"

        websocket_client.send_detection(detection_data, frame_bytes)
        mock_sio.emit.assert_called_once_with(
            "new_detection", {"detectionData": detection_data, "frame": frame_bytes}
        )

    def test_send_detection_without_connection(self, websocket_client, mock_sio):
        """Deve falhar silenciosamente se não estiver conectado ao WebSocket."""
        mock_sio.emit.side_effect = Exception("Falha no envio")

        detection_data = {"class": "weapon", "confidence": 0.90}
        frame_bytes = b"image_data"

        try:
            websocket_client.send_detection(detection_data, frame_bytes)
            assert True  # O teste deve continuar sem levantar exceção
        except Exception:
            assert False, "O método send_detection não deveria gerar exceção"

    def test_is_connected(self, websocket_client, mock_sio):
        """Deve retornar corretamente se está conectado ao WebSocket."""
        mock_sio.connected = True
        assert websocket_client.is_connected() is True

        mock_sio.connected = False
        assert websocket_client.is_connected() is False
