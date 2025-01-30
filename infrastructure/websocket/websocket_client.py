"""
Módulo para gerenciar a comunicação via WebSocket com um servidor Socket.IO.
"""

import time
from typing import Any

import socketio

from infrastructure.logging.log_config import get_logger

logger = get_logger(__name__)


class WebSocketClient:
    """Gerencia a conexão e comunicação com um servidor Socket.IO."""

    def __init__(self, server_url: str, max_retries: int = 5, retry_delay: float = 5.0):
        """
        Inicializa o cliente Socket.IO.

        Args:
            server_url (str): URL do servidor Socket.IO.
            max_retries (int): Número máximo de tentativas de conexão.
            retry_delay (float): Tempo em segundos entre tentativas de reconexão.
        """
        self.server_url = server_url
        self.sio = socketio.Client()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._setup_handlers()

    def _setup_handlers(self):
        """Configura os eventos do WebSocket."""

        @self.sio.event
        def connect():
            logger.info(f"Conectado ao servidor WebSocket. SID: {self.sio.sid}")

        @self.sio.event
        def disconnect():
            logger.warning("Desconectado do servidor WebSocket.")

        @self.sio.event
        def connect_error(data):
            logger.error(f"Erro ao conectar ao servidor WebSocket: {data}")

    def connect(self):
        """
        Estabelece a conexão com o servidor WebSocket, com tentativas automáticas
        de reconexão.
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Tentativa {attempt + 1} de conexão com {self.server_url}...")
                self.sio.connect(self.server_url, transports=["websocket"])
                return True
            except Exception as e:
                logger.warning(
                    f"Falha ao conectar ({e}). Tentando novamente em {self.retry_delay} segundos..."
                )
                time.sleep(self.retry_delay)

        logger.error(
            "Máximo de tentativas atingido. Não foi possível conectar ao servidor WebSocket."
        )
        return False

    def disconnect(self):
        """Desconecta do servidor WebSocket."""
        self.sio.disconnect()
        logger.info("Cliente WebSocket desconectado.")

    def send_detection(self, detection_data: dict[str, Any], frame_bytes: bytes):
        """
        Envia os dados da detecção para o servidor WebSocket.

        Args:
            detection_data (dict): Dados da detecção.
            frame_bytes (bytes): Frame da detecção em bytes.
        """
        try:
            self.sio.emit("new_detection", {"detectionData": detection_data, "frame": frame_bytes})
            logger.info("Dados de detecção enviados para o servidor WebSocket.")
        except Exception as e:
            logger.error(f"Falha ao enviar detecção via WebSocket: {e}")

    def is_connected(self) -> bool:
        """Verifica se o cliente está conectado ao servidor WebSocket."""
        return self.sio.connected
