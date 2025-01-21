"""Módulo com testes para as funções utilitárias de logging."""

import logging
from unittest.mock import MagicMock, patch

import pytest

from infrastructure.logging.log_config import get_logger


@pytest.fixture
def mock_settings():
    """Fixture para mockar as configurações do Dynaconf."""
    with patch("config.dynaconf_settings.settings") as mock:
        mock.app.loglevel = "INFO"
        yield mock


class TestLoggingModule:
    """Conjunto de testes para o módulo de configuração de logging."""

    def test_get_logger_default_level(self, mock_settings):
        """Deve criar um logger com o nível padrão definido nas configurações."""
        logger_name = "test_logger"
        logger = get_logger(logger_name)

        assert isinstance(
            logger, logging.Logger
        ), "O objeto retornado deve ser uma instância de logging.Logger."
        assert logger.name == logger_name, "O nome do logger deve corresponder ao fornecido."
        assert logger.level == logging.INFO, "O nível do logger deve ser INFO por padrão."

    def test_get_logger_custom_level(self):
        """Deve criar um logger com um nível de log personalizado."""
        logger_name = "custom_logger"
        logger = get_logger(logger_name, level="DEBUG")

        assert (
            logger.level == logging.DEBUG
        ), "O nível do logger deve ser DEBUG quando especificado."

    def test_get_logger_propagation_disabled(self):
        """Deve desabilitar a propagação do logger quando propagate é False."""
        logger_name = "no_propagation_logger"
        logger = get_logger(logger_name, propagate=False)

        assert not logger.propagate, "A propagação deve ser desabilitada quando propagate é False."

    def test_get_logger_propagation_enabled(self):
        """Deve habilitar a propagação do logger quando propagate é True."""
        logger_name = "propagation_logger"
        logger = get_logger(logger_name, propagate=True)

        assert logger.propagate, "A propagação deve ser habilitada quando propagate é True."

    @patch("logging.StreamHandler")
    def test_get_logger_creates_handler(self, mock_handler):
        """Deve criar um handler de stream se nenhum handler existir no logger."""
        logger_name = "handler_test_logger"
        mock_handler_instance = MagicMock()
        mock_handler.return_value = mock_handler_instance

        logger = get_logger(logger_name)

        mock_handler.assert_called_once(), "Deve criar um StreamHandler se nenhum existir."
        assert logger.hasHandlers(), "O logger deve ter pelo menos um handler configurado."

    def test_get_logger_uses_existing_handler(self):
        """Não deve criar novos handlers se o logger já tiver handlers configurados."""
        logger_name = "existing_handler_logger"
        logger = get_logger(logger_name)

        # Recria o logger, mas não deve adicionar um novo handler.
        logger = get_logger(logger_name)

        assert (
            len(logger.handlers) == 1
        ), "O logger não deve criar handlers adicionais se já existir um."
