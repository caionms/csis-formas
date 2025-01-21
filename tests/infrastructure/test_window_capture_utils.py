"""Módulo com testes para as funções utilitárias de captura de janela."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

if sys.platform != "win32":
    pytest.skip(
        "Todos os testes deste arquivo só são executados no Windows", allow_module_level=True
    )


from infrastructure.utils.window_capture_utils import (
    capture_window,
    list_window_names,
    setup_capture_window,
)


@pytest.fixture
def mock_win32gui():
    """Fixture para mockar funções do win32gui."""
    with patch("infrastructure.utils.window_capture_utils.win32gui") as mock:
        mock.IsWindowVisible.return_value = True
        mock.GetWindowText.side_effect = lambda hwnd: f"Window {hwnd}"
        mock.GetDesktopWindow.return_value = 100
        mock.FindWindow.side_effect = lambda _, title: 101 if title == "Test Window" else 0
        mock.GetWindowRect.return_value = (0, 0, 1280, 720)
        mock.SW_SHOWNOACTIVATE = 4
        mock.HWND_TOP = 0
        mock.SWP_NOMOVE = 1
        mock.SWP_NOSIZE = 2
        yield mock


@pytest.fixture
def mock_mss():
    """Fixture para mockar o mss."""
    with patch(
        "infrastructure.utils.window_capture_utils.mss.mss", new_callable=MagicMock
    ) as mock_mss:
        mock_instance = MagicMock()  # Suporte ao gerenciador de contexto
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.__exit__.return_value = None

        # Cria uma matriz 3D simulada com altura, largura e canais
        fake_screenshot = np.zeros((720, 1280, 4), dtype=np.uint8)
        mock_instance.grab.return_value = fake_screenshot

        mock_mss.return_value = mock_instance
        yield mock_mss


class TestWindowCaptureUtils:
    """Conjunto de testes para funções de captura de janela."""

    def test_list_window_names(self, mock_win32gui):
        """Deve listar os nomes de todas as janelas visíveis."""
        list_window_names()
        mock_win32gui.EnumWindows.assert_called_once()

    def test_setup_capture_window_valid(self, mock_win32gui):
        """Deve configurar corretamente a janela para captura."""
        hwnd = setup_capture_window("Test Window")

        assert hwnd == 101, "O identificador da janela retornado deve ser 101."
        mock_win32gui.ShowWindow.assert_called_once_with(101, mock_win32gui.SW_SHOWNOACTIVATE)
        mock_win32gui.SetWindowPos.assert_called_once_with(
            101,
            mock_win32gui.HWND_TOP,
            0,
            0,
            0,
            0,
            mock_win32gui.SWP_NOMOVE | mock_win32gui.SWP_NOSIZE,
        )

    def test_setup_capture_window_invalid(self, mock_win32gui):
        """Deve lançar um ValueError se a janela não for encontrada."""
        with pytest.raises(ValueError, match="Window not found: Invalid Window"):
            setup_capture_window("Invalid Window")

    def test_capture_window_with_id(self, mock_win32gui, mock_mss):
        """Deve capturar corretamente a janela usando o ID."""
        img = capture_window(window_id=100)

        assert img.shape == (
            760,
            1280,
            3,
        ), "A imagem deve ser redimensionada para FIXED_SIZE e conter 3 canais RGB."
        mock_mss.assert_called_once()

    def test_capture_window_with_title(self, mock_win32gui, mock_mss):
        """Deve capturar corretamente a janela usando o título."""
        img = capture_window(window_title="Test Window")

        assert img.shape == (
            760,
            1280,
            3,
        ), "A imagem deve ser redimensionada para FIXED_SIZE e conter 3 canais RGB."
        mock_mss.assert_called_once()

    def test_capture_window_no_id_or_title(self):
        """Deve lançar uma exceção se nenhum ID ou título for fornecido."""
        with pytest.raises(
            Exception, match="Deve ser especificado o identificador ou o título da janela."
        ):
            capture_window()

    def test_capture_window_invalid_title(self, mock_win32gui):
        """Deve lançar uma exceção se o título da janela não for encontrado."""
        with pytest.raises(Exception, match="Janela com o título 'Invalid Window' não encontrada."):
            capture_window(window_title="Invalid Window")
