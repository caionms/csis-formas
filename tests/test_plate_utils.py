"""Testes para as funções utilitárias de placas."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from infrastructure.utils.plate_utils import (
    PlateType,
    calculate_correct_plate,
    clean_text,
    format_license,
    insert_char_at_position,
    license_complies_format,
    read_license_plate,
    replace_char_at_position,
    run_ocr_inference,
)


@pytest.fixture
def mock_ocr():
    """Fixture para mockar o PaddleOCR."""
    ocr = Mock()
    ocr.ocr = Mock(return_value=[[(None, ("ABC123", 0.98))]])
    return ocr


@pytest.fixture
def mock_image():
    """Fixture para criar uma imagem simulada."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


class TestPlateUtils:
    """Conjunto de testes para as funções de utilitários de placas."""

    def test_replace_char_at_position_valid(self):
        """Deve substituir o caractere corretamente."""
        result = replace_char_at_position("ABC123", "X", 2)
        assert result == "ABX123", "Deve substituir o caractere corretamente."

    def test_replace_char_at_position_invalid(self):
        """Deve retornar ValueError para posição inválida."""
        with pytest.raises(ValueError):
            replace_char_at_position("ABC123", "X", 10)

    def test_insert_char_at_position_valid(self):
        """Deve inserir o caractere corretamente."""
        result = insert_char_at_position("ABC123", "X", 3)
        assert result == "ABCX123", "Deve inserir o caractere corretamente."

    def test_insert_char_at_position_invalid(self):
        """Deve retornar ValueError para posição inválida."""
        with pytest.raises(ValueError):
            insert_char_at_position("ABC123", "X", 10)

    def test_clean_text(self):
        """Deve limpar e formatar o texto corretamente."""
        result = clean_text(" ABC123 ")
        assert result == "ABC123", "Deve limpar e formatar o texto corretamente."

    def test_calculate_correct_plate(self):
        """Deve calcular a placa correta baseada na frequência."""
        plates = ["ABC1234", "ABC1234", "ABC2234"]
        result = calculate_correct_plate(plates)
        assert result == "ABC1234", "Deve calcular a placa correta baseada na frequência."

    def test_calculate_correct_plate_invalid_length(self):
        """Deve retornar ValueError para placas com comprimentos diferentes."""
        plates = ["ABC1234", "ABC123"]
        with pytest.raises(ValueError):
            calculate_correct_plate(plates)

    def test_license_complies_format_valid(self):
        """Deve retornar True para o formato válido."""
        assert license_complies_format(
            "ABC1234", PlateType.OLD
        ), "Deve retornar True para o formato válido."

    def test_license_complies_format_invalid(self):
        """Deve retornar False para o formato inválido."""
        assert not license_complies_format(
            "A3C1234", PlateType.MERCOSUL
        ), "Deve retornar False para o formato inválido."

    def test_format_license_valid(self):
        """Deve formatar a placa corretamente."""
        formatted, success = format_license("AB41234", PlateType.OLD)
        assert success and formatted == "ABA1234", "Deve formatar a placa corretamente."

    def test_format_license_invalid(self):
        """Deve retornar False para placa inválida."""
        formatted, success = format_license("A3C1234", PlateType.MERCOSUL)
        assert not success, "Deve retornar False para placa inválida."

    @patch("application.utils.plate_utils.run_ocr_inference")
    def test_read_license_plate(self, mock_run_ocr, mock_ocr):
        """Deve retornar as placas lidas pelo OCR."""
        mock_run_ocr.side_effect = [(True, "ABC123"), (True, "ABC123")]

        cropped_plate = MagicMock(
            rgb=np.zeros((50, 50, 3)),
            gray=np.zeros((50, 50)),
            plate_type="OLD",
        )

        plates = read_license_plate(cropped_plate, mock_ocr)
        assert plates == ["ABC123", "ABC123"], "Deve retornar as placas lidas pelo OCR."

    def test_run_ocr_inference_success(self, mock_ocr, mock_image):
        """Deve retornar o texto lido pela OCR com sucesso."""
        success, text = run_ocr_inference(mock_image, mock_ocr)
        assert success and text == "ABC123", "Deve retornar o texto lido pela OCR com sucesso."

    def test_run_ocr_inference_failure(self, mock_ocr, mock_image):
        """Deve retornar None quando a OCR falha."""
        mock_ocr.ocr.return_value = None
        success, text = run_ocr_inference(mock_image, mock_ocr)
        assert not success and text is None, "Deve retornar None quando a OCR falha."
