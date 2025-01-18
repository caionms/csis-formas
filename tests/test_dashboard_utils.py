"""Módulo com testes para as funções utilitárias do dashboard."""

import json
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from infrastructure.utils.dashboard_utils import (
    _read_json_file,
    _write_json_file,
    save_annotated_image,
    save_plate_results_to_json,
    save_results_to_json,
)
from infrastructure.utils.plate_utils import PlateType, VehicleEnum


@pytest.fixture
def mock_json_file(tmp_path):
    """Fixture para criar um arquivo JSON temporário."""
    file_path = tmp_path / "test_results.json"
    file_path.write_text("[]")  # JSON inicial vazio
    return str(file_path)


@pytest.fixture
def mock_image():
    """Fixture para criar uma imagem simulada."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def mock_classes_names():
    """Fixture para fornecer um dicionário de nomes de classes simulados."""
    return {0: "car", 1: "person"}


class TestDashboardUtils:
    """Conjunto de testes para as funções utilitárias do dashboard."""

    @patch("application.utils.dashboard_utils.os.path.exists", return_value=True)
    @patch("application.utils.dashboard_utils.open", new_callable=mock_open, read_data="[]")
    def test_read_json_file(self, mock_file, mock_exists):
        """Deve retornar uma lista vazia para um JSON inicial vazio."""
        result = _read_json_file("dummy_path.json")
        assert result == [], "Deve retornar uma lista vazia para um JSON inicial vazio."
        mock_file.assert_called_once_with("dummy_path.json")

    @patch("application.utils.dashboard_utils.json.dump")
    @patch("application.utils.dashboard_utils.open", new_callable=mock_open)
    def test_write_json_file(self, mock_file, mock_dump):
        """Deve escrever um arquivo JSON corretamente."""
        data = [{"key": "value"}]
        _write_json_file("dummy_path.json", data)
        mock_file.assert_called_once_with("dummy_path.json", "w")
        mock_dump.assert_called_once_with(data, mock_file(), indent=4)

    def test_save_results_to_json(self, mock_json_file, mock_classes_names):
        """Deve salvar os resultados de detecção em um arquivo JSON."""
        results = [
            MagicMock(
                boxes=[
                    MagicMock(cls=0, conf=0.9, xywh=MagicMock(return_value=[100, 100, 50, 50])),
                    MagicMock(cls=1, conf=0.3, xywh=MagicMock(return_value=[50, 50, 20, 20])),
                ]
            )
        ]
        for box in results[0].boxes:
            box.xywh.tolist = MagicMock(return_value=box.xywh())

        save_results_to_json(
            results=results,
            file_path=mock_json_file,
            classes_names=mock_classes_names,
            model_name="test_model",
            camera_location="Test Location",
        )

        with open(mock_json_file) as f:
            saved_data = json.load(f)

        assert len(saved_data) == 1
        assert saved_data[0]["detections"][0]["class"] == "car (0)"
        assert saved_data[0]["detections"][0]["confidence"] == 0.9

    @patch("application.utils.dashboard_utils.os.makedirs")
    @patch("application.utils.dashboard_utils.cv.imwrite", return_value=True)
    def test_save_annotated_image_success(self, mock_imwrite, mock_makedirs, mock_image, tmp_path):
        """Deve salvar uma imagem anotada com sucesso."""
        folder_path = str(tmp_path / "images")
        result = save_annotated_image(mock_image, folder_path)
        assert result is not None, "Deve retornar o caminho do arquivo salvo."
        mock_makedirs.assert_called_once_with(folder_path, exist_ok=True)
        mock_imwrite.assert_called_once()

    @patch("application.utils.dashboard_utils.cv.imwrite", side_effect=Exception("Test exception"))
    def test_save_annotated_image_failure(self, mock_imwrite, mock_image, tmp_path):
        """Deve retornar None em caso de falha ao salvar a imagem."""
        folder_path = str(tmp_path / "images")
        result = save_annotated_image(mock_image, folder_path)
        assert result is None, "Deve retornar None em caso de falha ao salvar a imagem."

    def test_save_plate_results_to_json(self, mock_json_file):
        """Deve salvar os resultados de detecção de placas em um arquivo JSON."""
        save_plate_results_to_json(
            file_path=mock_json_file,
            type_of_camera=VehicleEnum.IN,
            plate_text="ABC1234",
            plate_type=PlateType.MERCOSUL,
            camera_location="Entrance Gate",
        )
        with open(mock_json_file) as f:
            saved_data = json.load(f)
        assert len(saved_data) == 1
        assert saved_data[0]["plate_text"] == "ABC1234"
        assert saved_data[0]["plate_type"] == PlateType.MERCOSUL.value
