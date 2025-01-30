# Arquivo de testes usando pytest
"""Testes para o módulo de detecção via captura de janela."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from application.use_cases.run_window_capture import main
from domain.enums.detection_type_enum import DetectionTypeEnum
from domain.enums.plate_enum import VehicleEnum

if sys.platform != "win32":
    pytest.skip(
        "Todos os testes deste arquivo só são executados no Windows", allow_module_level=True
    )


@pytest.fixture
def get_expected_paths():
    """Retorna os caminhos base esperados no ambiente de teste."""
    base_path = Path(__file__).parents[2] / "application" / "data"
    return {
        "output_json_path": base_path / "output_plates.json",
        "image_folder_path": base_path / "frames",
    }


class TestWindowCaptureDetection:
    """Conjunto de testes para o módulo de detecção via captura de janela."""

    @patch("domain.window_capture.public_safety.main")
    def test_main_public_safety(self, mock_public_safety, get_expected_paths):
        """Deve chamar o método correto para detecção de segurança pública."""
        paths = get_expected_paths
        main(
            detection_type=DetectionTypeEnum.PUBLIC_SAFETY,
            window_title="Test Window",
        )
        mock_public_safety.assert_called_once_with(
            "Test Window",
            DetectionTypeEnum.PUBLIC_SAFETY,
            paths["output_json_path"],
            paths["image_folder_path"],
            "Portaria 1 - Ondina",
        )

    @patch("domain.window_capture.plate_recognition.main")
    def test_main_plate_recognition(self, mock_plate_recognition, get_expected_paths):
        """Deve chamar o método correto para reconhecimento de placas."""
        paths = get_expected_paths
        main(
            detection_type=DetectionTypeEnum.PLATE_RECOGNITION,
            window_title="Test Window",
            type_of_camera=VehicleEnum.OUT,
            camera_location="Portaria 2",
        )
        mock_plate_recognition.assert_called_once_with(
            "Test Window",
            paths["output_json_path"],
            paths["image_folder_path"],
            VehicleEnum.OUT,
            "Portaria 2",
        )

    @patch("domain.window_capture.suspicious_behavior.detect_suspicious_presence")
    def test_main_suspicious_presence(self, mock_suspicious_presence, get_expected_paths):
        """Deve chamar o método correto para detecção de presença suspeita."""
        paths = get_expected_paths
        main(
            detection_type=DetectionTypeEnum.SUSPICIOUS_PRESENCE,
            window_title="Test Window",
            suspicion_threshold_time=300,
        )
        mock_suspicious_presence.assert_called_once_with(
            "Test Window",
            paths["output_json_path"],
            paths["image_folder_path"],
            "Portaria 1 - Ondina",
            300,
        )

    @patch("domain.window_capture.suspicious_behavior.detect_proximity_to_vehicle")
    def test_main_suspicious_proximity_to_vehicle(
        self, mock_proximity_to_vehicle, get_expected_paths
    ):
        """Deve chamar o método correto para proximidade suspeita com veículos."""
        paths = get_expected_paths
        main(
            detection_type=DetectionTypeEnum.SUSPICIOUS_PROXIMITY_TO_VEHICLE,
            window_title="Test Window",
            suspicion_threshold_time=200,
        )
        mock_proximity_to_vehicle.assert_called_once_with(
            "Test Window",
            paths["output_json_path"],
            paths["image_folder_path"],
            "Portaria 1 - Ondina",
            200,
        )

    @patch("domain.window_capture.suspicious_behavior.detect_proximity_with_pose")
    def test_main_suspicious_proximity_with_pose(
        self, mock_proximity_with_pose, get_expected_paths
    ):
        """Deve chamar o método correto para proximidade suspeita com pose."""
        paths = get_expected_paths
        main(
            detection_type=DetectionTypeEnum.SUSPICIOUS_PROXIMITY_WITH_POSE,
            window_title="Test Window",
            suspicion_threshold_standing=100,
            suspicion_threshold_crouched=50,
        )
        mock_proximity_with_pose.assert_called_once_with(
            "Test Window",
            paths["output_json_path"],
            paths["image_folder_path"],
            "Portaria 1 - Ondina",
            100,
            50,
        )

    def test_main_invalid_detection_type(self):
        """Deve lançar ValueError para tipo de detecção inválido."""
        with pytest.raises(ValueError, match="Tipo de detecção não suportado"):
            main(
                detection_type="INVALID_TYPE",
                window_title="Test Window",
            )
