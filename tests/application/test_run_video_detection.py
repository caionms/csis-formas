# Arquivo de testes usando pytest
"""Testes para o módulo de detecção via vídeo."""

from pathlib import Path
from unittest.mock import patch

import pytest

from application.use_cases.run_video_detection import main
from domain.enums.detection_type_enum import DetectionTypeEnum
from domain.enums.plate_enum import VehicleEnum


@pytest.fixture
def mock_public_safety():
    """Mock para o módulo public_safety."""
    with patch("domain.video_detection.public_safety.main") as mock:
        yield mock


@pytest.fixture
def mock_plate_recognition():
    """Mock para o módulo plate_recognition."""
    with patch("domain.video_detection.plate_recognition.main") as mock:
        yield mock


@pytest.fixture
def mock_suspicious_behavior():
    """Mock para o módulo suspicious_behavior."""
    with (
        patch(
            "domain.video_detection.suspicious_behavior.detect_suspicious_presence"
        ) as suspicious_presence,
        patch(
            "domain.video_detection.suspicious_behavior.detect_proximity_to_vehicle"
        ) as proximity_to_vehicle,
        patch(
            "domain.video_detection.suspicious_behavior.detect_proximity_with_pose"
        ) as proximity_with_pose,
    ):
        yield suspicious_presence, proximity_to_vehicle, proximity_with_pose


class TestVideoDetection:
    """Conjunto de testes para o método principal do módulo de detecção via vídeo."""

    @staticmethod
    def get_expected_paths():
        """Retorna os caminhos base esperados no ambiente de teste."""
        base_path = Path(__file__).parents[2] / "application" / "data"
        return {
            "output_json_path": base_path / "output_plates.json",
            "image_folder_path": base_path / "frames",
        }

    def test_main_public_safety(self, mock_public_safety):
        """Deve chamar o método correto para detecção de segurança pública."""
        paths = self.get_expected_paths()
        main(
            video_path="test_video.mp4",
            detection_type=DetectionTypeEnum.PUBLIC_SAFETY,
        )
        mock_public_safety.assert_called_once_with(
            video_path="test_video.mp4",
            save_video=False,
            show_video=True,
            output_json_path=paths["output_json_path"],
            image_folder_path=paths["image_folder_path"],
            camera_location="Portaria 1 - Ondina",
            detection_type=DetectionTypeEnum.PUBLIC_SAFETY,
        )

    def test_main_plate_recognition(self, mock_plate_recognition):
        """Deve chamar o método correto para reconhecimento de placas."""
        paths = self.get_expected_paths()
        main(
            video_path="test_video.mp4",
            detection_type=DetectionTypeEnum.PLATE_RECOGNITION,
            type_of_camera=VehicleEnum.OUT,
            camera_location="Portaria 2",
            save_video=True,
        )
        mock_plate_recognition.assert_called_once_with(
            video_path="test_video.mp4",
            save_video=True,
            show_video=True,
            output_json_path=paths["output_json_path"],
            image_folder_path=paths["image_folder_path"],
            camera_location="Portaria 2",
            type_of_camera=VehicleEnum.OUT,
        )

    def test_main_suspicious_presence(self, mock_suspicious_behavior):
        """Deve chamar o método correto para detecção de presença suspeita."""
        paths = self.get_expected_paths()
        suspicious_presence, _, _ = mock_suspicious_behavior
        main(
            video_path="test_video.mp4",
            detection_type=DetectionTypeEnum.SUSPICIOUS_PRESENCE,
            suspicion_threshold_time=300,
        )
        suspicious_presence.assert_called_once_with(
            video_path="test_video.mp4",
            save_video=False,
            show_video=True,
            output_json_path=paths["output_json_path"],
            image_folder_path=paths["image_folder_path"],
            camera_location="Portaria 1 - Ondina",
            suspicion_threshold_time=300,
        )

    def test_main_suspicious_proximity_to_vehicle(self, mock_suspicious_behavior):
        """Deve chamar o método correto para proximidade suspeita com veículos."""
        paths = self.get_expected_paths()
        _, proximity_to_vehicle, _ = mock_suspicious_behavior
        main(
            video_path="test_video.mp4",
            detection_type=DetectionTypeEnum.SUSPICIOUS_PROXIMITY_TO_VEHICLE,
            suspicion_threshold_time=200,
        )
        proximity_to_vehicle.assert_called_once_with(
            video_path="test_video.mp4",
            save_video=False,
            show_video=True,
            output_json_path=paths["output_json_path"],
            image_folder_path=paths["image_folder_path"],
            camera_location="Portaria 1 - Ondina",
            suspicion_threshold_time=200,
        )

    def test_main_suspicious_proximity_with_pose(self, mock_suspicious_behavior):
        """Deve chamar o método correto para proximidade suspeita com pose."""
        paths = self.get_expected_paths()
        _, _, proximity_with_pose = mock_suspicious_behavior
        main(
            video_path="test_video.mp4",
            detection_type=DetectionTypeEnum.SUSPICIOUS_PROXIMITY_WITH_POSE,
            suspicion_threshold_standing=100,
            suspicion_threshold_crouched=50,
        )
        proximity_with_pose.assert_called_once_with(
            video_path="test_video.mp4",
            save_video=False,
            show_video=True,
            output_json_path=paths["output_json_path"],
            image_folder_path=paths["image_folder_path"],
            camera_location="Portaria 1 - Ondina",
            suspicion_threshold_standing=100,
            suspicion_threshold_crouched=50,
        )

    def test_main_invalid_detection_type(self):
        """Deve lançar um ValueError para tipo de detecção inválido."""
        with pytest.raises(ValueError, match="Tipo de detecção não suportado: INVALID_TYPE"):
            main(
                video_path="test_video.mp4",
                detection_type="INVALID_TYPE",
            )
