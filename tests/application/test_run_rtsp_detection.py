# Arquivo de testes usando pytest
"""Testes para o módulo RTSP detection."""

from unittest.mock import patch

import pytest

from application.use_cases.run_rtsp_detection import main
from domain.enums.detection_type_enum import DetectionTypeEnum
from domain.enums.plate_enum import VehicleEnum


@pytest.fixture
def mock_public_safety():
    """Mock para o módulo public_safety."""
    with patch("domain.rtsp_detection.public_safety.main") as mock:
        yield mock


@pytest.fixture
def mock_plate_recognition():
    """Mock para o módulo plate_recognition."""
    with patch("domain.rtsp_detection.plate_recognition.main") as mock:
        yield mock


@pytest.fixture
def mock_suspicious_behavior():
    """Mock para o módulo suspicious_behavior."""
    with (
        patch(
            "domain.rtsp_detection.suspicious_behavior.detect_suspicious_presence"
        ) as suspicious_presence,
        patch(
            "domain.rtsp_detection.suspicious_behavior.detect_proximity_to_vehicle"
        ) as proximity_to_vehicle,
        patch(
            "domain.rtsp_detection.suspicious_behavior.detect_proximity_with_pose"
        ) as proximity_with_pose,
    ):
        yield suspicious_presence, proximity_to_vehicle, proximity_with_pose


class TestRTSPDetection:
    """Conjunto de testes para o método principal do módulo RTSP detection."""

    def test_main_public_safety(self, mock_public_safety):
        """Deve chamar o método correto para detecção de segurança pública."""
        main(
            rtsp_url="rtsp://test_stream",
            detection_type=DetectionTypeEnum.PUBLIC_SAFETY,
        )
        mock_public_safety.assert_called_once()

    def test_main_plate_recognition(self, mock_plate_recognition):
        """Deve chamar o método correto para reconhecimento de placas."""
        main(
            rtsp_url="rtsp://test_stream",
            detection_type=DetectionTypeEnum.PLATE_RECOGNITION,
            type_of_camera=VehicleEnum.IN,
            camera_location="Portaria 2",
        )
        mock_plate_recognition.assert_called_once()

    def test_main_suspicious_presence(self, mock_suspicious_behavior):
        """Deve chamar o método correto para detecção de presença suspeita."""
        suspicious_presence, _, _ = mock_suspicious_behavior
        main(
            rtsp_url="rtsp://test_stream",
            detection_type=DetectionTypeEnum.SUSPICIOUS_PRESENCE,
        )
        suspicious_presence.assert_called_once()

    def test_main_suspicious_proximity_to_vehicle(self, mock_suspicious_behavior):
        """Deve chamar o método correto para proximidade suspeita com veículos."""
        _, proximity_to_vehicle, _ = mock_suspicious_behavior
        main(
            rtsp_url="rtsp://test_stream",
            detection_type=DetectionTypeEnum.SUSPICIOUS_PROXIMITY_TO_VEHICLE,
        )
        proximity_to_vehicle.assert_called_once()

    def test_main_suspicious_proximity_with_pose(self, mock_suspicious_behavior):
        """Deve chamar o método correto para proximidade suspeita com pose."""
        _, _, proximity_with_pose = mock_suspicious_behavior
        main(
            rtsp_url="rtsp://test_stream",
            detection_type=DetectionTypeEnum.SUSPICIOUS_PROXIMITY_WITH_POSE,
            suspicion_threshold_standing=100,
            suspicion_threshold_crouched=50,
        )
        proximity_with_pose.assert_called_once()

    def test_main_invalid_detection_type(self):
        """Deve lançar um ValueError para tipo de detecção inválido."""
        with pytest.raises(ValueError, match="Tipo de detecção não suportado: INVALID_TYPE"):
            main(
                rtsp_url="rtsp://test_stream",
                detection_type="INVALID_TYPE",
            )
