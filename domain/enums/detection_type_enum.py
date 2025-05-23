"""Módulo de enumerações de tipos de detecção."""

from enum import Enum

from config.globals import (
    FIRE_SMOKE_MODEL_DROPBOX_PATH,
    FLOOD_MODEL_DROPBOX_PATH,
    GRAFFITI_SPRAY_MODEL_DROPBOX_PATH,
    PUBLIC_SAFETY_MODEL_DROPBOX_PATH,
    WEAPON_MODEL_DROPBOX_PATH,
)


class DetectionTypeEnum(Enum):
    """Enumeração de tipos de detecção."""

    PLATE_RECOGNITION = "PLATE_RECOGNITION"
    """Tipo de detecção de placa."""
    PLATE_RECOGNITION_WITHOUT_TRACKING = "PLATE_RECOGNITION_WITHOUT_TRACKING"
    """Tipo de detecção de placa sem rastreamento."""
    SUSPICIOUS_PRESENCE = "SUSPICIOUS_PRESENCE"
    """Tipo de detecção de comportamento suspeito por presença na câmera."""
    SUSPICIOUS_PROXIMITY_TO_VEHICLE = "SUSPICIOUS_PROXIMITY_TO_VEHICLE"
    """Tipo de detecção de comportamento suspeito por proximidade a veículo."""
    SUSPICIOUS_PROXIMITY_WITH_POSE = "SUSPICIOUS_PROXIMITY_WITH_POSE"
    """Tipo de detecção de comportamento suspeito por proximidade com pose."""
    PUBLIC_SAFETY = PUBLIC_SAFETY_MODEL_DROPBOX_PATH
    """Tipo de detecção de segurança pública."""
    FIRE_SMOKE_DETECTION = FIRE_SMOKE_MODEL_DROPBOX_PATH
    """Tipo de detecção de fogo e fumaça."""
    FLOOD_DETECTION = FLOOD_MODEL_DROPBOX_PATH
    """Tipo de detecção de alagamento."""
    WEAPON_DETECTION = WEAPON_MODEL_DROPBOX_PATH
    """Tipo de detecção de armas."""
    GRAFFITI_SPRAY_DETECTION = GRAFFITI_SPRAY_MODEL_DROPBOX_PATH
    """Tipo de detecção de grafite."""
