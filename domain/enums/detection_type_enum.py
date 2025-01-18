"""Módulo de enumerações de tipos de detecção."""

from enum import Enum

from application import (
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
    SUSPICIOUS_BEHAVIOR = "SUSPICIOUS_BEHAVIOR"
    """Tipo de detecção de comportamento suspeito."""
    PUBLIC_SAFETY = PUBLIC_SAFETY_MODEL_DROPBOX_PATH
    """Tipo de detecção de segurança pública."""
    FIRE_SMOKE_DETECTION = FIRE_SMOKE_MODEL_DROPBOX_PATH
    """Tipo de detecção de fogo e fumaça."""
    FLOOD_DETECTION = FLOOD_MODEL_DROPBOX_PATH
    """Tipo de detecção de alagamento."""
    WEAPON_DETECTION = WEAPON_MODEL_DROPBOX_PATH
    """Tipo de detecção de armas."""
    GRAFFITI_SPRAY_DETECTION = GRAFFITI_SPRAY_MODEL_DROPBOX_PATH
