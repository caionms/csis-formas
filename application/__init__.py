"""
Módulo de configuração do aplicativo.
"""

from config import settings

PUBLIC_SAFETY_MODEL_DROPBOX_PATH = settings.dropbox.models.yolo.public_safety.public_safety
"""Path do modelo de segurança pública no Dropbox."""
FIRE_SMOKE_MODEL_DROPBOX_PATH = settings.dropbox.models.yolo.public_safety.fire_smoke_detection
"""Path do modelo de detecção de fogo e fumaça no Dropbox."""
FLOOD_MODEL_DROPBOX_PATH = settings.dropbox.models.yolo.public_safety.flood_detection
"""Path do modelo de detecção de alagamento no Dropbox."""
WEAPON_MODEL_DROPBOX_PATH = settings.dropbox.models.yolo.public_safety.weapon_detection
"""Path do modelo de detecção de armas no Dropbox."""
GRAFFITI_SPRAY_MODEL_DROPBOX_PATH = (
    settings.dropbox.models.yolo.public_safety.graffiti_spray_detection
)
"""Path do modelo de detecção de spray e graffiti no Dropbox."""

YOLO11X_MODEL_DROPBOX_PATH = settings.dropbox.models.yolo.yolo11x
"""Path do modelo YOLO11X no Dropbox."""
PLATE_YOLO_DETECTION_MODEL_DROPBOX_PATH = settings.dropbox.models.yolo.plate_detection
"""Path do modelo de detecção de placas no Dropbox."""

PLATE_PADDLE_RECOGNITION_MODEL_DROPBOX_PATH = settings.dropbox.models.paddleocr.plate_recognition
"""Path do modelo de reconhecimento de texto em placas no Dropbox."""
PLATE_PADDLE_DETECTION_MODEL_DROPBOX_PATH = settings.dropbox.models.paddleocr.plate_detection
"""Path do modelo de detecção de texto em placas no Dropbox."""
PLATE_PADDLE_CLS_MODEL_DROPBOX_PATH = settings.dropbox.models.paddleocr.plate_cls
"""Path do modelo de classificação de texto em placas no Dropbox."""

DROPBOX_ACCESS_TOKEN = settings.dropbox.access_token
"""Token de acesso do Dropbox."""
