"""
Módulo de configuração do aplicativo.
"""

from config import settings

YOLOV8X_MODEL_DROPBOX_PATH = settings.dropbox.models.yolov8x
"""Path do modelo YOLOv8X no Dropbox."""
PUBLIC_SAFETY_MODEL_DROPBOX_PATH = settings.dropbox.models.public_safety
"""Path do modelo de segurança pública no Dropbox."""
DROPBOX_ACCESS_TOKEN = settings.dropbox.access_token
"""Token de acesso do Dropbox."""
