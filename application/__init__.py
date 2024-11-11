"""
Módulo de configuração do aplicativo.
"""

from config import settings

YOLOV8X_MODEL_DROPBOX_PATH = settings.dropbox.models.yolov8x
"""Path do modelo YOLOv8X no Dropbox."""
PUBLIC_SAFETY_MODEL_DROPBOX_PATH = settings.dropbox.models.public_safety
"""Path do modelo de segurança pública no Dropbox."""
PLATE_YOLO_DETECTION_MODEL_DROPBOX_PATH = settings.dropbox.models.plate_yolo_detection
"""Path do modelo de detecção de placas no Dropbox."""
PLATE_PADDLE_RECOGNITION_MODEL_DROPBOX_PATH = settings.dropbox.models.plate_paddle_recognition
"""Path do modelo de reconhecimento de texto em placas no Dropbox."""
PLATE_PADDLE_DETECTION_MODEL_DROPBOX_PATH = settings.dropbox.models.plate_paddle_detection
"""Path do modelo de detecção de texto em placas no Dropbox."""
DROPBOX_ACCESS_TOKEN = settings.dropbox.access_token
"""Token de acesso do Dropbox."""
