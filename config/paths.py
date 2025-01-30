"""Módulo de configuração de caminhos de pastas."""

from pathlib import Path

DATA_FOLDER_PATH = Path(__file__).parents[1] / "application" / "data"
FRAMES_FOLDER_PATH = DATA_FOLDER_PATH / "frames"
PLATES_FOLDER_PATH = DATA_FOLDER_PATH / "plates"
VIDEOS_FOLDER_PATH = DATA_FOLDER_PATH / "videos"
MODELS_FOLDER_PATH = Path(__file__).parents[1] / "infrastructure" / "models"
RESOURCES_FOLDER_PATH = Path(__file__).parents[1] / "infrastructure" / "resources"
