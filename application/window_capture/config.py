"""Módulo de configuração de caminhos de pastas."""

from pathlib import Path

DATA_FOLDER_PATH = Path(__file__).parents[1] / "data"
FRAMES_FOLDER_PATH = DATA_FOLDER_PATH / "frames"
MODELS_FOLDER_PATH = Path(__file__).parents[1] / "models"
