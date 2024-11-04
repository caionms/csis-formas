"""
Módulo de utilitários para manipulação de modelos.
"""

from pathlib import Path

import torch
from ultralytics import YOLO

from application.dropbox_manager import DropboxManager


class NoModelAvailableException(Exception):
    """Exception utilizada para quando não existe modelo disponível e não é
    possível baixar do Dropbox."""

    ...


def download_model(
    model_path: Path,
    model_dropbox_path: str,
    access_token: str,
) -> None:
    """
    Faz o download do modelo do Dropbox para o caminho local especificado.

    Args:
        model_path (Path): O caminho local onde o modelo será salvo.
        model_dropbox_path (str): O caminho do modelo no Dropbox.
        access_token (str): O token de acesso do Dropbox.

    Raises:
        NoModelAvailableException: Se não existe modelo disponível e não é possível
            baixar do Dropbox.
    """
    if not model_path.is_file():
        dropbox_manager = DropboxManager(access_token=access_token)
        if not dropbox_manager.download(
            dropbox_path=model_dropbox_path,
            local_file_path=str(model_path),
        ):
            raise NoModelAvailableException(
                "No model available and unable to download the model from Dropbox."
            )


def download_models(
    models_folder_path: Path,
    models_dropbox_paths: dict[str, str],
    access_token: str,
) -> None:
    """
    Faz o download dos modelos do Dropbox para o caminho local especificado.

    Args:
        models_folder_path (Path): O caminho da pasta onde os modelos serão salvos.
        models_dropbox_paths (dict): Um dicionário com os caminhos dos modelos no
            Dropbox e os respectivos nomes dos modelos.
        access_token (str): O token de acesso do Dropbox.

    Raises:
        NoModelAvailableException: Se não existe modelo disponível e não é possível
            baixar do Dropbox.
    """
    for model_name, model_dropbox_path in models_dropbox_paths.items():
        model_path = models_folder_path / model_name
        download_model(model_path, model_dropbox_path, access_token)


def initialize_yolo_model(
    model_path: Path,
) -> YOLO:
    """
    Inicializa o modelo YOLO e o move para a GPU se disponível.

    Args:
        model_path (Path): O caminho do modelo YOLO.
    """
    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    return model
