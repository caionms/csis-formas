"""
Módulo de utilitários para manipulação de modelos.
"""

from pathlib import Path
from typing import Any

import torch
from paddleocr import PaddleOCR
from ultralytics import YOLO

from application.dropbox_manager import DropboxManager
from config import settings

REC_ALGORITHM = settings.paddleocr.rec_algorithm
REC_IMAGE_SHAPE = settings.paddleocr.rec_image_shape
USE_SPACE_CHAR = settings.paddleocr.use_space_char
USE_GPU = settings.paddleocr.use_gpu
CHAR_DICT = Path(__file__).parent / "data" / "en_dict.txt"


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


def download_paddle_folder_model(
    model_path: Path,
    model_dropbox_path: str,
    access_token: str,
) -> None:
    """
    Faz o download da pasta de inferência do modelo do Paddle
    hospedado no Dropbox para o caminho local especificado.

    Args:
        model_path (Path): O caminho local onde o modelo será salvo.
        model_dropbox_path (str): O caminho do modelo no Dropbox.
        access_token (str): O token de acesso do Dropbox.

    Raises:
        NoModelAvailableException: Se não existe modelo disponível e não é possível
            baixar do Dropbox.
    """
    if not model_path.is_dir():
        dropbox_manager = DropboxManager(access_token=access_token)
        if not dropbox_manager.download_folder(
            dropbox_folder_path=model_dropbox_path,
            local_folder_path=str(model_path),
        ):
            raise NoModelAvailableException(
                "No model available and unable to download the model from Dropbox."
            )


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


def initialize_paddleocr_model(
    text_detection_model_path: Path,
    text_recognition_model_path: Path,
    text_cls_model_path: Path,
) -> Any:
    """
    Inicializa o modelo do PaddleOCR.

    Args:
        text_detection_model_path (Path): O caminho do modelo de detecção de texto.
        text_recognition_model_path (Path): O caminho do modelo de reconhecimento de texto.
        text_cls_model_path (Path): O caminho do modelo de classificação de texto.
    """
    ocr = PaddleOCR(
        det_model_dir=str(text_detection_model_path),
        rec_model_dir=str(text_recognition_model_path),
        cls_model_dir=str(text_cls_model_path),
        use_angle_cls=True,
        lang="en",
        rec_algorithm=REC_ALGORITHM,
        rec_image_shape=REC_IMAGE_SHAPE,
        rec_char_dict_path=str(CHAR_DICT),
        use_space_char=USE_SPACE_CHAR,
        use_gpu=USE_GPU,
    )

    return ocr
