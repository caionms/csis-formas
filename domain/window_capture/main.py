"""Módulo principal que executa detecção em uma janela."""

from pathlib import Path

from config.paths import DATA_FOLDER_PATH, FRAMES_FOLDER_PATH
from domain.enums.detection_type_enum import DetectionTypeEnum
from infrastructure.utils.plate_utils import VehicleEnum


def main(
    detection_type: DetectionTypeEnum = DetectionTypeEnum.PUBLIC_SAFETY,
    window_title: str | None = None,
    output_json_path: Path = DATA_FOLDER_PATH / "output_plates.json",
    image_folder_path: Path = FRAMES_FOLDER_PATH,
    type_of_camera: VehicleEnum = VehicleEnum.IN,
    camera_location: str = "Portaria 1 - Ondina",
):
    """
    Captura continuamente a tela de uma janela específica ou da área de trabalho,
    realiza detecção com o modelo respectivo e salva os resultados em um arquivo JSON.

    A função exibe os frames anotados com as detecções em uma janela OpenCV e salva os
    resultados de detecção e a imagem anotada a cada segundo, caso haja detecções.

    Args:
        detection_type (DetectionTypeEnum): O tipo de detecção a ser realizada.
        window_title (Optional[str]): O título da janela a ser capturada. Se não for
            especificado, captura a área de trabalho.
        output_json_path (Path): O caminho do arquivo JSON onde os resultados das
            detecções serão salvos.
        image_folder_path (Path): O caminho da pasta onde as imagens anotadas serão salvas.
        type_of_camera (VehicleEnum): O tipo de câmera (entrada ou saída).
        camera_location (str): O local da câmera onde a detecção está sendo realizada.
    """
    if (
        detection_type == DetectionTypeEnum.PUBLIC_SAFETY
        or DetectionTypeEnum.FIRE_SMOKE_DETECTION
        or DetectionTypeEnum.FLOOD_DETECTION
        or DetectionTypeEnum.WEAPON_DETECTION
        or DetectionTypeEnum.GRAFFITI_SPRAY_DETECTION
    ):
        from domain.window_capture.public_safety import main as public_safety_main

        public_safety_main(window_title, output_json_path, image_folder_path, camera_location)
    elif detection_type == DetectionTypeEnum.PLATE_RECOGNITION:
        from domain.window_capture.plate_recognition import main as plate_recognition_main

        plate_recognition_main(
            window_title, output_json_path, image_folder_path, type_of_camera, camera_location
        )
    elif detection_type == DetectionTypeEnum.SUSPICIOUS_BEHAVIOR:
        from domain.window_capture.suspicious_behavior import (
            detect_suspicious_presence as suspicious_behavior_main,
        )

        suspicious_behavior_main(window_title, output_json_path, image_folder_path, camera_location)
    else:
        raise ValueError(f"Tipo de detecção não suportado: {detection_type}")


if __name__ == "__main__":
    main(DetectionTypeEnum.PUBLIC_SAFETY, "Reprodutor Multimídia")
