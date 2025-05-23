"""Módulo para executar detecção via vídeo."""

from pathlib import Path

from config.paths import DATA_FOLDER_PATH, FRAMES_FOLDER_PATH
from domain.enums.detection_type_enum import DetectionTypeEnum
from domain.enums.plate_enum import VehicleEnum


def main(
    video_path: str,
    save_video: bool = False,
    show_video: bool = True,
    detection_type: DetectionTypeEnum = DetectionTypeEnum.PUBLIC_SAFETY,
    output_json_path: Path = DATA_FOLDER_PATH / "output.json",
    image_folder_path: Path = FRAMES_FOLDER_PATH,
    type_of_camera: VehicleEnum = VehicleEnum.IN,
    camera_location: str = "Portaria 1 - Ondina",
    suspicion_threshold_time: int = 180,
    suspicion_threshold_standing: int = 180,
    suspicion_threshold_crouched: int = 60,
):
    """
    Captura continuamente a tela de uma janela específica ou da área de trabalho,
    realiza detecção com o modelo respectivo e salva os resultados em um arquivo JSON.

    A função exibe os frames anotados com as detecções em uma janela OpenCV e salva os
    resultados de detecção e a imagem anotada a cada segundo, caso haja detecções.

    Args:
        video_path (str): O caminho do vídeo a ser executado.
        save_video (bool): Se True, salva o vídeo anotado.
        show_video (bool): Se True, exibe o vídeo anotado.
        detection_type (DetectionTypeEnum): O tipo de detecção a ser realizada.
        output_json_path (Path): O caminho do arquivo JSON onde os resultados das
            detecções serão salvos.
        image_folder_path (Path): O caminho da pasta onde as imagens anotadas serão salvas.
        type_of_camera (VehicleEnum): O tipo de câmera (entrada ou saída).
        camera_location (str): O local da câmera onde a detecção está sendo realizada.
        suspicion_threshold_time (int): O tempo em segundos para considerar uma presença suspeita.
        suspicion_threshold_standing (int): O tempo em segundos para considerar uma
            presença em pé suspeita.
        suspicion_threshold_crouched (int): O tempo em segundos para considerar uma
            presença agachada suspeita.
    """
    if (
        detection_type == DetectionTypeEnum.PUBLIC_SAFETY
        or detection_type == DetectionTypeEnum.FIRE_SMOKE_DETECTION
        or detection_type == DetectionTypeEnum.FLOOD_DETECTION
        or detection_type == DetectionTypeEnum.WEAPON_DETECTION
        or detection_type == DetectionTypeEnum.GRAFFITI_SPRAY_DETECTION
    ):
        from domain.video_detection.public_safety import main as public_safety_main

        public_safety_main(
            video_path=video_path,
            save_video=save_video,
            show_video=show_video,
            output_json_path=output_json_path,
            image_folder_path=image_folder_path,
            camera_location=camera_location,
            detection_type=detection_type,
        )
    elif detection_type == DetectionTypeEnum.PLATE_RECOGNITION:
        from domain.video_detection.plate_recognition import main as plate_recognition_main

        plate_recognition_main(
            video_path=video_path,
            save_video=save_video,
            show_video=show_video,
            camera_location=camera_location,
            type_of_camera=type_of_camera,
        )
    elif detection_type == DetectionTypeEnum.PLATE_RECOGNITION_WITHOUT_TRACKING:
        from domain.video_detection.plate_recognition import (
            main_without_tracking as plate_recognition_main_without_tracking,
        )

        plate_recognition_main_without_tracking(
            video_path=video_path,
            show_video=show_video,
            type_of_camera=type_of_camera,
            camera_location=camera_location,
            qty_frames_before_detection=3,
        )
    elif detection_type == DetectionTypeEnum.SUSPICIOUS_PRESENCE:
        from domain.video_detection.suspicious_behavior import detect_suspicious_presence

        detect_suspicious_presence(
            video_path=video_path,
            save_video=save_video,
            show_video=show_video,
            output_json_path=output_json_path,
            image_folder_path=image_folder_path,
            camera_location=camera_location,
            suspicion_threshold_time=suspicion_threshold_time,
        )
    elif detection_type == DetectionTypeEnum.SUSPICIOUS_PROXIMITY_TO_VEHICLE:
        from domain.video_detection.suspicious_behavior import detect_proximity_to_vehicle

        detect_proximity_to_vehicle(
            video_path=video_path,
            save_video=save_video,
            show_video=show_video,
            output_json_path=output_json_path,
            image_folder_path=image_folder_path,
            camera_location=camera_location,
            suspicion_threshold_time=suspicion_threshold_time,
        )

    elif detection_type == DetectionTypeEnum.SUSPICIOUS_PROXIMITY_WITH_POSE:
        from domain.video_detection.suspicious_behavior import detect_proximity_with_pose

        detect_proximity_with_pose(
            video_path=video_path,
            save_video=save_video,
            show_video=show_video,
            output_json_path=output_json_path,
            image_folder_path=image_folder_path,
            camera_location=camera_location,
            suspicion_threshold_standing=suspicion_threshold_standing,
            suspicion_threshold_crouched=suspicion_threshold_crouched,
        )
    else:
        raise ValueError(f"Tipo de detecção não suportado: {detection_type}")


if __name__ == "__main__":
    main(
        video_path="D:\\Documents\\TCC\\ICs\\Caio\\20230616_111617.mp4",
        suspicion_threshold_standing=6,
        suspicion_threshold_crouched=3,
        show_video=True,
        save_video=True,
        detection_type=DetectionTypeEnum.SUSPICIOUS_PROXIMITY_WITH_POSE,
    )
