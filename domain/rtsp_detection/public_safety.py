"""
Módulo que executa detecção de segurança pública em uma janela.
"""

import os
from pathlib import Path
from time import time

import cv2 as cv

from config.globals import DROPBOX_ACCESS_TOKEN, PUBLIC_SAFETY_MODEL_DROPBOX_PATH
from config.paths import DATA_FOLDER_PATH, FRAMES_FOLDER_PATH, MODELS_FOLDER_PATH
from domain.enums.detection_type_enum import DetectionTypeEnum
from infrastructure.logging.log_config import get_logger
from infrastructure.utils.dashboard_utils import save_annotated_image, save_results_to_json
from infrastructure.utils.model_utils import (
    NoModelAvailableException,
    download_model,
    initialize_yolo_model,
)
from infrastructure.utils.suspicious_behavior_utils import calculate_bbox_iou

logger = get_logger(__name__)


def main(
    rtsp_url: str,
    show_video: bool = True,
    detection_type: DetectionTypeEnum = DetectionTypeEnum.PUBLIC_SAFETY,
    output_json_path: Path = DATA_FOLDER_PATH / "output.json",
    image_folder_path: Path = FRAMES_FOLDER_PATH,
    camera_location: str = "Portaria 1 - Ondina",
) -> None:
    """
    Captura continuamente a tela de uma janela específica ou da área de trabalho,
    realiza detecção com o modelo YOLO e salva os resultados em um arquivo JSON.

    A função exibe os frames anotados com as detecções em uma janela OpenCV e salva os
    resultados de detecção e a imagem anotada a cada segundo, caso haja detecções.

    Args:
        rtsp_url (str): A URL do stream RTSP a ser capturado.
        show_video (bool): Se True, exibe o vídeo anotado.
        detection_type (DetectionTypeEnum): O tipo de detecção a ser realizada.
        output_json_path (Path): O caminho do arquivo JSON onde os resultados das
            detecções serão salvos.
        image_folder_path (Path): O caminho da pasta onde as imagens anotadas serão salvas.
        camera_location (str): O local da câmera onde a detecção está sendo realizada.
    """
    # Change the working directory to the folder this script is in.
    # Doing this because I'll be putting the files from each video in their own folder on GitHub
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Prepara captura via RTSP
    cap = cv.VideoCapture(rtsp_url)

    if cap is None:
        logger.error(
            f"[PublicSafety_RTSPDetection] Could not open RTSP stream at {rtsp_url}. "
            f"Detection cannot be performed."
        )
        return

    # Load the model
    if detection_type not in (
        DetectionTypeEnum.PUBLIC_SAFETY,
        DetectionTypeEnum.FIRE_SMOKE_DETECTION,
        DetectionTypeEnum.WEAPON_DETECTION,
        DetectionTypeEnum.FLOOD_DETECTION,
        DetectionTypeEnum.GRAFFITI_SPRAY_DETECTION,
    ):
        detection_type = DetectionTypeEnum.PUBLIC_SAFETY
    model_dropbox_path = detection_type.value
    model_filename = model_dropbox_path.split("/")[-1]
    model_path = MODELS_FOLDER_PATH / model_filename

    try:
        download_model(model_path, PUBLIC_SAFETY_MODEL_DROPBOX_PATH, DROPBOX_ACCESS_TOKEN)
    except NoModelAvailableException as e:
        logger.error(f"[PublicSafetyDetection] {e} Detection cannot be performed.")
        return

    model = initialize_yolo_model(model_path)

    # Obtem o nome das classes
    classes_names = model.names
    weapon_ids = [key for key, value in classes_names.items() if value in ["gun", "knife"]]
    person_id = [key for key, value in classes_names.items() if value == "person"]

    # Cria a pasta de frames se ela não existir (e consequentemente a de dados)
    image_folder_path.mkdir(parents=True, exist_ok=True)

    last_save_time = time()
    while True:
        loop_time = time()

        # Read the current frame
        success, frame = cap.read()

        # Check if the read was successful and the frame is not None
        if not success or frame is None:
            logger.error("[PublicSafety_RTSPDetection] Could not read frame from RTSP stream.")
            break

        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Display the annotated frame
        annotated_frame = results[0].plot()
        if show_video:
            cv.imshow("Public Safety Inference", annotated_frame)

        # Salva imagem anotada e resultados em um arquivo JSON a cada segundo se houver detecções
        if time() - last_save_time >= 1.0 and len(results[0].boxes) > 0:
            persons = []
            weapons = []

            for box, cls in zip(
                results[0].boxes.xyxy.cpu(),
                results[0].boxes.cls.int(),
            ):
                if cls in person_id and box is not None:
                    persons.append(box)
                elif cls in weapon_ids and box is not None:
                    weapons.append(box)

            # Verifica se alguma arma se sobrepõe a alguma pessoa
            weapon_overlaps_person = any(
                calculate_bbox_iou(weapon_bbox, person_bbox) > 0
                for weapon_bbox in weapons
                for person_bbox in persons
            )

            # Se não há sobreposição e todas as detecções são armas ou pessoas, ignora
            if not weapon_overlaps_person and len(results) == len(weapons) + len(persons):
                continue

            ignore_classes = person_id if weapon_overlaps_person else (weapon_ids + person_id)

            frame_path = save_annotated_image(annotated_frame, str(image_folder_path))

            save_results_to_json(
                results=results,
                file_path=str(output_json_path),
                frame_path=frame_path,
                model_name=model_filename,
                classes_names=classes_names,
                camera_location=camera_location,
                ignore_classes=ignore_classes,
            )
            last_save_time = time()

        # Debug da taxa de atualização
        logger.info(f"FPS: {1 / (time() - loop_time):.2f}")

        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
    logger.info("[PublicSafety_RTSPDetection] Done.")


if __name__ == "__main__":
    rtsp_url = "rtsp://"
    main(
        rtsp_url=rtsp_url,
    )
