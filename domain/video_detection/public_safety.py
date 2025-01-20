"""
Módulo que executa detecção de segurança pública em um vídeo.
"""

import os
from pathlib import Path
from time import time

import cv2 as cv

from config.globals import DROPBOX_ACCESS_TOKEN, PUBLIC_SAFETY_MODEL_DROPBOX_PATH
from config.paths import (
    DATA_FOLDER_PATH,
    FRAMES_FOLDER_PATH,
    MODELS_FOLDER_PATH,
    VIDEOS_FOLDER_PATH,
)
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
    video_path: str,
    save_video: bool = False,
    show_video: bool = True,
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
        video_path (str): O caminho do vídeo a ser executado.
        save_video (bool): Se True, salva o vídeo anotado.
        show_video (bool): Se True, exibe o vídeo anotado.
        output_json_path (Path): O caminho do arquivo JSON onde os resultados das
            detecções serão salvos.
        image_folder_path (Path): O caminho da pasta onde as imagens anotadas serão salvas.
        camera_location (str): O local da câmera onde a detecção está sendo realizada.
    """
    # Change the working directory to the folder this script is in.
    # Doing this because I'll be putting the files from each video in their own folder on GitHub
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Extract the video file name
    video_name = Path(video_path).name

    # Open the video
    cap = cv.VideoCapture(video_path)

    if cap is None:
        logger.error(f"[PublicSafety_VideoDetection] Could not open video file: {video_path}")
        return

    # Configure video saving if necessary
    if save_video:
        VIDEOS_FOLDER_PATH.mkdir(parents=True, exist_ok=True)
        output_file = (
            VIDEOS_FOLDER_PATH / f"{Path(video_name).stem}_output{Path(video_name).suffix}"
        )
        fps = cap.get(cv.CAP_PROP_FPS) or 30.0
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        out = cv.VideoWriter(
            output_file,
            fourcc,
            fps,
            (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))),
        )

    # Load the model
    model_filename = PUBLIC_SAFETY_MODEL_DROPBOX_PATH.split("/")[-1]
    model_path = MODELS_FOLDER_PATH / model_filename

    try:
        download_model(model_path, PUBLIC_SAFETY_MODEL_DROPBOX_PATH, DROPBOX_ACCESS_TOKEN)
    except NoModelAvailableException as e:
        logger.error(f"[PublicSafety_VideoDetection] {e} Detection cannot be performed.")
        return

    model = initialize_yolo_model(model_path)

    # Obtem o nome das classes
    classes_names = model.names

    # Cria a pasta de frames se ela não existir (e consequentemente a de dados)
    image_folder_path.mkdir(parents=True, exist_ok=True)

    last_save_time = 0
    while cap.isOpened():
        loop_time = time()

        # Read the current frame
        success, frame = cap.read()

        # Check if the read was successful and the frame is not None
        if not success or frame is None:
            logger.error("[PublicSafety_VideoDetection] Could not read frame from video.")
            break

        # Obtém o tempo atual em milissegundos
        current_time_ms = cap.get(cv.CAP_PROP_POS_MSEC)
        # Converte para segundos
        current_time_sec = current_time_ms / 1000

        # Run YOLOv11 inference on the frame
        results = model(frame)

        # Display the annotated frame
        annotated_frame = results[0].plot()

        # Salva imagem anotada e resultados em um arquivo JSON a cada segundo se houver detecções
        if current_time_sec - last_save_time >= 1.0 and len(results[0].boxes) > 0:
            last_save_time = current_time_sec
            persons = []
            weapons = []

            for box, cls in zip(
                results[0].boxes.xyxy.cpu(),
                results[0].boxes.cls.int(),
            ):
                if cls == 9 and box is not None:
                    persons.append(box)
                elif cls in [2, 5] and box is not None:
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

            ignore_classes = [10] if weapon_overlaps_person else [2, 5, 10]

            frame_path = save_annotated_image(
                annotated_frame, str(image_folder_path), current_time_sec
            )

            save_results_to_json(
                results=results,
                file_path=str(output_json_path),
                frame_path=frame_path,
                model_name=model_filename,
                classes_names=classes_names,
                camera_location=camera_location,
                ignore_classes=ignore_classes,
                video_time=current_time_sec,
            )

        if show_video:
            cv.imshow("Public Safety Inference", annotated_frame)

        if save_video and out is not None:
            out.write(annotated_frame)

        # Debug da taxa de atualização
        logger.info(f"[PublicSafety_VideoDetection] FPS: {1 / (time() - loop_time):.2f}")

        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    if save_video and out is not None:
        out.release()
    cv.destroyAllWindows()

    logger.info("[PublicSafety_VideoDetection] Done.")


if __name__ == "__main__":
    main(
        video_path="D:\\Documents\\TCC\\ICs\\Natan\\drive\\Formas\\incendio.mp4",
        save_video=True,
        show_video=True,
    )
