"""
Módulo que executa detecção de segurança pública em uma janela.
"""

import importlib.resources as pkg_resources
import os
from pathlib import Path
from time import time
from typing import Any

import cv2 as cv
import numpy as np
import yaml

from application import (
    DROPBOX_ACCESS_TOKEN,
    YOLOV8X_MODEL_DROPBOX_PATH,
)
from application.log_config import get_logger
from application.utils.dashboard_utils import save_annotated_image, save_results_to_json
from application.utils.model_utils import (
    NoModelAvailableException,
    download_model,
    initialize_yolo_model,
)
from application.utils.plot_utils import plot_bbox
from application.utils.window_capture_utils import capture_window, setup_capture_window

logger = get_logger(__name__)

DATA_FOLDER_PATH = Path(__file__).parents[1] / "data"
FRAMES_FOLDER_PATH = DATA_FOLDER_PATH / "frames"
MODELS_FOLDER_PATH = Path(__file__).parents[1] / "models"

TrackingData = dict[int, dict[str, Any]]


def update_tracked_objects(
    tracks_ids: list[int], current_time: float, tracking_data: TrackingData
) -> None:
    """
    Atualiza o tempo de rastreamento dos objetos e controla o estado dos mesmos.

    Args:
        tracks_ids (list[Any]): The list of tracker results ids for the current frame.
        current_time (float): Timestamp atual (tempo atual em segundos).
        tracking_data (Dict[int, Dict[str, Any]]): The dictionary holding tracking information
        for each ID.
    """
    for track_id in tracks_ids:
        if track_id is None:
            continue  # Se não houver ID, o objeto não está sendo rastreado

        if track_id not in tracking_data:
            # Novo objeto sendo rastreado
            tracking_data[track_id] = {
                "start_time": current_time,  # Quando o objeto começou a ser rastreado
                "last_seen_time": current_time,  # Último tempo que o objeto foi visto
                "total_time_tracked": 0,  # Tempo total que o objeto foi rastreado
                "alert_sent": False,  # Diz se o alerta foi enviado
            }
        else:
            # Atualiza o último tempo que o objeto foi visto
            last_seen_time: float = tracking_data[track_id]["last_seen_time"]
            tracking_data[track_id]["total_time_tracked"] += current_time - last_seen_time
            tracking_data[track_id]["last_seen_time"] = current_time


def remove_stale_tracks(
    tracking_data: dict[int, dict[str, Any]], current_time: float, expiration_time: int = 240
) -> None:
    """
    Remove objects from tracking_data that haven't been seen for the specified
    expiration time (in seconds).

    Args:
        tracking_data (Dict[int, Dict[str, Any]]): Dictionary of tracked objects.
        current_time (float): Current timestamp.
        expiration_time (int): The time (in seconds) after which an object is considered stale.
    """
    stale_ids = [
        track_id
        for track_id, data in tracking_data.items()
        if current_time - data["last_seen_time"] > expiration_time
    ]

    for stale_id in stale_ids:
        logger.info(
            f"Removing track ID {stale_id} from tracking data (inactive for "
            f"{expiration_time} seconds)."
        )
        del tracking_data[stale_id]


def main(
    window_title: str | None = None,
    output_json_path: Path = DATA_FOLDER_PATH / "output.json",
    image_folder_path: Path = FRAMES_FOLDER_PATH,
) -> None:
    """
    Captura continuamente a tela de uma janela específica ou da área de trabalho,
    realiza detecção com o modelo YOLO e salva os resultados em um arquivo JSON.

    A função exibe os frames anotados com as detecções em uma janela OpenCV e salva os
    resultados de detecção e a imagem anotada a cada segundo, caso haja detecções.

    Args:
        window_title (Optional[str]): O título da janela a ser capturada. Se não for
            especificado, captura a área de trabalho.
        output_json_path (Path): O caminho do arquivo JSON onde os resultados das
            detecções serão salvos.
        image_folder_path (Path): O caminho da pasta onde as imagens anotadas serão salvas.
    """
    # Change the working directory to the folder this script is in.
    # Doing this because I'll be putting the files from each video in their own folder on GitHub
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Get the max_time_lost to calculate the time that a track is lost
    # Access the YAML file inside the `ultralytics.cfg` package
    with pkg_resources.open_text("ultralytics.cfg.trackers", "botsort.yaml") as file:
        config = yaml.safe_load(file)

    # Get the value of 'track_buffer'
    track_buffer = config.get("track_buffer")
    logger.info(f"[SuspiciousBehaviorDetection] The value of track_buffer is: {track_buffer}")

    # Prepara captura de janela
    window_id = setup_capture_window(window_title)

    # Load the model
    model_filename = YOLOV8X_MODEL_DROPBOX_PATH.split("/")[-1]
    model_path = MODELS_FOLDER_PATH / model_filename

    try:
        download_model(model_path, YOLOV8X_MODEL_DROPBOX_PATH, DROPBOX_ACCESS_TOKEN)
    except NoModelAvailableException as e:
        logger.error(f"[SuspiciousBehaviorDetection] {e} Detection cannot be performed.")
        return

    model = initialize_yolo_model(model_path)

    # Obtem o nome das classes
    classes_names = model.names

    # Cria a pasta de frames se ela não existir (e consequentemente a de dados)
    image_folder_path.mkdir(parents=True, exist_ok=True)

    # Dictionary to store tracking data (presence and absence) by ID
    tracking_data: dict[int, dict[str, Any]] = TrackingData()

    while True:
        loop_time = time()

        screenshot = capture_window(window_id=window_id)
        screenshot = np.ascontiguousarray(screenshot)
        # screenshot = screenshot.astype(np.uint8)

        # Run YOLOv8 inference on the frame
        results = list(model.track(source=screenshot, classes=[0], persist=True, stream=True))

        if len(results[0].boxes) > 0 and any([box.id for box in results[0].boxes]):
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int()
            confidences = results[0].boxes.conf.tolist()

            # Update tracked objects
            update_tracked_objects(
                tracks_ids=[track_id for i, track_id in enumerate(track_ids) if classes[i] == 0],
                current_time=loop_time,
                tracking_data=tracking_data,
            )

            suspects_ids = []
            for box, track_id, cls, confidence in zip(boxes, track_ids, classes, confidences):
                color: tuple[int, int, int] | None = None
                if cls == 0 and track_id is not None:  # Class 0 indicates a person
                    total_time = tracking_data.get(track_id, {}).get("total_time_tracked", 0)

                    # Check if the person has been present for more than the suspicious time limit
                    if total_time > 180:  # limite de 3 minutos
                        suspects_ids.append(track_id)
                        label = f"{track_id}: {round(total_time,2)}s (suspect)"
                        color = (0, 0, 255)  # Red for suspicious persons
                    else:
                        label = f"{track_id}: {total_time}s"
                else:
                    # For vehicles, display the confidence score
                    label = f"vehicle: {confidence:.2f}"

                plot_bbox(
                    img=screenshot, class_id=int(cls), box_coordinates=box, label=label, color=color
                )

            if len(suspects_ids) > 0:
                frame_path = save_annotated_image(screenshot, str(image_folder_path))

                for suspect_id in suspects_ids:
                    save_results_to_json(
                        results=results,
                        file_path=str(output_json_path),
                        frame_path=frame_path,
                        model_name=model_filename,
                        classes_names=classes_names,
                    )
                    tracking_data[suspect_id]["alert_sent"] = True

        # Display the annotated frame
        cv.imshow("Suspicious Behavior Inference", screenshot)

        # Remove stale tracks
        remove_stale_tracks(tracking_data, current_time=loop_time)

        # Debug da taxa de atualização
        logger.info(f"FPS: {1 / (time() - loop_time):.2f}")

        if cv.waitKey(1) == ord("q"):
            print(f"{loop_time} - {tracking_data}")
            cv.destroyAllWindows()
            break

    logger.info("Done.")


if __name__ == "__main__":
    main(
        window_title="Reprodutor Multimídia",
    )
