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
from scipy.optimize import linear_sum_assignment

from config.globals import (
    DROPBOX_ACCESS_TOKEN,
    YOLO11X_MODEL_DROPBOX_PATH,
    YOLO11X_POSE_MODEL_DROPBOX_PATH,
)
from config.paths import DATA_FOLDER_PATH, FRAMES_FOLDER_PATH, MODELS_FOLDER_PATH
from domain import TrackingData
from domain.enums.pose_state_enum import PoseStateEnum
from infrastructure.logging.log_config import get_logger
from infrastructure.utils.dashboard_utils import save_annotated_image, save_results_to_json
from infrastructure.utils.model_utils import (
    NoModelAvailableException,
    download_model,
    initialize_yolo_model,
)
from infrastructure.utils.plot_utils import plot_bbox, plot_keypoints_detection
from infrastructure.utils.suspicious_behavior_utils import (
    calculate_bbox_iou,
    is_squat,
    remove_stale_tracks,
    update_tracked_objects,
    update_tracked_objects_proximity_to_vehicle,
)
from infrastructure.utils.window_capture_utils import capture_window, setup_capture_window

logger = get_logger(__name__)


def detect_suspicious_presence(
    window_title: str | None = None,
    output_json_path: Path = DATA_FOLDER_PATH / "output.json",
    image_folder_path: Path = FRAMES_FOLDER_PATH,
    camera_location: str = "Portaria 1 - Ondina",
    suspicion_threshold_time: int = 180,
) -> None:
    """
    Detecta objetos que permanecem no ambiente por mais tempo que o limite definido.

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
        camera_location (str): O local da câmera onde a detecção está sendo realizada.
        suspicion_threshold_time (int): O tempo limite (em segundos) para considerar um objeto
            como suspeito.
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
    model_filename = YOLO11X_MODEL_DROPBOX_PATH.split("/")[-1]
    model_path = MODELS_FOLDER_PATH / model_filename

    try:
        download_model(model_path, YOLO11X_MODEL_DROPBOX_PATH, DROPBOX_ACCESS_TOKEN)
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
                    if total_time > suspicion_threshold_time:
                        suspects_ids.append(track_id)
                        label = f"{track_id}: {round(total_time,2)}s (suspect)"
                        color = (0, 0, 255)  # Red for suspicious persons
                    else:
                        label = f"{track_id}: {round(total_time,2)}s"
                else:
                    # For vehicles, display the confidence score
                    label = f"vehicle: {confidence:.2f}"

                plot_bbox(
                    img=screenshot, class_id=int(cls), box_coordinates=box, label=label, color=color
                )

            if len(suspects_ids) > 0 and any(
                tracking_data.get(suspect_id, {}).get("alert_sent") is False
                for suspect_id in suspects_ids
            ):
                frame_path = save_annotated_image(screenshot, str(image_folder_path))

                save_results_to_json(
                    results=results,
                    file_path=str(output_json_path),
                    frame_path=frame_path,
                    model_name=model_filename,
                    classes_names=classes_names,
                    camera_location=camera_location,
                    suspect_ids=suspects_ids,
                    tracking_data=tracking_data,
                )
                for suspect_id in suspects_ids:
                    tracking_data.get(suspect_id, {})["alert_sent"] = True

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


def detect_proximity_to_vehicle(
    window_title: str | None = None,
    output_json_path: Path = DATA_FOLDER_PATH / "output.json",
    image_folder_path: Path = FRAMES_FOLDER_PATH,
    camera_location: str = "Portaria 1 - Ondina",
    suspicion_threshold_time: int = 180,
) -> None:
    """
    Detecta objetos que permanecem próximos de veículos além de tempo limite definido.

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
        camera_location (str): O local da câmera onde a detecção está sendo realizada.
        suspicion_threshold_time (int): O tempo limite (em segundos) para considerar um objeto
            como suspeito.
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
    model_filename = YOLO11X_MODEL_DROPBOX_PATH.split("/")[-1]
    model_path = MODELS_FOLDER_PATH / model_filename

    try:
        download_model(model_path, YOLO11X_MODEL_DROPBOX_PATH, DROPBOX_ACCESS_TOKEN)
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
        results = list(model.track(source=screenshot, classes=[0, 2, 3], persist=True, stream=True))

        if len(results[0].boxes) > 0 and any([box.id for box in results[0].boxes]):
            persons, vehicles, persons_near_vehicle = {}, {}, {}
            suspects_ids = []
            for box, track_id, cls in zip(
                results[0].boxes.xyxy.cpu(),
                results[0].boxes.id.int().cpu().tolist(),
                results[0].boxes.cls.int(),
            ):
                if cls == 0 and track_id is not None:
                    persons[track_id] = box

                elif cls in [2, 3] and track_id is not None:
                    vehicles[track_id] = box

            # Calulate if a person is near a vehicle
            for person_id, person_box in list(persons.items()):  # usa lista para iterar pela copia
                color: tuple[int, int, int] | None = None
                total_time_near_vehicle = tracking_data.get(person_id, {}).get(
                    "total_time_near_vehicle", 0
                )
                label = f"{person_id}: {round(total_time_near_vehicle, 2)}s"
                for vehicle_id, vehicle_box in vehicles.items():
                    if calculate_bbox_iou(vehicle_box, person_box) > 0:
                        if total_time_near_vehicle > suspicion_threshold_time:
                            suspects_ids.append(person_id)
                            color = (0, 0, 255)
                            label = f"{person_id}: {round(total_time_near_vehicle,2)}s (suspect)"
                        persons_near_vehicle[person_id] = person_box
                        persons.pop(person_id)
                        break
                plot_bbox(
                    img=screenshot, class_id=0, box_coordinates=person_box, label=label, color=color
                )

            # Plota veículos
            for vehicle_id, vehicle_box in vehicles.items():
                label = "vehicle"
                plot_bbox(
                    img=screenshot,
                    class_id=2,
                    box_coordinates=vehicle_box,
                    label=label,
                )

            # Update tracked objects
            update_tracked_objects_proximity_to_vehicle(
                tracked_ids_no_vehicle_near=list(persons.keys()),
                tracked_ids_vehicle_near=list(persons_near_vehicle.keys()),
                current_time=loop_time,
                tracking_data=tracking_data,
            )

            if len(suspects_ids) > 0 and any(
                tracking_data.get(suspect_id, {}).get("alert_sent") is False
                for suspect_id in suspects_ids
            ):
                frame_path = save_annotated_image(screenshot, str(image_folder_path))

                save_results_to_json(
                    results=results,
                    file_path=str(output_json_path),
                    frame_path=frame_path,
                    model_name=model_filename,
                    classes_names=classes_names,
                    camera_location=camera_location,
                    suspect_ids=suspects_ids,
                    tracking_data=tracking_data,
                )
                for suspect_id in suspects_ids:
                    tracking_data.get(suspect_id, {})["alert_sent"] = True

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


def detect_proximity_with_pose(
    window_title: str | None = None,
    output_json_path: Path = DATA_FOLDER_PATH / "output.json",
    image_folder_path: Path = FRAMES_FOLDER_PATH,
    camera_location: str = "Portaria 1 - Ondina",
    suspicion_threshold_standing: int = 180,
    suspicion_threshold_crouched: int = 60,
) -> None:
    """
    Detecta proximidade de veículos combinada com análise de pose.

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
        camera_location (str): O local da câmera onde a detecção está sendo realizada.
        suspicion_threshold_standing (int): O tempo limite (em segundos) para considerar um objeto
            como suspeito quando está em pé.
        suspicion_threshold_crouched (int): O tempo limite (em segundos) para considerar um objeto
            como suspeito quando está agachado.
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
    yolo11x_model_filename = YOLO11X_MODEL_DROPBOX_PATH.split("/")[-1]
    yolo11x_model_path = MODELS_FOLDER_PATH / yolo11x_model_filename
    pose_model_filename = YOLO11X_POSE_MODEL_DROPBOX_PATH.split("/")[-1]
    pose_model_path = MODELS_FOLDER_PATH / pose_model_filename

    try:
        download_model(yolo11x_model_path, YOLO11X_MODEL_DROPBOX_PATH, DROPBOX_ACCESS_TOKEN)
        download_model(pose_model_path, YOLO11X_POSE_MODEL_DROPBOX_PATH, DROPBOX_ACCESS_TOKEN)
    except NoModelAvailableException as e:
        logger.error(f"[SuspiciousBehaviorDetection] {e} Detection cannot be performed.")
        return

    yolo11x_model = initialize_yolo_model(yolo11x_model_path)
    pose_model = initialize_yolo_model(pose_model_path)

    # Obtem o nome das classes
    pose_classes_names = pose_model.names

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
        results = list(
            yolo11x_model.track(source=screenshot, classes=[0, 2, 3], persist=True, stream=True)
        )

        if len(results[0].boxes) > 0 and any([box.id for box in results[0].boxes]):
            persons, vehicles, persons_near_vehicle = {}, {}, {}
            suspects_ids = []
            for box, track_id, cls in zip(
                results[0].boxes.xyxy.cpu(),
                results[0].boxes.id.int().cpu().tolist(),
                results[0].boxes.cls.int(),
            ):
                if cls == 0 and track_id is not None:
                    persons[track_id] = box

                elif cls in [2, 3] and track_id is not None:
                    vehicles[track_id] = box

            # Testa ocorreu interseccao entre pessoa e veiculo para rodar o outro modelo
            intersection = False

            # Calulate if a person is near a vehicle
            for person_id, person_box in list(persons.items()):  # usa lista para iterar pela copia
                for vehicle_id, vehicle_box in vehicles.items():
                    if calculate_bbox_iou(vehicle_box, person_box) > 0:
                        intersection = True
                        persons_near_vehicle[person_id] = person_box
                        persons.pop(person_id)
                        break

            color: tuple[int, int, int] | None = None

            # Plota pessoas longe de veículos
            for person_id, person_box in persons.items():
                total_time_near_vehicle = tracking_data.get(person_id, {}).get(
                    "total_time_near_vehicle", 0
                )
                label = f"{person_id}: {round(total_time_near_vehicle, 2)}s"
                plot_bbox(
                    img=screenshot, class_id=0, box_coordinates=person_box, label=label, color=color
                )

            # Plota veículos
            for vehicle_id, vehicle_box in vehicles.items():
                label = "vehicle"
                plot_bbox(
                    img=screenshot,
                    class_id=2,
                    box_coordinates=vehicle_box,
                    label=label,
                    color=color,
                )

            # Update tracked objects
            update_tracked_objects_proximity_to_vehicle(
                tracked_ids_no_vehicle_near=list(persons.keys()),
                tracked_ids_vehicle_near=list(persons_near_vehicle.keys()),
                current_time=loop_time,
                tracking_data=tracking_data,
            )

            # Executa a inferencia do YOLOv11 de pontos-chave e verifica se a pessoa está agachada
            if intersection:
                pose_results = list(pose_model(source=screenshot, stream=True))
                if len(pose_results[0].boxes) > 0:
                    pose_boxes = pose_results[0].boxes.xyxy.cpu().numpy()
                    pose_keypoints = pose_results[0].keypoints.xy.cpu().numpy()

                    # Calcula centros dos bounding boxes
                    person_centers = {
                        track_id: ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                        for track_id, box in persons_near_vehicle.items()
                    }
                    pose_centers = [
                        ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in pose_boxes
                    ]

                    # Criação da matriz de custo
                    cost_matrix = np.zeros((len(person_centers), len(pose_centers)))
                    person_ids = list(person_centers.keys())
                    for i, person_center in enumerate(person_centers.values()):
                        for j, pose_center in enumerate(pose_centers):
                            cost_matrix[i, j] = np.linalg.norm(
                                np.array(person_center) - np.array(pose_center)
                            )

                    # Resolve correspondência via Hungarian
                    row_ind, col_ind = linear_sum_assignment(cost_matrix)

                    # Armazena correspondências válidas
                    matched_pose_indices = set()
                    valid_matches = []
                    for row, col in zip(row_ind, col_ind):
                        track_id = person_ids[row]
                        pose_box = pose_boxes[col]
                        person_box = persons_near_vehicle[track_id]

                        # Verifique o IoU entre as bounding boxes
                        iou = calculate_bbox_iou(person_box, pose_box)
                        if iou >= 0.2:  # Apenas correspondências com IoU >= 0.5 são aceitas
                            valid_matches.append((track_id, col))
                            matched_pose_indices.add(col)  # Marca a pose como correspondente

                    # Processa cada pessoa detectada no primeiro modelo
                    for track_id, person_box in persons_near_vehicle.items():
                        # Verifica se o track_id tem correspondência válida
                        matched_pose = next(
                            (col for t_id, col in valid_matches if t_id == track_id), None
                        )

                        if matched_pose is not None:  # Se houve correspondência
                            squat = is_squat(pose_keypoints[matched_pose])
                        else:  # Sem correspondência, assume-se "em pé"
                            squat = False

                        # Atualiza tracking_data
                        if not tracking_data[track_id].get("crouched", False) and squat:
                            tracking_data[track_id]["crouched"] = True

                        total_time_near_vehicle = tracking_data.get(track_id, {}).get(
                            "total_time_near_vehicle", 0
                        )

                        # Define o estado do indivíduo
                        if total_time_near_vehicle > (
                            suspicion_threshold_crouched if squat else suspicion_threshold_standing
                        ):
                            state = PoseStateEnum.SUSPECT
                            suspects_ids.append(track_id)
                        else:
                            state = PoseStateEnum.SQUATTING if squat else PoseStateEnum.STANDING

                        # Visualiza os resultados
                        if matched_pose is not None:
                            plot_keypoints_detection(
                                frame=screenshot,
                                kpts=pose_keypoints[matched_pose],
                                kpts_conf=pose_results[0].keypoints.conf[matched_pose],
                                box=person_box,
                                state=state,
                                person_id=track_id,
                                time_near_vehicle=total_time_near_vehicle,
                            )
                        else:
                            plot_bbox(
                                img=screenshot,
                                class_id=0,
                                box_coordinates=person_box,
                                label=f"{track_id}: {round(total_time_near_vehicle, 2)}s",
                            )

                if len(suspects_ids) > 0 and any(
                    tracking_data.get(suspect_id, {}).get("alert_sent") is False
                    for suspect_id in suspects_ids
                ):
                    frame_path = save_annotated_image(screenshot, str(image_folder_path))

                    save_results_to_json(
                        results=results,
                        file_path=str(output_json_path),
                        frame_path=frame_path,
                        model_name=pose_model_filename,
                        classes_names=pose_classes_names,
                        camera_location=camera_location,
                        suspect_ids=suspects_ids,
                        tracking_data=tracking_data,
                    )
                    for suspect_id in suspects_ids:
                        tracking_data.get(suspect_id, {})["alert_sent"] = True

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


if __name__ == "__main__":
    #detect_proximity_with_pose(
    #    window_title="Genetec™ Security Desk",
    #    suspicion_threshold_standing=10,
    #    suspicion_threshold_crouched=5,
    #)
    detect_proximity_to_vehicle(
       window_title="Genetec™ Security Desk",
       suspicion_threshold_time=10,
    )
