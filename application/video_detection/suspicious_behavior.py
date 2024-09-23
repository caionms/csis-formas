import importlib.resources as pkg_resources
import os
from pathlib import Path
from time import time
from typing import Any, Optional, Tuple

import cv2 as cv
import yaml
from ultralytics import YOLO

from application.dropbox_manager import DropboxManager
from application.log_config import get_logger
from application.utils.plot_utils import plot_bbox
from config import settings

YOLOV8X_MODEL_DROPBOX_PATH = settings.dropbox.models.yolov8x
DROPBOX_ACCESS_TOKEN = settings.dropbox.access_token

logger = get_logger(__name__)

TrackingData = dict[int, dict[str, Any]]


def update_tracking(tracker_results: list[Any], current_frame: int, tracking_data: TrackingData,
                    track_buffer: int) -> None:
    """Updates the tracking data with the current frame results.

    Args:
        tracker_results (list[Any]): The list of tracker results for the current frame.
        current_frame (int): The current frame number.
        tracking_data (Dict[int, Dict[str, Any]]): The dictionary holding tracking information for each ID.
        track_buffer (int): The number of frames a person can be absent before being forgotten.

    """

    # Update tracking data for detected persons in the current frame
    for track_id in tracker_results:
        #track_id = result.track_id

        # Check if the person has already been detected
        if track_id in tracking_data:
            # Update the last seen time and total presence frames
            tracking_data[track_id]['last_seen'] = current_frame
            tracking_data[track_id]['total_present_frames'] += 1
            tracking_data[track_id]['absent_frames'] = 0  # Reset the absence counter
        else:
            # Initialize tracking for a newly detected person
            tracking_data[track_id] = {
                'first_seen': current_frame,
                'last_seen': current_frame,
                'total_present_frames': 1,
                'absent_frames': 0
            }

    # Handle persons not detected in the current frame
    for track_id, data in list(tracking_data.items()):
        if data['last_seen'] != current_frame:
            # Increment the absence counter
            data['absent_frames'] += 1

            # Forget the person if absent for more than the track_buffer
            if data['absent_frames'] > track_buffer:
                print(f"Person with ID {track_id} was forgotten after {data['absent_frames']} frames of absence.")
                del tracking_data[track_id]  # Remove the ID from tracking
        else:
            # Print an alert if the person has been present for more than track_buffer frames
            if data['total_present_frames'] > track_buffer:
                logger.info(
                    f"[SuspiciousBehaviorDetection][update_tracking] Alert! Person with ID {track_id} has been in the video for {data['total_present_frames']} frames.")


def main(video_path: str, supicious_frame_limit: int = 60, save_video: bool = False, show_video: bool = True) -> None:
    """
    Roda a detecção de comportamento suspeito por permanência em um vídeo.

    Args:
        video_path (str): O caminho até o vídeo que o modelo irá rodar
        supicious_frame_limit (int): O limite de frames antes de uma pessoa ser considerada suspeita
        save_video (bool): Flag para sabe se deve salvar o vídeo ou não
        show_video (bool): Flag para saber se deve exibir o vídeo

    Raises:
        FileNotFoundError: Se o arquivo do modelo não for encontrado.
    """
    # Change the working directory to the folder this script is in.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Extract the video file name
    video_name = Path(video_path).name

    # Open the video
    cap = cv.VideoCapture(video_path)

    if cap is None:
        logger.error(f"[SuspiciousBehaviorDetection] Could not open video file: {video_path}")
        return

    # Get the frame rate to calculate time
    fps = cap.get(cv.CAP_PROP_FPS) or 30.0

    # Get the max_time_lost to calculate the time that a track is lost
    # Access the YAML file inside the `ultralytics.cfg` package
    with pkg_resources.open_text('ultralytics.cfg.trackers', 'botsort.yaml') as file:
        config = yaml.safe_load(file)

    # Get the value of 'track_buffer'
    track_buffer = config.get('track_buffer')
    logger.info(f"[SuspiciousBehaviorDetection] The value of track_buffer is: {track_buffer}")

    # The maximum number of frames that a track can be lost
    max_time_lost = int(fps / 30.0 * track_buffer)

    # Configure video saving if necessary
    if save_video:
        output_file = f"output/{Path(video_name).stem}_output{Path(video_name).suffix}"
        fourcc = cv.VideoWriter_fourcc(*'MP4V')
        out = cv.VideoWriter(output_file, fourcc, fps,
                             (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

    # Load the model
    model_filename = YOLOV8X_MODEL_DROPBOX_PATH.split("/")[-1]
    model_path = Path(__file__).parents[1] / "models" / model_filename
    if not model_path.is_file():
        dropbox_manager = DropboxManager(access_token=DROPBOX_ACCESS_TOKEN)
        if not dropbox_manager.download(
                dropbox_path=YOLOV8X_MODEL_DROPBOX_PATH,
                local_file_path=str(model_path),
        ):
            logger.error(
                "[SuspiciousBehaviorDetection] No model available and unable to download the model from Dropbox. "
                "Detection cannot be performed.")
            return
    model = YOLO(model_path)

    # Dictionary to store tracking data (presence and absence) by ID
    tracking_data = TrackingData()

    frame_counter = 0
    loop_time = time()
    # Iterate over the video frames
    while cap.isOpened():
        frame_counter += 1

        # Read the current frame
        success, frame = cap.read()

        # Check if the read was successful and the frame is not None
        if not success or frame is None:
            break

        # Perform YOLO inference
        results = list(model.track(source=frame, classes=[0, 2, 3], persist=True, stream=True))

        if len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int()
            confidences = results[0].boxes.conf.tolist()

            # Update tracking data of humans with the current frame results
            update_tracking(
                tracker_results=[track_id for i, track_id in enumerate(track_ids) if classes[i] == 0],
                current_frame=frame_counter,
                tracking_data=tracking_data,
                track_buffer=max_time_lost
            )

            for box, track_id, cls, confidence in zip(boxes, track_ids, classes, confidences):
                color: Optional[Tuple[int, int, int]] = None
                if cls == 0:  # Class 0 indicates a person
                    total_frames = tracking_data.get(track_id, {}).get('total_present_frames', 0)

                    # Check if the person has been present for more than the suspicious frame limit
                    if total_frames > supicious_frame_limit:
                        label = f"{track_id}: {total_frames}f (suspect)"
                        color = (0, 0, 255)  # Red for suspicious persons
                    else:
                        label = f"{track_id}: {total_frames}f"
                else:
                    # For vehicles, display the confidence score
                    label = f"vehicle: {confidence:.2f}"

                # Plot the bounding box with the label and color
                plot_bbox(
                    img=frame,
                    class_id=cls,
                    box_coordinates=box,
                    label=label,
                    color=color
                )

        if show_video:
            cv.imshow("Suspicious Behavior Detection", frame)

        if save_video and out is not None:
            out.write(frame)

        logger.info(f'[SuspiciousBehaviorDetection] FPS: {1 / (time() - loop_time):.2f}')
        loop_time = time()

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_video:
        out.release()
    cv.destroyAllWindows()
    logger.info('[SuspiciousBehaviorDetection] Done.')


if __name__ == "__main__":
    # Para finalizar o programa, apertar Q depois de clicar na visualização do plot ou fechar a janela que está sendo detectada
    #path = Path(__file__).parents[2] / "data" / "public_safety" / "1.mp4"
    main("D:\\Documents\\TCC\\TIC\\2024-06\\EFGYarRUC2.mp4", supicious_frame_limit=5, save_video=False, show_video=True)
