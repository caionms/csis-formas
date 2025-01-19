"""
Módulo de utilitários para integração com o dashboard.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Any

import cv2 as cv
import numpy as np
from ultralytics.engine.results import Results

from domain.enums.plate_enum import PlateType, VehicleEnum
from infrastructure.logging.log_config import get_logger

logger = get_logger(__name__)


def save_results_to_json(
    results: list[Results],
    file_path: str,
    classes_names: dict[int, str],
    model_name: str,
    camera_location: str,
    tracking_data: dict[int, dict[str, Any]] | None = None,
    frame_path: str | None = None,
    suspect_ids: list[str] | None = None,
    ignore_classes: list[int] | None = None,
    video_time: float | None = None,
) -> None:
    """
    Save YOLO inference results to a JSON file, adding a new entry with a timestamp.

    Args:
        results (List[Results]): The YOLO inference results.
        file_path (str): Path to the JSON file to save the results.
        classes_names (List[str]): List of class names corresponding to the model output classes.
        model_name (str): The name of the model used for inference.
        camera_location (str): The location of the camera that captured the frame.
        frame_path (Optional[str]): Path to the frame image file, if available.
        suspect_ids (Optional[List[str]]): List of suspect IDs detected in the frame.
        tracking_data (Optional[Dict[int, Dict[str, Any]]]): The tracking data for the detections.
        ignore_classes (Optional[List[int]]): List of classes to ignore in the results.
        video_time (Optional[float]): The time in seconds of the video where the detections occurred
    """
    # Use list comprehension to collect detections above a confidence threshold
    detections = [
        {
            "class": f"{classes_names[int(box.cls)]} ({int(box.cls)})"
            if classes_names[int(box.cls)] is not None
            else f"{int(box.cls)}",  # Class name and number
            "confidence": round(float(box.conf), 2),  # Detection confidence rounded to 2 decimals
            "bbox": box.xywh.tolist(),  # Bounding box coordinates (x, y, w, h)
        }
        for result in results
        for box in result.boxes
        # TODO: Aumentar esse valor depois do desenvolvimento
        if (
            box.conf > 0.2  # Filter out low-confidence detections
            and int(box.id)
            and (int(box.id) in suspect_ids if suspect_ids else True)
            and tracking_data.get(int(box.id), {})["alert_sent"] is False
            if tracking_data
            else True and (int(box.cls) not in ignore_classes if ignore_classes else True)
        )
    ]

    if not detections:
        return

    # Generate the filename based on video time or current timestamp
    if video_time is not None:
        timestamp = str(timedelta(seconds=video_time))
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare data with timestamp and optional frame path
    data = {
        "timestamp": timestamp,
        "detections": detections,
        "frame_path": frame_path,
        "model": model_name,
        "camera_location": camera_location,
    }

    # Read existing JSON file content or initialize an empty list
    file_data = _read_json_file(file_path)

    # Append new data to the list
    file_data.append(data)

    # Write updated data back to the JSON file
    _write_json_file(file_path, file_data)


def _read_json_file(file_path: str) -> list[dict[str, Any]]:
    """
    Reads and returns the content of a JSON file or initializes an empty list.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        List[Dict[str, Any]]: The existing data in the file or an empty list.
    """
    if os.path.exists(file_path):
        try:
            with open(file_path) as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        except (json.JSONDecodeError, OSError):
            # Return empty list if file is corrupted or unreadable
            return []
    return []


def _write_json_file(file_path: str, data: list[dict[str, Any]]) -> None:
    """
    Writes the given data to a JSON file.

    Args:
        file_path (str): Path to the JSON file.
        data (List[Dict[str, Any]]): The data to be written to the file.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def save_annotated_image(
    image: np.ndarray, folder_path: str, video_time: float | None = None
) -> str | None:
    """
    Save the annotated image with detections, naming it with the current timestamp.

    Args:
        image (np.ndarray): The annotated image to save.
        folder_path (str): Path to the folder where the image will be saved.
        video_time (Optional[float]): The time in seconds of the video where the detections occurred

    Returns:
        Optional[str]: The path to the saved image file, or None if an error occurred.
    """
    # Ensure the directory exists
    os.makedirs(folder_path, exist_ok=True)

    # Generate the filename based on video time or current timestamp
    if video_time is not None:
        timestamp = f"{video_time:.2f}".replace(".", "-")
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_name = f"detection_{timestamp}.png"
    file_path = os.path.join(folder_path, file_name)

    try:
        # Save the annotated image
        cv.imwrite(file_path, image)
        return file_path
    except Exception:
        logger.exception(f"Error saving annotated image: {file_path}")
        return None


def save_plate_results_to_json(
    file_path: str,
    type_of_camera: VehicleEnum,
    plate_text: str,
    plate_type: PlateType,
    camera_location: str = "Portaria 1 - Ondina",
    frame_path: str | None = None,
    video_time: float | None = None,
) -> None:
    """
    Save plate recognition results to a JSON file, adding a new entry with a timestamp.

    Args:
        file_path (str): Path to the JSON file to save the results.
        type_of_camera (VehicleEnum): The type of camera used for plate recognition
        plate_text (str): The recognized plate text.
        plate_type (PlateType): The type of plate detected.
        camera_location (str): The location of the camera that captured the frame.
        frame_path (Optional[str]): Path to the frame image file, if available.
        video_time (Optional[float]): The time in seconds of the video where the detections occurred
    """
    # Generate the filename based on video time or current timestamp
    if video_time is not None:
        timestamp = str(timedelta(seconds=video_time))
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Prepare data with timestamp and optional frame path
    data = {
        "timestamp": timestamp,
        "plate_text": plate_text,
        "in_or_out": type_of_camera.value,
        "plate_type": plate_type.value,
        "frame_path": frame_path,
        "camera_location": camera_location,
    }

    # Read existing JSON file content or initialize an empty list
    file_data = _read_json_file(file_path)

    # Append new data to the list
    file_data.append(data)

    # Write updated data back to the JSON file
    _write_json_file(file_path, file_data)
