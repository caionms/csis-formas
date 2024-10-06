"""
Módulo de utilitários para integração com o dashboard.
"""

import json
import os
from datetime import datetime
from typing import Any

import cv2 as cv
import numpy as np
from ultralytics.engine.results import Results

from application.log_config import get_logger

logger = get_logger(__name__)


def save_results_to_json(
    results: list[Results],
    file_path: str,
    classes_names: list[str],
    model_name: str,
    frame_path: str | None = None,
) -> None:
    """
    Save YOLO inference results to a JSON file, adding a new entry with a timestamp.

    Args:
        results (List[Results]): The YOLO inference results.
        file_path (str): Path to the JSON file to save the results.
        classes_names (List[str]): List of class names corresponding to the model output classes.
        model_name (str): The name of the model used for inference.
        frame_path (Optional[str]): Path to the frame image file, if available.
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
        if box.conf > 0.2  # Filter out low-confidence detections
    ]

    if not detections:
        return

    # Prepare data with timestamp and optional frame path
    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detections": detections,
        "frame_path": frame_path,
        "model": model_name,
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


def save_annotated_image(image: np.ndarray, folder_path: str) -> str | None:
    """
    Save the annotated image with detections, naming it with the current timestamp.

    Args:
        image (np.ndarray): The annotated image to save.
        folder_path (str): Path to the folder where the image will be saved.

    Returns:
        Optional[str]: The path to the saved image file, or None if an error occurred.
    """
    # Ensure the directory exists
    os.makedirs(folder_path, exist_ok=True)

    # Generate the filename with the current timestamp
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
