from typing import List, Optional, Tuple

import cv2
import numpy as np


def colors(index: int, bgr: bool = True) -> Tuple[int, int, int]:
    """
    Generates a color based on the given index, avoiding red tones.

    Args:
        index (int): Index for generating a color.
        bgr (bool): Flag to return the color in BGR format. Default is True.

    Returns:
        Tuple[int, int, int]: Generated color in BGR or RGB format.
    """
    # Palette of 6 different colors excluding red tones
    palette = [
        (0, 100, 100),  # Dark Cyan
        (100, 100, 0),  # Dark Yellow
        (0, 100, 0),  # Dark Green
        (100, 0, 100),  # Dark Magenta
        (0, 50, 100),  # Dark Orange-like (muted blueish)
        (50, 0, 100),  # Dark Purple
    ]

    # Select color from the palette based on index
    color = palette[index % len(palette)]

    # Return BGR format (OpenCV standard) or RGB format
    return color if bgr else color[::-1]


def plot_bboxes(
        img: np.ndarray,
        results: List,
        color: Optional[Tuple[int, int, int]] = None,
        label: Optional[str] = None,
        line_thickness: int = 3
) -> np.ndarray:
    """
    Plots bounding boxes and labels on an image.

    Args:
        img (np.ndarray): The image on which to plot the bounding boxes.
        results (List): Detection results containing bounding boxes and labels.
        color (Optional[Tuple[int, int, int]]): Color for the bounding boxes. If None, color is generated based on class ID.
        label (Optional[str]): Text label to display on the bounding boxes. If None, labels are generated based on class names and confidence.
        line_thickness (int): Thickness of the bounding box lines.

    Returns:
        np.ndarray: The image with plotted bounding boxes and labels.
    """
    for result in results:
        for box in result.boxes:
            coordinates = box.xyxy[0].numpy()
            left, top, right, bottom = map(int, coordinates)

            confidence = float(box.conf.cpu())
            class_id = int(box.cls)
            label_text = label or f"{result.names[class_id]}: {confidence:.2f}"

            # Define espessura da linha
            tl = line_thickness or max(1, round(0.002 * (img.shape[0] + img.shape[1]) / 2))

            # Define cor para as caixas
            box_color = color or colors(class_id, True)

            # Define coordenadas da caixa delimitadora
            top_left, bottom_right = (left, top), (right, bottom)

            # Plota a caixa delimitadora
            cv2.rectangle(img, top_left, bottom_right, box_color, thickness=tl, lineType=cv2.LINE_AA)

            # Se existir um label, plota ele
            if label_text:
                # Define espessura da fonte
                tf = max(tl - 1, 1)

                # Extrai o tamanho do texto
                text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, tl / 3, tf)[0]

                # Define nova coordenada para o retângulo do texto
                text_bottom_left = (left, top - text_size[1] - 3)
                text_top_right = (left + text_size[0], top)

                # Plota o retângulo do texto
                cv2.rectangle(img, text_bottom_left, text_top_right, box_color, thickness=-1, lineType=cv2.LINE_AA)

                # Plota o texto definido
                cv2.putText(img, label_text, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, tl / 3, (225, 255, 255), tf,
                            cv2.LINE_AA)

    return img


def plot_bbox(
        img: np.ndarray,
        class_id: int,
        box_coordinates: Tuple[int, int, int, int],
        label: str,
        color: Optional[Tuple[int, int, int]] = None,
        line_thickness: int = 3
) -> np.ndarray:
    """
    Plots bounding box and label on an image.

    Args:
        img (np.ndarray): The image on which to plot the bounding boxes.
        class_id (int): Class ID for the bounding box.
        box_coordinates (Tuple[int, int, int, int]): Bounding box coordinates in (left, top, right, bottom) format.
        label (Optional[str]): Text label to display on the bounding box.
        color (Optional[Tuple[int, int, int]]): Color for the bounding boxes. If None, color is generated based on class ID.
        line_thickness (int): Thickness of the bounding box lines.

    Returns:
        np.ndarray: The image with plotted bounding boxes and labels.
    """
    left, top, right, bottom = map(int, box_coordinates)

    # Set line thickness based on image size if not provided
    tl = line_thickness or max(1, round(0.001 * (img.shape[0] + img.shape[1]) / 2))

    # Set color for the box, defaulting to one based on class_id
    box_color = color or colors(class_id, True)

    # Define the top-left and bottom-right coordinates for the bounding box
    top_left = (left, top)
    bottom_right = (right, bottom)

    # Draw the bounding box
    cv2.rectangle(img, top_left, bottom_right, box_color, thickness=tl, lineType=cv2.LINE_AA)

    # Set font thickness and size for the label text
    tf = max(tl - 1, 1)
    text_scale = tl / 4

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, tf)[0]

    # Coordinates for the background rectangle behind the text
    text_bottom_left = (left, top - text_size[1] - 3)
    text_top_right = (left + text_size[0], top)

    # Draw the rectangle for the label background
    cv2.rectangle(img, text_bottom_left, text_top_right, box_color, thickness=-1, lineType=cv2.LINE_AA)

    # Draw the label text
    cv2.putText(img, label, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (225, 255, 255), tf,
                cv2.LINE_AA)

    return img
