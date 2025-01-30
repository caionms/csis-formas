"""Módulo de utilitários para plotagem de caixas delimitadoras e rótulos em imagens."""

import cv2
import numpy as np
import torch

from domain.enums.pose_state_enum import PoseStateEnum


def colors(index: int, bgr: bool = True) -> tuple[int, int, int]:
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
    results: list,
    color: tuple[int, int, int] | None = None,
    label: str | None = None,
    line_thickness: int = 3,
) -> np.ndarray:
    """
    Plots bounding boxes and labels on an image.

    Args:
        img (np.ndarray): The image on which to plot the bounding boxes.
        results (List): Detection results containing bounding boxes and labels.
        color (Optional[Tuple[int, int, int]]): Color for the bounding boxes. If None, color is
        generated based on class ID.
        label (Optional[str]): Text label to display on the bounding boxes. If None, labels are
        generated based on class names and confidence.
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
            cv2.rectangle(
                img, top_left, bottom_right, box_color, thickness=tl, lineType=cv2.LINE_AA
            )

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
                cv2.rectangle(
                    img,
                    text_bottom_left,
                    text_top_right,
                    box_color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

                # Plota o texto definido
                cv2.putText(
                    img,
                    label_text,
                    (left, top - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    tl / 3,
                    (225, 255, 255),
                    tf,
                    cv2.LINE_AA,
                )

    return img


def plot_bbox(
    img: np.ndarray,
    class_id: int,
    box_coordinates: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int] | None = None,
    line_thickness: int = 3,
) -> np.ndarray:
    """
    Plots bounding box and label on an image.

    Args:
        img (np.ndarray): The image on which to plot the bounding boxes.
        class_id (int): Class ID for the bounding box.
        box_coordinates (Tuple[int, int, int, int]): Bounding box coordinates in
        (left, top, right, bottom) format.
        label (Optional[str]): Text label to display on the bounding box.
        color (Optional[Tuple[int, int, int]]): Color for the bounding boxes. If None, color
        is generated based on class ID.
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
    cv2.rectangle(
        img=img, pt1=top_left, pt2=bottom_right, color=box_color, thickness=tl, lineType=cv2.LINE_AA
    )

    # Set font thickness and size for the label text
    tf = max(tl - 1, 1)
    text_scale = tl / 4

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, text_scale, tf)[0]

    # Coordinates for the background rectangle behind the text
    text_bottom_left = (left, top - text_size[1] - 3)
    text_top_right = (left + text_size[0], top)

    # Draw the rectangle for the label background
    cv2.rectangle(
        img=img,
        pt1=text_bottom_left,
        pt2=text_top_right,
        color=box_color,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # Draw the label text
    cv2.putText(
        img,
        label,
        (left, top - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        (225, 255, 255),
        tf,
        cv2.LINE_AA,
    )

    return img


def plot_skeleton_kpts(
    frame: np.ndarray,
    kpts: list[tuple[float, float]],
    kpts_conf: list[float],
    color: tuple[int, int, int] | None = None,
    orig_shape: tuple[int, int] | None = None,
) -> None:
    """
    Plota o esqueleto humano e pontos-chave em uma imagem.

    Args:
        frame (np.ndarray): Frame onde o esqueleto será plotado.
        kpts (List[Tuple[float, float]]): Lista de coordenadas (x, y) dos pontos-chave.
        kpts_conf (List[float]): Confianças associadas a cada ponto-chave.
        color (Optional[Tuple[int, int, int]]): Cor em formato RGB. Padrão é None.
        orig_shape (Optional[Tuple[int, int]]): Forma original da imagem. Padrão é None.
    """
    skeleton = [
        [1, 2],
        [1, 3],
        [2, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [9, 11],
        [7, 13],
        [6, 12],
        [12, 13],
        [14, 12],
        [15, 13],
        [16, 14],
        [17, 15],
    ]

    radius = 5

    r, g, b = color if color else (0, 255, 0)  # Cor padrão verde

    for kid, (x_coord, y_coord) in enumerate(kpts):
        if kpts_conf[kid] >= 0.1 and 0 < x_coord < 640 and 0 < y_coord < 640:
            cv2.circle(frame, (int(x_coord), int(y_coord)), radius, (r, g, b), -1)

    for sk_id, (p1, p2) in enumerate(skeleton):
        x1, y1 = kpts[p1 - 1]
        x2, y2 = kpts[p2 - 1]
        conf1, conf2 = kpts_conf[p1 - 1], kpts_conf[p2 - 1]

        if all(
            [conf1 >= 0.1, conf2 >= 0.1, 0 < x1 < 640, 0 < y1 < 640, 0 < x2 < 640, 0 < y2 < 640]
        ):
            pos1, pos2 = (int(x1), int(y1)), (int(x2), int(y2))
            cv2.line(frame, pos1, pos2, (r, g, b), thickness=2)


def plot_keypoints_detection(
    frame: np.ndarray,
    kpts: list[tuple[float, float]],
    kpts_conf: list[float],
    box: torch.Tensor,
    state: PoseStateEnum,
    person_id: int,
    time_near_vehicle: float,
    orig_shape: tuple[int, int] | None = None,
) -> None:
    """
    Plota a detecção de pontos-chave e esqueleto em um frame, incluindo o estado da pessoa.

    Args:
        frame (np.ndarray): Frame onde o esqueleto e os textos serão plotados.
        kpts (List[Tuple[float, float]]): Coordenadas dos pontos-chave.
        kpts_conf (List[float]): Confiança dos pontos-chave.
        box (torch.Tensor): Coordenadas da caixa delimitadora.
        state (PoseStateEnum): Estado da pessoa (em pé, agachado, suspeito).
        person_id (int): ID da pessoa rastreada.
        time_near_vehicle (float): Tempo que a pessoa passou perto de um veículo.
        orig_shape (Optional[Tuple[int, int]]): Forma original da imagem, se aplicável.
    """
    # Define o texto e a cor com base no estado da pessoa
    state_labels = {
        PoseStateEnum.STANDING: ("Em pé proximo a um veiculo", (0, 215, 255)),
        PoseStateEnum.SQUATTING: ("Agachado(a) proximo a um veiculo", (0, 95, 255)),
        PoseStateEnum.SUSPECT: ("Suspeito(a)", (0, 0, 255)),
    }
    label, color = state_labels[state]
    label = f"{person_id}: {label} ({round(time_near_vehicle, 2)}s)"

    # Plota o esqueleto com as cores definidas
    plot_skeleton_kpts(frame, kpts, kpts_conf, color, orig_shape)

    # Define as cores e espessuras
    r, g, b = color
    x1, y1, x2, y2 = map(lambda v: int(v.item()), box)
    line_thickness = round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1
    font_thickness = max(line_thickness - 1, 1)

    # Calcula o tamanho do texto
    text_size = cv2.getTextSize(label, 0, fontScale=line_thickness / 3.7, thickness=font_thickness)[
        0
    ]
    text_width, text_height = text_size

    # Define as coordenadas para o retângulo do texto
    text_rect_bottom_right = (x1 + text_width, y1 - text_height - 3)

    # Plota a caixa delimitadora
    cv2.rectangle(frame, (x1, y1), (x2, y2), (r, g, b), 2)

    # Plota o retângulo de fundo do texto
    cv2.rectangle(frame, (x1, y1), text_rect_bottom_right, (r, g, b), -1, cv2.LINE_AA)

    # Adiciona o texto no frame
    cv2.putText(
        frame,
        label,
        (x1, y1 - 2),
        0,
        line_thickness / 3.7,
        [255, 255, 255],
        font_thickness,
        cv2.LINE_AA,
    )


def plot_only_label(img, box, text, font_scale=0.7, color=(0, 255, 0), thickness=2):
    """
    Desenha um texto próximo a um bounding box em uma imagem.

    Args:
        img (numpy.ndarray): A imagem em que o bounding box e o texto serão desenhados.
        box (list or tuple): Coordenadas do bounding box [x1, y1, x2, y2].
        text (str): O texto a ser desenhado próximo ao bounding box.
        font_scale (float): Escala da fonte do texto.
        color (tuple): Cor do texto e do bounding box (formato BGR).
        thickness (int): Espessura do texto e do bounding box.

    Returns:
        numpy.ndarray: A imagem com o bounding box e o texto desenhados.
    """
    # Coordenadas do bounding box
    x1, y1, x2, y2 = map(int, box)

    # Calcular a posição do texto
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = x1
    text_y = y2 + text_size[1] + 5  # Ajusta para colocar abaixo do bounding box

    # Desenhar o texto
    cv2.putText(img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    return img
