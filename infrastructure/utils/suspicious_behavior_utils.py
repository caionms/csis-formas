"""
Módulo com funções utilitárias para detecção de comportamento suspeito.
"""

import math
from typing import Any

from domain import TrackingData
from infrastructure.logging.log_config import get_logger

logger = get_logger(__name__)


def calculate_bbox_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """
    Calcula o Intersection over Union (IoU) entre duas caixas delimitadoras.

    Args:
        box1 (Tuple[int, int, int, int]): Coordenadas (x1, y1, x2, y2) da primeira caixa.
        box2 (Tuple[int, int, int, int]): Coordenadas (x1, y1, x2, y2) da segunda caixa.

    Returns:
        float: Valor do IoU.
    """
    # Coordenadas da intersecção
    x1_intersection = max(box1[0], box2[0])
    y1_intersection = max(box1[1], box2[1])
    x2_intersection = min(box1[2], box2[2])
    y2_intersection = min(box1[3], box2[3])

    # Área da intersecção
    intersection_area = max(0, x2_intersection - x1_intersection) * max(
        0, y2_intersection - y1_intersection
    )

    # Áreas das caixas delimitadoras
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # União das áreas
    union_area = box1_area + box2_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def three_points_angle(kpts: list[tuple[float, float]], kpts_ind: list[int]) -> float:
    """
    Calcula o ângulo entre três pontos definidos.

    Args:
        kpts (List[Tuple[float, float]]): Lista de pontos-chave com coordenadas (x, y).
        kpts_ind (List[int]): Índices dos três pontos na lista de pontos-chave.

    Returns:
        float: Ângulo em graus entre os três pontos.
    """
    # Extrai os pontos para definir o ângulo (linha = k_1-k_2 | k_2-k_3)
    x1, y1 = kpts[kpts_ind[0]][0], kpts[kpts_ind[0]][1]
    x2, y2 = kpts[kpts_ind[1]][0], kpts[kpts_ind[1]][1]
    x3, y3 = kpts[kpts_ind[2]][0], kpts[kpts_ind[2]][1]

    # Calcula os vetores
    v1 = (x1 - x2, y1 - y2)
    v2 = (x3 - x2, y3 - y2)

    # Calcula o produto escalar
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]

    # Calcula as magnitudes dos vetores
    magnitude_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    magnitude_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    # Evita divisão por zero
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0  # Ângulo padrão para vetores degenerados

    # Calcula o angulo entre os vetores
    angle = math.acos(min(max(dot_product / (magnitude_v1 * magnitude_v2), -1.0), 1.0))
    return math.degrees(angle)


def is_front(kpts: list[tuple[float, float]]) -> bool:
    """
    Determina se o sujeito está de frente com base em pontos-chave.

    Args:
        kpts (List[Tuple[float, float]]): Lista de pontos-chave com coordenadas (x, y).

    Returns:
        bool: Verdadeiro se o sujeito está de frente, Falso caso contrário.
    """
    x_nose = kpts[0][0]
    coord_x_eyes = sorted([kpts[1][0], kpts[2][0]])
    coord_x_ears = sorted([kpts[3][0], kpts[4][0]])

    # Se o nariz não estiver entre os olhos ou entre os ouvidos, está de lado
    return (
        coord_x_eyes[0] <= x_nose <= coord_x_eyes[1]
        and coord_x_ears[0] <= x_nose <= coord_x_ears[1]
    )


def is_squat(kpts: list[tuple[float, float]]) -> bool:
    """
    Verifica se o sujeito está em posição de agachamento.

    Args:
        kpts (List[Tuple[float, float]]): Lista de pontos-chave com coordenadas (x, y).

    Returns:
        bool: Verdadeiro se o sujeito está agachado, Falso caso contrário.
    """
    # Indices dos pontos-chave dos joelhos
    left_knee_indices = [11, 13, 15]
    right_knee_indices = [12, 14, 16]

    # Calcula os angulos internos dos joelhos
    left_knee_angle = three_points_angle(kpts, left_knee_indices)
    right_knee_angle = three_points_angle(kpts, right_knee_indices)

    # Calcula a media dos angulos dos joelhos (externo e interno)
    avg_leg_angle_ext = (180 - left_knee_angle + 180 - right_knee_angle) / 2
    avg_leg_angle_int = (left_knee_angle + right_knee_angle) / 2

    # Condicao para definir agachamento (frontal e lateral)
    return (
        is_front(kpts)
        and left_knee_angle < 145
        and right_knee_angle < 145
        and avg_leg_angle_int < 130
    ) or (avg_leg_angle_ext > 80)


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


def update_tracked_objects_proximity_to_vehicle(
    tracked_ids_no_vehicle_near: list[int],
    tracked_ids_vehicle_near: list[int],
    current_time: float,
    tracking_data: TrackingData,
) -> None:
    """
    Atualiza o tempo de rastreamento dos objetos e controla o estado dos mesmos.

    Args:
        tracked_ids_no_vehicle_near (list[Any]): The list of tracker results ids that are not
        near a vehicle.
        tracked_ids_vehicle_near (list[Any]): The list of tracker results ids that are near a
         vehicle.
        current_time (float): Timestamp atual (tempo atual em segundos).
        tracking_data (Dict[int, Dict[str, Any]]): The dictionary holding tracking information
        for each ID.
    """
    for track_id in tracked_ids_no_vehicle_near:
        if track_id is None:
            continue  # Se não houver ID, o objeto não está sendo rastreado

        if track_id not in tracking_data:
            # Novo objeto sendo rastreado
            tracking_data[track_id] = {
                "start_time": current_time,  # Quando o objeto começou a ser rastreado
                "last_seen_time": current_time,  # Último tempo que o objeto foi visto
                "total_time_tracked": 0,  # Tempo total que o objeto foi rastreado
                "total_time_near_vehicle": 0,  # Tempo total que o objeto esteve perto de um veículo
                "near_vehicle": False,  # Diz se o objeto está perto de um veículo
                "crouched": False,  # Diz se o objeto está agachado
                "alert_sent": False,  # Diz se o alerta foi enviado
            }
        else:
            # Atualiza o último tempo que o objeto foi visto
            last_seen_time: float = tracking_data[track_id]["last_seen_time"]
            tracking_data[track_id]["total_time_tracked"] += current_time - last_seen_time
            tracking_data[track_id]["last_seen_time"] = current_time
            tracking_data[track_id]["near_vehicle"] = False

    for track_id in tracked_ids_vehicle_near:
        if track_id is None:
            continue

        if track_id not in tracking_data:
            # Novo objeto sendo rastreado
            tracking_data[track_id] = {
                "start_time": current_time,  # Quando o objeto começou a ser rastreado
                "last_seen_time": current_time,  # Último tempo que o objeto foi visto
                "total_time_tracked": 0,  # Tempo total que o objeto foi rastreado
                "total_time_near_vehicle": 0,  # Tempo total que o objeto esteve perto de um veículo
                "near_vehicle": True,  # Diz se o objeto está perto de um veículo
                "crouched": False,  # Diz se o objeto está agachado
                "alert_sent": False,  # Diz se o alerta foi enviado
            }
        else:
            # Atualiza o último tempo que o objeto foi visto
            last_seen_time: float = tracking_data[track_id]["last_seen_time"]  # type: ignore
            tracking_data[track_id]["total_time_tracked"] += current_time - last_seen_time
            tracking_data[track_id]["last_seen_time"] = current_time
            tracking_data[track_id]["near_vehicle"] = True
            tracking_data[track_id]["total_time_near_vehicle"] += current_time - last_seen_time


def remove_stale_tracks(
    tracking_data: dict[int, dict[str, Any]], current_time: float, expiration_time: int = 240
) -> None:
    """
    Remove objects from tracking_data that haven't been seen for the specified
    expiration time (in seconds).

    Args:
        tracking_data (Dict[int, Dict[str, Any]]): Dictionary of tracked objects.
        current_time (float): The current time (in seconds).
        expiration_time (int): The time (in seconds) after which an object is considered stale.
    """
    stale_ids = [
        track_id
        for track_id, data in tracking_data.items()
        if (current_time - data["last_seen_time"] > expiration_time)
    ]

    for stale_id in stale_ids:
        logger.info(
            f"Removing track ID {stale_id} from tracking data (inactive for "
            f"{expiration_time} seconds or alert sent)."
        )
        del tracking_data[stale_id]
