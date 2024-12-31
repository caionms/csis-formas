"""
Módulo com funções utilitárias para detecção de comportamento suspeito.
"""


def bbox_iou_vehicle(box1, box2):
    # Extrai as coordenadas das caixas delimitadoras
    x1_box1, y1_box1, x2_box1, y2_box1 = box1
    x1_box2, y1_box2, x2_box2, y2_box2 = box2

    # Coordenadas da intersecção
    x1_intersection = max(x1_box1, x1_box2)
    y1_intersection = max(y1_box1, y1_box2)
    x2_intersection = min(x2_box1, x2_box2)
    y2_intersection = min(y2_box1, y2_box2)

    # Área da intersecção
    intersection_area = max(0, x2_intersection - x1_intersection + 1) * max(
        0, y2_intersection - y1_intersection + 1
    )

    # Áreas das caixas delimitadoras
    box1_area = (x2_box1 - x1_box1 + 1) * (y2_box1 - y1_box1 + 1)
    box2_area = (x2_box2 - x1_box2 + 1) * (y2_box2 - y1_box2 + 1)

    # União das áreas
    union_area = box1_area + box2_area - intersection_area

    # Cálculo do IoU
    iou = intersection_area / union_area

    return iou
