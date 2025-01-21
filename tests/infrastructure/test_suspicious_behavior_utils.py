"""Módulo com testes para as funções utilitárias de comportamento suspeito."""

from math import isclose

import pytest

from infrastructure.utils.suspicious_behavior_utils import (
    calculate_bbox_iou,
    is_front,
    is_squat,
    remove_stale_tracks,
    three_points_angle,
    update_tracked_objects,
    update_tracked_objects_proximity_to_vehicle,
)


@pytest.fixture
def sample_keypoints():
    """Fixture para fornecer pontos-chave de exemplo."""
    return [
        (0.5, 0.5),  # Nose
        (0.4, 0.4),
        (0.6, 0.4),  # Eyes
        (0.3, 0.3),
        (0.7, 0.3),  # Ears
        (0.4, 0.7),
        (0.6, 0.7),  # Shoulders
        (0.5, 1.0),  # Mid Hip
        (0.4, 1.3),
        (0.6, 1.3),  # Knees
        (0.4, 1.7),
        (0.6, 1.7),  # Ankles
    ]


@pytest.fixture
def sample_tracking_data():
    """Fixture para dados de rastreamento simulados."""
    return {
        1: {
            "start_time": 0.0,
            "last_seen_time": 100.0,
            "total_time_tracked": 100.0,
            "alert_sent": False,
        },
        2: {
            "start_time": 50.0,
            "last_seen_time": 100.0,
            "total_time_tracked": 50.0,
            "alert_sent": True,
        },
    }


class TestTrackingUtilities:
    """Conjunto de testes para funções utilitárias de rastreamento."""

    def test_calculate_bbox_iou(self):
        """Deve calcular o IoU corretamente para caixas delimitadoras."""
        box1 = (0, 0, 2, 2)
        box2 = (1, 1, 3, 3)
        result = calculate_bbox_iou(box1, box2)
        assert isclose(result, 1 / 7, rel_tol=1e-9), "O IoU calculado está incorreto."

    def test_calculate_bbox_iou_no_intersection(self):
        """Deve retornar 0.0 para caixas delimitadoras sem interseção."""
        box1 = (0, 0, 1, 1)
        box2 = (2, 2, 3, 3)
        result = calculate_bbox_iou(box1, box2)
        assert result == 0.0, "O IoU deve ser zero para caixas sem interseção."

    def test_three_points_angle(self, sample_keypoints):
        """Deve calcular o ângulo corretamente entre três pontos."""
        kpts_indices = [5, 7, 9]  # Alterado para representar pontos alinhados ou específicos
        result = three_points_angle(sample_keypoints, kpts_indices)
        expected_angle = 180.0  # Ajustar conforme o esperado para os pontos fornecidos
        assert isclose(result, expected_angle, rel_tol=1e-9), "O ângulo calculado está incorreto."

    def test_is_front(self, sample_keypoints):
        """Deve determinar corretamente se o sujeito está de frente."""
        result = is_front(sample_keypoints)
        assert result is True, "A função deve identificar o sujeito como estando de frente."

    def test_is_squat_not_squatting(self, sample_keypoints):
        """Deve retornar False quando o sujeito não está agachado."""
        # Pontos que não atendem a nenhuma condição de agachamento
        kpts = sample_keypoints + [
            (0.4, 2.5),
            (0.6, 2.5),  # Joelhos retos e altos
            (0.4, 3.0),
            (0.6, 3.0),  # Tornozelos muito distantes
            (0.4, 3.5),
            (0.6, 3.5),  # Pés ainda mais distantes
        ]
        result = is_squat(kpts)
        assert not result, "Deve retornar False para um sujeito não agachado."

    def test_is_squat_lateral_squat(self, sample_keypoints):
        """Deve retornar True quando o sujeito está agachado lateralmente."""
        # Pontos que atendem à condição lateral (avg_leg_angle_ext > 80)
        kpts = sample_keypoints + [
            (0.4, 1.0),
            (0.6, 1.0),  # Joelhos dobrados
            (0.4, 1.3),
            (0.6, 1.3),  # Tornozelos próximos
            (0.4, 1.5),
            (0.6, 1.5),  # Pés
        ]
        result = is_squat(kpts)
        assert result, "Deve retornar True para um sujeito agachado lateralmente."

    def test_is_squat_front_squat(self, sample_keypoints):
        """Deve retornar True quando o sujeito está agachado de frente."""
        # Pontos que atendem à condição frontal
        kpts = sample_keypoints + [
            (0.4, 1.3),
            (0.6, 1.3),  # Joelhos dobrados
            (0.4, 1.7),
            (0.6, 1.7),  # Tornozelos
            (0.4, 2.0),
            (0.6, 2.0),  # Pés
        ]
        result = is_squat(kpts)
        assert result, "Deve retornar True para um sujeito agachado de frente."

    def test_update_tracked_objects(self, sample_tracking_data):
        """Deve atualizar o tempo de rastreamento corretamente."""
        tracks_ids = [1, 3]
        update_tracked_objects(tracks_ids, 110.0, sample_tracking_data)

        assert (
            sample_tracking_data[1]["total_time_tracked"] == 110.0
        ), "O tempo de rastreamento do objeto 1 não foi atualizado corretamente."
        assert (
            3 in sample_tracking_data
        ), "Um novo objeto deve ser adicionado aos dados de rastreamento."

    def test_update_tracked_objects_proximity_to_vehicle(self, sample_tracking_data):
        """Deve atualizar corretamente a proximidade com o veículo."""
        no_vehicle = [1]
        near_vehicle = [3]
        update_tracked_objects_proximity_to_vehicle(
            no_vehicle, near_vehicle, 110.0, sample_tracking_data
        )

        assert sample_tracking_data[3][
            "near_vehicle"
        ], "O objeto 3 deve ser marcado como próximo a um veículo."
        assert not sample_tracking_data[1][
            "near_vehicle"
        ], "O objeto 1 não deve ser marcado como próximo a um veículo."

    def test_remove_stale_tracks(self, sample_tracking_data):
        """Deve remover objetos antigos com base no tempo de expiração."""
        remove_stale_tracks(sample_tracking_data, 300.0, expiration_time=150)

        assert 2 not in sample_tracking_data, "Objetos antigos devem ser removidos."
