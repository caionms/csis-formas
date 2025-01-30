"""Módulo que contém a enumeração de estados de pose."""

from enum import Enum


class PoseStateEnum(Enum):
    """Enum que mapeia os estados de pose."""

    STANDING = "STANDING"
    """Em pé próximo à um veículo."""
    SQUATTING = "SQUATTING"
    """Agachado próximo à um veículo."""
    SUSPECT = "SUSPECT"
    """Comportamento suspeito próximo à um veículo."""
