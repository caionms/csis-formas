"""Enum que mapeia a entrada ou saída de veículos e o tipo de placa."""

from enum import Enum


class VehicleEnum(Enum):
    """Enum que mapeia a entrada ou saída de veículos."""

    IN = "IN"
    """Entrada de veículo."""
    OUT = "OUT"
    """Saída de veículo."""


class PlateType(Enum):
    """Enum que mapeia o tipo de placa."""

    OLD = "old"
    MERCOSUL = "mercosul"
