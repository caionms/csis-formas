"""Testes para as funções de plotagem."""

import numpy as np
import pytest

from application.utils.plot_utils import colors, plot_bbox, plot_bboxes


class TestColors:
    """Testes para a função `colors`."""

    @pytest.mark.parametrize(
        "index, bgr, expected",
        [
            (0, True, (0, 100, 100)),  # Dark Cyan
            (1, True, (100, 100, 0)),  # Dark Yellow
            (6, True, (0, 100, 100)),  # Wrap-around palette
            (2, False, (0, 100, 0)),  # RGB
        ],
    )
    def test_valid_colors(self, index, bgr, expected):
        """Deve retornar as cores esperadas."""
        assert colors(index, bgr) == expected

    def test_default_bgr(self):
        """Deve retornar a cor em formato BGR por padrão."""
        assert colors(0) == (0, 100, 100)


class TestPlotBboxes:
    """Testes para a função `plot_bboxes`."""

    @pytest.fixture
    def sample_image(self):
        """Gera uma imagem de exemplo para os testes."""
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_plot_bboxes_empty_results(self, sample_image):
        """Deve retornar a imagem original se não houver resultados."""
        output = plot_bboxes(sample_image.copy(), results=[])
        assert np.array_equal(output, sample_image)


class TestPlotBBox:
    """Testes para a função `plot_bbox`."""

    @pytest.fixture
    def sample_image(self):
        """Gera uma imagem de exemplo para os testes."""
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def test_plot_bbox_invalid_coordinates(self, sample_image):
        """Deve lançar um erro ao passar coordenadas inválidas."""
        with pytest.raises(ValueError):
            plot_bbox(sample_image.copy(), 0, (10,), "Label")
