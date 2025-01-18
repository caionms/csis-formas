"""Conjunto de testes para as funções de utilitários de modelos."""

import unittest
from pathlib import Path
from unittest.mock import patch

from config.dynaconf_settings import settings
from infrastructure.utils.model_utils import (
    NoModelAvailableException,
    download_model,
    download_models,
    download_paddle_folder_model,
    initialize_paddleocr_model,
    initialize_yolo_model,
)


class TestModelUtils(unittest.TestCase):
    """Conjunto de testes para as funções de utilitários de modelos."""

    @patch("application.utils.model_utils.DropboxManager")
    def test_download_model_success(self, MockDropboxManager):
        """Teste que verifica o download de um modelo único com sucesso."""
        mock_manager = MockDropboxManager.return_value
        mock_manager.download.return_value = True
        model_path = Path("/fake/path/model.pt")
        model_dropbox_path = "/dropbox/path/model.pt"
        access_token = "fake_token"

        with patch("pathlib.Path.is_file", return_value=False):
            download_model(model_path, model_dropbox_path, access_token)

        mock_manager.download.assert_called_once_with(
            dropbox_path=model_dropbox_path, local_file_path=str(model_path)
        )

    @patch("application.utils.model_utils.DropboxManager")
    def test_download_model_failure(self, MockDropboxManager):
        """Teste que verifica falha no download de um modelo único."""
        mock_manager = MockDropboxManager.return_value
        mock_manager.download.return_value = False
        model_path = Path("/fake/path/model.pt")
        model_dropbox_path = "/dropbox/path/model.pt"
        access_token = "fake_token"

        with patch("pathlib.Path.is_file", return_value=False):
            with self.assertRaises(NoModelAvailableException):
                download_model(model_path, model_dropbox_path, access_token)

        mock_manager.download.assert_called_once_with(
            dropbox_path=model_dropbox_path, local_file_path=str(model_path)
        )

    @patch("application.utils.model_utils.download_model")
    def test_download_models(self, mock_download_model):
        """Teste que verifica o download de múltiplos modelos."""
        models_folder_path = Path("/fake/path/models")
        models_dropbox_paths = {
            "model1.pt": "/dropbox/path/model1.pt",
            "model2.pt": "/dropbox/path/model2.pt",
        }
        access_token = "fake_token"

        download_models(models_folder_path, models_dropbox_paths, access_token)

        for model_name, dropbox_path in models_dropbox_paths.items():
            mock_download_model.assert_any_call(
                models_folder_path / model_name, dropbox_path, access_token
            )

    @patch("application.utils.model_utils.DropboxManager")
    def test_download_paddle_folder_model_success(self, MockDropboxManager):
        """Teste que verifica o download de uma pasta Paddle com sucesso."""
        mock_manager = MockDropboxManager.return_value
        mock_manager.download_folder.return_value = True
        model_path = Path("/fake/path/paddle_model")
        model_dropbox_path = "/dropbox/path/paddle_model"
        access_token = "fake_token"

        with patch("pathlib.Path.is_dir", return_value=False):
            download_paddle_folder_model(model_path, model_dropbox_path, access_token)

        mock_manager.download_folder.assert_called_once_with(
            dropbox_folder_path=model_dropbox_path, local_folder_path=str(model_path)
        )

    @patch("application.utils.model_utils.DropboxManager")
    def test_download_paddle_folder_model_failure(self, MockDropboxManager):
        """Teste que verifica falha no download de uma pasta Paddle."""
        mock_manager = MockDropboxManager.return_value
        mock_manager.download_folder.return_value = False
        model_path = Path("/fake/path/paddle_model")
        model_dropbox_path = "/dropbox/path/paddle_model"
        access_token = "fake_token"

        with patch("pathlib.Path.is_dir", return_value=False):
            with self.assertRaises(NoModelAvailableException):
                download_paddle_folder_model(model_path, model_dropbox_path, access_token)

        mock_manager.download_folder.assert_called_once_with(
            dropbox_folder_path=model_dropbox_path, local_folder_path=str(model_path)
        )

    @patch("application.utils.model_utils.torch.cuda.is_available", return_value=True)
    @patch("application.utils.model_utils.YOLO")
    def test_initialize_yolo_model(self, MockYOLO, mock_is_available):
        """Teste que verifica a inicialização do modelo YOLO."""
        mock_model = MockYOLO.return_value
        model_path = Path("/fake/path/model.pt")

        initialized_model = initialize_yolo_model(model_path)

        MockYOLO.assert_called_once_with(model_path)
        mock_model.to.assert_called_once_with("cuda")
        self.assertEqual(initialized_model, mock_model)

    @patch("application.utils.model_utils.PaddleOCR")
    def test_initialize_paddleocr_model(self, MockPaddleOCR):
        """Teste que verifica a inicialização do modelo PaddleOCR."""
        mock_ocr = MockPaddleOCR.return_value
        text_detection_model_path = Path("/fake/path/det")
        text_recognition_model_path = Path("/fake/path/rec")
        text_cls_model_path = Path("/fake/path/cls")

        initialized_model = initialize_paddleocr_model(
            text_detection_model_path,
            text_recognition_model_path,
            text_cls_model_path,
        )

        MockPaddleOCR.assert_called_once_with(
            det_model_dir=str(text_detection_model_path),
            rec_model_dir=str(text_recognition_model_path),
            cls_model_dir=str(text_cls_model_path),
            use_angle_cls=True,
            lang="en",
            rec_algorithm=settings.paddleocr.rec_algorithm,
            rec_image_shape=settings.paddleocr.rec_image_shape,
            rec_char_dict_path=str(
                Path(__file__).parents[1] / "application" / "utils" / "data" / "en_dict.txt"
            ),
            use_space_char=settings.paddleocr.use_space_char,
            use_gpu=settings.paddleocr.use_gpu,
        )
        self.assertEqual(initialized_model, mock_ocr)


if __name__ == "__main__":
    unittest.main()
