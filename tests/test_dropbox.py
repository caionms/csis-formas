import unittest
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest

from application.dropbox_manager import DropboxManager
from config import settings


class DropboxManagerTest(unittest.TestCase):

    @patch('builtins.open', new_callable=mock_open)  # Mockando o método open
    @patch('dropbox.Dropbox')  # Mockando a classe Dropbox do SDK
    def test_download_model_from_dropbox(self, MockDropbox, mock_file):
        """Teste que verifica se o download do modelo do Dropbox funciona corretamente."""
        # Criando uma instância mockada do DropboxManager
        mock_client = MockDropbox.return_value
        mock_client.files_download.return_value = (None, Mock(content=b"fake_content"))

        manager = DropboxManager(access_token="fake_token")

        # Simulando download sem exceções
        result = manager.download(
            dropbox_path="/path/to/model.pt",
            local_file_path="/local/path/model.pt",
        )

        # Verificar se o método files_download foi chamado corretamente
        mock_client.files_download.assert_called_once_with(path="/path/to/model.pt")
        # Verificar se o arquivo foi aberto para escrita em modo binário
        mock_file.assert_called_once_with("/local/path/model.pt", "wb")

        self.assertTrue(result)

    @patch('builtins.open', new_callable=mock_open)
    @patch('dropbox.Dropbox')
    def test_download_model_from_dropbox_failure(self, MockDropbox, mock_file):
        """Teste que simula falha no download do modelo do Dropbox."""
        mock_client = MockDropbox.return_value
        mock_client.files_download.side_effect = Exception("Falha no download")

        manager = DropboxManager(access_token="fake_token")

        # Simulando falha no download
        result = manager.download(
            dropbox_path="/path/to/invalid_model.pt",
            local_file_path="/local/path/invalid_model.pt",
        )

        # Verificar se o método files_download foi chamado corretamente
        mock_client.files_download.assert_called_once_with(path="/path/to/invalid_model.pt")

        # Verificar que o arquivo não foi aberto, já que houve uma falha
        mock_file.assert_not_called()

        self.assertFalse(result)

    @patch('builtins.open', new_callable=mock_open)
    @patch('dropbox.Dropbox')
    def test_upload_model_to_dropbox(self, MockDropbox, mock_file):
        """Teste que verifica se o upload do modelo para o Dropbox funciona corretamente."""
        mock_client = MockDropbox.return_value

        manager = DropboxManager(access_token="fake_token")

        # Simulando upload sem exceções
        result = manager.upload(
            dropbox_path="/path/to/upload/model.pt",
            local_file_path="/local/path/model.pt",
        )

        # Verificar se o arquivo foi aberto para leitura em modo binário
        mock_file.assert_called_once_with("/local/path/model.pt", "rb")
        # Verificar se o método files_upload foi chamado
        mock_client.files_upload.assert_called_once()

        self.assertTrue(result)

    @patch('builtins.open', new_callable=mock_open)
    @patch('dropbox.Dropbox')
    def test_upload_model_to_dropbox_failure(self, MockDropbox, mock_file):
        """Teste que simula falha no upload do modelo para o Dropbox."""
        mock_client = MockDropbox.return_value
        mock_client.files_upload.side_effect = Exception("Falha no upload")

        manager = DropboxManager(access_token="fake_token")

        # Simulando falha no upload
        result = manager.upload(
            dropbox_path="/path/to/upload/model.pt",
            local_file_path="/local/path/model.pt",
        )

        # Verificar se o arquivo foi aberto para leitura em modo binário
        mock_file.assert_called_once_with("/local/path/model.pt", "rb")

        self.assertFalse(result)

    @pytest.mark.skip(reason="Teste funcional que faz o download de um arquivo do Dropbox. "
                             "Não deve ser usado como teste unitário.")
    def test_download_dropbox(self):
        """Teste funcional que faz o download de um arquivo do Dropbox."""
        access_token = settings.dropbox.access_token

        dropbox_model_path = settings.dropbox.models.yolov8x

        path_to_model_folder = Path(__file__).parents[1] / "dev"

        path_to_model = path_to_model_folder / "yolov8x.pt"

        path_to_model_folder.mkdir(parents=True, exist_ok=True)

        manager = DropboxManager(access_token=access_token)
        manager.download(
            dropbox_path=str(dropbox_model_path),
            local_file_path=str(path_to_model),
        )

if __name__ == '__main__':
    unittest.main()
