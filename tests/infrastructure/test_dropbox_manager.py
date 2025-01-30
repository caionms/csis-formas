# Arquivo de testes usando pytest
"""Testes para a classe DropboxManager."""

import unittest
from pathlib import Path
from unittest.mock import Mock, mock_open, patch

import pytest
import requests

from config.dynaconf_settings import settings
from infrastructure.dropbox.dropbox_manager import DropboxManager

# Configurações fictícias para mockar constantes
MOCK_APP_KEY = "mock_app_key"
MOCK_APP_SECRET = "mock_app_secret"
MOCK_REFRESH_TOKEN = "mock_refresh_token"
MOCK_OFFLINE_ACCESS_CODE = "mock_offline_access_code"
MOCK_ACCESS_TOKEN = "mock_access_token"


@pytest.fixture
def mock_constants():
    """Mocka as constantes necessárias para o DropboxManager."""
    with (
        patch("infrastructure.dropbox.dropbox_manager.DROPBOX_APP_KEY", MOCK_APP_KEY),
        patch("infrastructure.dropbox.dropbox_manager.DROPBOX_APP_SECRET", MOCK_APP_SECRET),
        patch("infrastructure.dropbox.dropbox_manager.DROPBOX_REFRESH_TOKEN", MOCK_REFRESH_TOKEN),
        patch(
            "infrastructure.dropbox.dropbox_manager.DROPBOX_OFFLINE_ACCESS_CODE",
            MOCK_OFFLINE_ACCESS_CODE,
        ),
        patch(
            "infrastructure.dropbox.dropbox_manager.DROPBOX_INITIAL_ACCESS_TOKEN", MOCK_ACCESS_TOKEN
        ),
    ):
        yield


@pytest.fixture
def mock_models_folder():
    """Mocka a criação do diretório MODELS_FOLDER_PATH."""
    with patch("os.makedirs") as mock_makedirs:
        yield mock_makedirs


@pytest.fixture
def dropbox_manager(mock_constants, mock_models_folder):
    """Retorna uma instância mockada do DropboxManager."""
    return DropboxManager(MOCK_ACCESS_TOKEN)


class TestDropboxManager:
    """Conjunto de testes para a classe DropboxManager."""

    @patch("dropbox.Dropbox")
    def test_init_success(self, mock_dropbox, mock_constants, mock_models_folder):
        """Deve inicializar o cliente Dropbox com sucesso."""
        DropboxManager(MOCK_ACCESS_TOKEN)
        mock_dropbox.assert_called_once_with(MOCK_ACCESS_TOKEN)

    @patch("dropbox.Dropbox", side_effect=Exception("Erro na inicialização"))
    def test_init_failure(self, mock_dropbox, mock_constants, mock_models_folder):
        """Deve falhar ao inicializar o cliente Dropbox."""
        manager = DropboxManager(MOCK_ACCESS_TOKEN)
        mock_dropbox.assert_called_once_with(MOCK_ACCESS_TOKEN)
        assert manager.dbx_client is None

    def test_get_base64_authorization(self, dropbox_manager):
        """Deve retornar a string de autorização em Base64."""
        result = dropbox_manager.get_base64_authorization()
        expected = (
            "bW9ja19hcHBfa2V5Om1vY2tfYXBwX3NlY3JldA=="  # base64("mock_app_key:mock_app_secret")
        )
        assert result == expected, "Deve retornar a string Base64 corretamente."

    @patch("requests.post")
    def test_request_token_success(self, mock_post, dropbox_manager):
        """Deve obter um token com sucesso."""
        mock_post.return_value = Mock(status_code=200, json=lambda: {"access_token": "new_token"})
        data = {"mock": "data"}
        response = dropbox_manager._request_token(data)
        assert response == {"access_token": "new_token"}
        mock_post.assert_called_once()

    @patch("requests.post", side_effect=requests.exceptions.RequestException("Erro na requisição"))
    @patch("infrastructure.dropbox.dropbox_manager.logger")
    def test_request_token_failure(self, mock_logger, mock_post, dropbox_manager):
        """Deve falhar ao obter um token."""
        data = {"mock": "data"}
        response = dropbox_manager._request_token(data)
        assert response is None
        mock_logger.exception.assert_called_once_with(
            "[DropboxManager][request_token] Erro ao requisitar token."
        )

    @patch("infrastructure.dropbox.dropbox_manager.DropboxManager.update_access_token")
    @patch("infrastructure.dropbox.dropbox_manager.DropboxManager._request_token")
    def test_refresh_access_token_success(
        self, mock_request_token, mock_update_access_token, dropbox_manager
    ):
        """Deve renovar o token de acesso com sucesso."""
        mock_request_token.return_value = {"access_token": "new_token"}
        result = dropbox_manager._refresh_access_token()
        assert result is True
        assert dropbox_manager.temporary_access_token == "new_token"
        mock_update_access_token.assert_called_once()

    @patch(
        "infrastructure.dropbox.dropbox_manager.DropboxManager._request_token", return_value=None
    )
    def test_refresh_access_token_failure(self, mock_request_token, dropbox_manager):
        """Deve falhar ao renovar o token de acesso."""
        result = dropbox_manager._refresh_access_token()
        assert result is False
        mock_request_token.assert_called_once()

    @patch("builtins.open", new_callable=mock_open)
    def test_save_to_file_success(self, mock_file, dropbox_manager):
        """Deve salvar conteúdo em arquivo com sucesso."""
        result = dropbox_manager._save_to_file("/fake/path/file.txt", b"content")
        assert result is True
        mock_file.assert_called_once_with("/fake/path/file.txt", "wb")

    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_save_to_file_failure(self, mock_file, dropbox_manager):
        """Deve falhar ao salvar conteúdo em arquivo."""
        result = dropbox_manager._save_to_file("/invalid/path/file.txt", b"content")
        assert result is False
        mock_file.assert_called_once_with("/invalid/path/file.txt", "wb")

    @patch("os.makedirs")
    @patch("dropbox.Dropbox.files_list_folder")
    def test_download_folder_success(self, mock_list_folder, mock_makedirs, dropbox_manager):
        """Deve fazer o download de uma pasta com sucesso."""
        mock_list_folder.return_value = Mock(entries=[])
        result = dropbox_manager.download_folder("/local/folder", "/dropbox/folder")
        assert result is True
        mock_makedirs.assert_called_once_with("/local/folder")

    @patch("dropbox.Dropbox.files_list_folder", side_effect=Exception("Erro ao listar"))
    def test_download_folder_failure(self, mock_list_folder, dropbox_manager):
        """Deve falhar ao fazer o download de uma pasta."""
        result = dropbox_manager.download_folder("/local/folder", "/dropbox/folder")
        assert result is False
        mock_list_folder.assert_called_once_with("/dropbox/folder")

    @patch("builtins.open", new_callable=mock_open)  # Mockando o método open
    @patch("dropbox.Dropbox")  # Mockando a classe Dropbox do SDK
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

        assert result is True

    @patch("builtins.open", new_callable=mock_open)
    @patch("dropbox.Dropbox")
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

        assert result is False

    @patch("builtins.open", new_callable=mock_open)
    @patch("dropbox.Dropbox")
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

        assert result is True

    @patch("builtins.open", new_callable=mock_open)
    @patch("dropbox.Dropbox")
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

        assert result is False

    @pytest.mark.skip(
        reason="Teste funcional que faz o download de um arquivo do Dropbox. "
        "Não deve ser usado como teste unitário."
    )
    def test_download_dropbox(self):
        """Teste funcional que faz o download de um arquivo do Dropbox."""
        access_token = settings.dropbox.access_token

        dropbox_model_path = settings.dropbox.models.yolo11x

        model_filename = dropbox_model_path.split("/")[-1]

        path_to_model_folder = Path(__file__).parents[1] / "dev"

        path_to_model = path_to_model_folder / model_filename

        path_to_model_folder.mkdir(parents=True, exist_ok=True)

        manager = DropboxManager(access_token=access_token)
        manager.download(
            dropbox_path=str(dropbox_model_path),
            local_file_path=str(path_to_model),
        )


if __name__ == "__main__":
    unittest.main()
