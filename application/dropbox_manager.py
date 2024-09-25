import base64
import os
from pathlib import Path
from typing import Any, Callable, Optional

import dropbox
import requests
import toml
from dropbox.exceptions import ApiError, AuthError

from application.log_config import get_logger
from config import settings

logger = get_logger(__name__)

MODELS_PATH = Path(__file__).parent / "models"

DROPBOX_APP_KEY = settings.dropbox.app_key
DROPBOX_APP_SECRET = settings.dropbox.app_secret
DROPBOX_REFRESH_TOKEN = settings.dropbox.refresh_token
DROPBOX_OFFLINE_ACCESS_CODE = settings.dropbox.offline_access_code
DROPBOX_INITIAL_ACCESS_TOKEN = settings.dropbox.access_token


class DropboxManager:
    """Gerencia operações de upload e download com o Dropbox."""

    def __init__(self, access_token: str) -> None:
        """
        Inicializa o cliente Dropbox.

        Args:
            access_token (Optional[str]): O token de acesso para autenticação com o Dropbox.
        """
        MODELS_PATH.mkdir(parents=True, exist_ok=True)

        self.temporary_access_token = access_token
        self.app_key = DROPBOX_APP_KEY
        self.app_secret = DROPBOX_APP_SECRET
        self.refresh_token = DROPBOX_REFRESH_TOKEN
        self.offline_access_code = DROPBOX_OFFLINE_ACCESS_CODE

        try:
            self.dbx_client = dropbox.Dropbox(access_token)
            logger.info("[DropboxManager][init] Cliente Dropbox inicializado com sucesso.")
        except Exception:
            self.dbx_client = None
            logger.exception("[DropboxManager][init] Erro ao inicializar cliente Dropbox.")

    @staticmethod
    def ensure_client(func: Callable[..., Any]) -> Callable[..., Any]:
        """
        Decorador para verificar se o cliente Dropbox foi inicializado antes de executar o método.

        Args:
            func (Callable[..., Any]): Função a ser decorada.

        Returns:
            Callable[..., Any]: Função decorada que executa apenas se o cliente estiver inicializado.
        """

        def wrapper(self, *args, **kwargs) -> Any:
            if not self.dbx_client:
                logger.error(f"[DropboxManager][{func.__name__}] No client instantiated")
                return
            return func(self, *args, **kwargs)

        return wrapper

    @staticmethod
    def update_access_token(new_token: str, file_path: str = ".secrets.local.toml") -> None:
        """
        Atualiza o access_token no arquivo .secrets.local.toml.

        Args:
            new_token (str): O novo access token que será inserido no arquivo.
            file_path (str): O caminho para o arquivo .secrets.local.toml.
        """
        try:
            if Path(file_path).exists():
                with open(file_path, "r") as file:
                    config = toml.load(file)
            else:
                config = {}

            if 'dropbox' in config and 'access_token' in config['dropbox']:
                config['dropbox']['access_token'] = new_token
            else:
                config['dropbox'] = {'access_token': new_token}

            with open(file_path, "w") as file:
                toml.dump(config, file)

            logger.info(f"[DropboxManager][update_access_token] Access token atualizado com sucesso: {new_token}")

        except Exception:
            logger.exception("[DropboxManager][update_access_token] Erro ao atualizar o access_token.")

    def get_base64_authorization(self) -> str:
        """Gera a string de autorização Base64.

        Returns:
            str: A string de autorização codificada em Base64.
        """
        return base64.b64encode(f"{self.app_key}:{self.app_secret}".encode()).decode()

    def _request_token(self, data: dict) -> Optional[dict]:
        """Envia uma requisição para a API do Dropbox para obter o token.

        Args:
            data (dict): O corpo da requisição.

        Returns:
            dict: A resposta da API do Dropbox, se bem-sucedida.
        """
        try:
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded',
                'Authorization': f"Basic {self.get_base64_authorization()}"
            }
            response = requests.post(
                'https://api.dropbox.com/oauth2/token',
                headers=headers,
                data=data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            logger.exception("[DropboxManager][request_token] Erro ao requisitar token.")
            return None

    def _refresh_access_token(self) -> bool:
        """
        Renova o access token expirado usando o refresh token.
        Caso o processo seja um sucesso, atualiza o access token no arquivo .secrets.local.toml.

        Returns:
            bool: True se o token foi renovado com sucesso, False caso contrário.
        """
        data = {
            'refresh_token': self.refresh_token,
            'grant_type': 'refresh_token',
        }

        try:
            response = self._request_token(data)
            if response and not response.get('error'):
                self.temporary_access_token = response['access_token']
                logger.info(
                    f"[DropboxManager][_refresh_access_token] Access token renovado com sucesso: {self.temporary_access_token}")
                secrets_local_path = Path(__file__).parents[1] / ".secrets.local.toml"
                self.update_access_token(self.temporary_access_token, str(secrets_local_path))
                return True
            else:
                logger.error(f"[DropboxManager][_refresh_access_token] Erro ao renovar access token: {response}")
        except Exception:
            logger.error("[DropboxManager][_refresh_access_token] Falha desconhecida ao renovar o access token.")

        return False

    def _attempt_download(self, dropbox_path: str) -> Optional[tuple]:
        """
        Tenta fazer o download de um arquivo do Dropbox.

        Args:
            dropbox_path (str): Caminho do arquivo no Dropbox.

        Returns:
            Optional[tuple]: Metadados e conteúdo do arquivo, ou None em caso de falha.
        """
        try:
            return self.dbx_client.files_download(path=dropbox_path)
        except ApiError:
            logger.exception(f"[DropboxManager][_attempt_download] Erro ao baixar {dropbox_path}.")
            return None

    def _save_to_file(self, local_file_path: str, content: bytes) -> bool:
        """
        Salva o conteúdo baixado em um arquivo local.

        Args:
            local_file_path (str): Caminho local onde o arquivo será salvo.
            content (bytes): Conteúdo a ser salvo no arquivo.

        Returns:
            bool: True se o arquivo foi salvo com sucesso, False caso contrário.
        """
        try:
            with open(local_file_path, "wb") as file:
                file.write(content)
            logger.info(f"[DropboxManager][_save_to_file] Arquivo salvo em {local_file_path}.")
            return True
        except FileNotFoundError:
            logger.exception(f"[DropboxManager][_save_to_file] Caminho local {local_file_path} não encontrado.")
        except Exception:
            logger.exception("[DropboxManager][_save_to_file] Erro desconhecido ao salvar o arquivo.")
        return False

    @ensure_client
    def download(self, local_file_path: str, dropbox_path: str) -> bool:
        """
        Faz o download de um arquivo do Dropbox e salva no caminho local especificado.

        Args:
            local_file_path (str): Caminho local para salvar o arquivo.
            dropbox_path (str): Caminho do arquivo no Dropbox.

        Returns:
            bool: Retorna True se o download foi bem-sucedido, False caso contrário.
        """
        logger.info(f"[DropboxManager][download] Iniciando o download de {dropbox_path}.")

        try:
            metadata, res = self._attempt_download(dropbox_path)
            if res:
                return self._save_to_file(local_file_path, res.content)
        except AuthError as auth_err:
            if auth_err.error.is_expired_access_token():
                logger.error("[DropboxManager][download] Token expirado. Tentando renovar...")
                if self._refresh_access_token():
                    metadata, res = self._attempt_download(dropbox_path)
                    if res:
                        return self._save_to_file(local_file_path, res.content)
            else:
                logger.exception("[DropboxManager][download] Erro de autenticação.")
        except ApiError:
            logger.exception(f"[DropboxManager][download] Erro da API ao baixar {dropbox_path}.")
        except Exception:
            logger.exception(f"[DropboxManager][download] Erro desconhecido ao baixar {dropbox_path}.")

        return False

    @ensure_client
    def download_folder(self, local_folder_path: str, dropbox_folder_path: str) -> bool:
        """
        Faz o download de uma pasta do Dropbox e salva no caminho local especificado.

        Args:
            local_folder_path (str): Caminho local para salvar a pasta.
            dropbox_folder_path (str): Caminho da pasta no Dropbox.

        Returns:
            bool: Retorna True se o download foi bem-sucedido, False caso contrário.
        """
        try:
            logger.info(f"[DropboxManager][download_folder] Iniciando o download da pasta {dropbox_folder_path}.")

            if not os.path.exists(local_folder_path):
                os.makedirs(local_folder_path)

            # Lista os arquivos e subpastas no Dropbox
            response = self.dbx_client.files_list_folder(dropbox_folder_path)

            # Itera pelos itens da pasta
            for entry in response.entries:
                dropbox_entry_path = entry.path_lower
                local_entry_path = os.path.join(local_folder_path, entry.name)

                if isinstance(entry, dropbox.files.FileMetadata):
                    # Se for um arquivo, faz o download
                    self.download(local_entry_path, dropbox_entry_path)
                elif isinstance(entry, dropbox.files.FolderMetadata):
                    # Se for uma subpasta, chama a função recursivamente
                    self.download_folder(local_entry_path, dropbox_entry_path)

            return True
        except Exception:
            logger.exception(
                f"[DropboxManager][download_folder] Erro desconhecido ao baixar a pasta {dropbox_folder_path}.")
            return False
    @ensure_client
    def upload(self, local_file_path: str, dropbox_path: str) -> bool:
        """
        Faz o upload de um arquivo do caminho local para o Dropbox.

        Args:
            local_file_path (str): Caminho do arquivo local a ser enviado.
            dropbox_path (str): Caminho destino no Dropbox.

        Returns:
            bool: Retorna se baixou com sucesso ou não.

        Raises:
            ApiError: Se houver um erro da API do Dropbox.
            FileNotFoundError: Se o arquivo local não for encontrado.
            Exception: Para outros erros não especificados.
        """
        try:
            logger.info(f"[DropboxManager][upload] Iniciando o upload de {local_file_path} para {dropbox_path}.")
            with open(local_file_path, "rb") as file:
                self.dbx_client.files_upload(file.read(), dropbox_path)
            logger.info(f"[DropboxManager][upload] Upload concluído: {local_file_path} enviado para {dropbox_path}.")
            return True
        except ApiError:
            logger.exception(f"[DropboxManager][upload] Erro da API ao enviar {local_file_path}.")
        except FileNotFoundError:
            logger.exception(f"[DropboxManager][upload] Arquivo local {local_file_path} não encontrado.")
        except Exception:
            logger.exception(
                f"[DropboxManager][upload] Erro desconhecido ao enviar {local_file_path} para {dropbox_path}.")
        return False


if __name__ == '__main__':
    manager = DropboxManager(DROPBOX_INITIAL_ACCESS_TOKEN)
    manager._refresh_access_token()
    print(manager.temporary_access_token)
