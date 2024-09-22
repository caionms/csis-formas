from pathlib import Path
from typing import Any, Callable, Optional

import dropbox
from dropbox.exceptions import ApiError

from application.log_config import get_logger

logger = get_logger(__name__)

MODELS_PATH = Path(__file__).parents[1] / "models"


class DropboxManager:
    """Gerencia operações de upload e download com o Dropbox."""

    def __init__(self, access_token: Optional[str] = None) -> None:
        """
        Inicializa o cliente Dropbox.

        Args:
            access_token (Optional[str]): O token de acesso para autenticação com o Dropbox.
        """
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

    @ensure_client
    async def download(self, local_file_path: Path, dropbox_path: str) -> None:
        """
        Faz o download de um arquivo do Dropbox e salva no caminho local especificado.

        Args:
            local_file_path (Path): Caminho local para salvar o arquivo.
            dropbox_path (str): Caminho do arquivo no Dropbox.

        Raises:
            ApiError: Se houver um erro da API do Dropbox.
            FileNotFoundError: Se o caminho local não for encontrado.
            Exception: Para outros erros não especificados.
        """
        try:
            logger.info(f"[DropboxManager][download] Iniciando o download de {dropbox_path}.")
            with local_file_path.open("wb") as file:
                metadata, res = self.dbx_client.files_download(path=dropbox_path)
                file.write(res.content)
            logger.info(f"[DropboxManager][download] Download concluído: {dropbox_path} salvo em {local_file_path}.")
        except ApiError:
            logger.exception(f"[DropboxManager][download] Erro da API ao baixar {dropbox_path}.")
        except FileNotFoundError:
            logger.exception(f"[DropboxManager][download] Caminho local {local_file_path} não encontrado.")
        except Exception:
            logger.exception(f"[DropboxManager][download] Erro desconhecido ao baixar {dropbox_path}.")

    @ensure_client
    async def upload(self, local_file_path: Path, dropbox_path: str) -> None:
        """
        Faz o upload de um arquivo do caminho local para o Dropbox.

        Args:
            local_file_path (Path): Caminho do arquivo local a ser enviado.
            dropbox_path (str): Caminho destino no Dropbox.

        Raises:
            ApiError: Se houver um erro da API do Dropbox.
            FileNotFoundError: Se o arquivo local não for encontrado.
            Exception: Para outros erros não especificados.
        """
        try:
            logger.info(f"[DropboxManager][upload] Iniciando o upload de {local_file_path} para {dropbox_path}.")
            with local_file_path.open("rb") as file:
                self.dbx_client.files_upload(file.read(), dropbox_path)
            logger.info(f"[DropboxManager][upload] Upload concluído: {local_file_path} enviado para {dropbox_path}.")
        except ApiError:
            logger.exception(f"[DropboxManager][upload] Erro da API ao enviar {local_file_path}.")
        except FileNotFoundError:
            logger.exception(f"[DropboxManager][upload] Arquivo local {local_file_path} não encontrado.")
        except Exception:
            logger.exception(f"[DropboxManager][upload] Erro desconhecido ao enviar {local_file_path} para {dropbox_path}.")
