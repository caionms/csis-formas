"""
Módulo de utilitários para captura de janela do windows.
"""

from time import sleep

import mss
import numpy as np
import win32con
import win32gui


def list_window_names() -> None:
    """
    Lista os nomes de todas as janelas visíveis no sistema.

    Esta função utiliza a API do Windows para enumerar todas as janelas
    visíveis e imprime seus identificadores e títulos.
    """

    def winEnumHandler(hwnd: int, ctx: None) -> None:
        if win32gui.IsWindowVisible(hwnd):
            print(hex(hwnd), win32gui.GetWindowText(hwnd))

    win32gui.EnumWindows(winEnumHandler, None)


def setup_capture_window(window_title: str | None) -> int:
    """
    Configura a janela para captura de tela, restaurando a janela se necessário
    e garantindo que ela esteja visível.

    Args:
        window_title (Optional[str]): O título da janela a ser capturada.
        Se None, captura a área de trabalho.

    Returns:
        int: O identificador da janela (hwnd) a ser capturada.

    Raises:
        ValueError: Se a janela com o título especificado não for encontrada.
    """
    hwnd = (
        win32gui.GetDesktopWindow()
        if window_title is None
        else win32gui.FindWindow(None, window_title)
    )
    if not hwnd:
        raise ValueError(f"Window not found: {window_title}")

    # Restaura a janela se estiver minimizada, mas sem redimensioná-la
    win32gui.ShowWindow(hwnd, win32con.SW_SHOWNOACTIVATE)

    # Coloca a janela no topo da pilha sem dar foco e sem redimensionar
    win32gui.SetWindowPos(
        hwnd, win32con.HWND_TOP, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE
    )

    # Aguarda um momento para garantir que a janela esteja visível
    sleep(0.5)

    return hwnd


def capture_window(window_id: int | None = None, window_title: str | None = None) -> np.ndarray:
    """
    Captura a tela de uma janela específica e retorna a imagem como um array numpy.
    A identificação da janela é feita pelo identificador ou pelo título.

    Args:
        window_id (int): O identificador da janela a ser capturada.
        window_title (str): O título da janela a ser capturada.

    Returns:
        np.ndarray: Uma imagem da janela capturada como um array numpy.

    Raises:
        Exception: Se a janela com o título especificado não for encontrada ou
        se não for especificado o identificador ou o título da janela.
    """
    if window_id is None:
        if window_title is None:
            raise Exception("Deve ser especificado o identificador ou o título da janela.")
        # Encontra a janela pelo título
        window_id = win32gui.FindWindow(None, window_title)
        if not window_id:
            raise Exception(f"Janela com o título '{window_title}' não encontrada.")

    # Obtém as dimensões da janela
    left, top, right, bottom = win32gui.GetWindowRect(window_id)
    width = right - left
    height = bottom - top

    with mss.mss() as sct:
        # Define a área a ser capturada
        monitor = {"top": top, "left": left, "width": width, "height": height}
        screenshot = sct.grab(monitor)

        # Converte a captura para um array numpy
        img = np.array(screenshot)

        # Remove o canal alfa para converter a imagem para 3 canais (RGB) compatível como YOLO
        img = img[:, :, :3]

    return img


if __name__ == "__main__":
    # Para conferir o nome da janela, basta rodar o list_windows_names e
    # pegar o nome da janela depois do código dela
    list_window_names()
