"""
Módulo de utilitários para captura de janela do windows.
"""

import mss
import numpy as np
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


def capture_window(window_title: str) -> np.ndarray:
    """
    Captura a tela de uma janela específica e retorna a imagem como um array numpy.

    Args:
        window_title (str): O título da janela a ser capturada.

    Returns:
        np.ndarray: Uma imagem da janela capturada como um array numpy.

    Raises:
        Exception: Se a janela com o título especificado não for encontrada.
    """
    # Encontra a janela pelo título
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd:
        raise Exception(f"Janela com o título '{window_title}' não encontrada.")

    # Obtém as dimensões da janela
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
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
