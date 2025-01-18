"""
Módulo que executa detecção de segurança pública em uma janela.
"""

import os
from pathlib import Path
from time import time

import cv2 as cv

from application import (
    DROPBOX_ACCESS_TOKEN,
    PUBLIC_SAFETY_MODEL_DROPBOX_PATH,
)
from domain.window_capture.wc_config import (
    DATA_FOLDER_PATH,
    FRAMES_FOLDER_PATH,
    MODELS_FOLDER_PATH,
)
from infrastructure.logging.log_config import get_logger
from infrastructure.utils.dashboard_utils import save_annotated_image, save_results_to_json
from infrastructure.utils.model_utils import (
    NoModelAvailableException,
    download_model,
    initialize_yolo_model,
)
from infrastructure.utils.window_capture_utils import capture_window, setup_capture_window

logger = get_logger(__name__)


def main(
    window_title: str | None = None,
    output_json_path: Path = DATA_FOLDER_PATH / "output.json",
    image_folder_path: Path = FRAMES_FOLDER_PATH,
    camera_location: str = "Portaria 1 - Ondina",
) -> None:
    """
    Captura continuamente a tela de uma janela específica ou da área de trabalho,
    realiza detecção com o modelo YOLO e salva os resultados em um arquivo JSON.

    A função exibe os frames anotados com as detecções em uma janela OpenCV e salva os
    resultados de detecção e a imagem anotada a cada segundo, caso haja detecções.

    Args:
        window_title (Optional[str]): O título da janela a ser capturada. Se não for
            especificado, captura a área de trabalho.
        output_json_path (Path): O caminho do arquivo JSON onde os resultados das
            detecções serão salvos.
        image_folder_path (Path): O caminho da pasta onde as imagens anotadas serão salvas.
        camera_location (str): O local da câmera onde a detecção está sendo realizada.
    """
    # Change the working directory to the folder this script is in.
    # Doing this because I'll be putting the files from each video in their own folder on GitHub
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Prepara captura de janela
    window_id = setup_capture_window(window_title)

    # Load the model
    model_filename = PUBLIC_SAFETY_MODEL_DROPBOX_PATH.split("/")[-1]
    model_path = MODELS_FOLDER_PATH / model_filename

    try:
        download_model(model_path, PUBLIC_SAFETY_MODEL_DROPBOX_PATH, DROPBOX_ACCESS_TOKEN)
    except NoModelAvailableException as e:
        logger.error(f"[PublicSafetyDetection] {e} Detection cannot be performed.")
        return

    model = initialize_yolo_model(model_path)

    # Obtem o nome das classes
    classes_names = model.names

    # Cria a pasta de frames se ela não existir (e consequentemente a de dados)
    image_folder_path.mkdir(parents=True, exist_ok=True)

    last_save_time = time()
    while True:
        loop_time = time()

        screenshot = capture_window(window_id=window_id)

        # Run YOLOv8 inference on the frame
        results = model(screenshot)

        # Display the annotated frame
        annotated_frame = results[0].plot()
        cv.imshow("Public Safety Inference", annotated_frame)

        # Salva imagem anotada e resultados em um arquivo JSON a cada segundo se houver detecções
        if time() - last_save_time >= 1.0 and len(results[0].boxes) > 0:
            frame_path = save_annotated_image(annotated_frame, str(image_folder_path))
            save_results_to_json(
                results=results,
                file_path=str(output_json_path),
                frame_path=frame_path,
                model_name=model_filename,
                classes_names=classes_names,
                camera_location=camera_location,
            )
            last_save_time = time()

        # Debug da taxa de atualização
        logger.info(f"FPS: {1 / (time() - loop_time):.2f}")

        if cv.waitKey(1) == ord("q"):
            cv.destroyAllWindows()
            break

    print("Done.")


if __name__ == "__main__":
    main(
        window_title="Reprodutor Multimídia",
    )
