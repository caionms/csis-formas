"""
Módulo que executa detecção de placas veículares em uma janela.
"""

import os
from pathlib import Path
from time import time
from typing import Any

import cv2 as cv

from config.globals import (
    DROPBOX_ACCESS_TOKEN,
    PLATE_PADDLE_CLS_MODEL_DROPBOX_PATH,
    PLATE_PADDLE_DETECTION_MODEL_DROPBOX_PATH,
    PLATE_PADDLE_RECOGNITION_MODEL_DROPBOX_PATH,
    PLATE_YOLO_DETECTION_MODEL_DROPBOX_PATH,
)
from config.paths import (
    DATA_FOLDER_PATH,
    FRAMES_FOLDER_PATH,
    MODELS_FOLDER_PATH,
)
from domain.enums.plate_enum import VehicleEnum
from infrastructure.logging.log_config import get_logger
from infrastructure.utils.dashboard_utils import save_annotated_image, save_plate_results_to_json
from infrastructure.utils.model_utils import (
    NoModelAvailableException,
    download_model,
    download_paddle_folder_model,
    initialize_paddleocr_model,
    initialize_yolo_model,
)
from infrastructure.utils.plate_utils import (
    calculate_correct_plate,
    experimental_add_or_update_ocr,
    experimental_extract_and_save_cropped_images,
    experimental_read_license_plate,
    format_license,
)
from infrastructure.utils.plot_utils import plot_only_label
from infrastructure.utils.window_capture_utils import capture_window, setup_capture_window

logger = get_logger(__name__)

validated_plates = [
    "QQV6O13",
]


def main(
    window_title: str | None = None,
    output_json_path: Path = DATA_FOLDER_PATH / "output_plates.json",
    image_folder_path: Path = FRAMES_FOLDER_PATH,
    type_of_camera: VehicleEnum = VehicleEnum.IN,
    camera_location: str = "Portaria 1 - Ondina",
    qty_frames_before_detection: int = 3,
) -> None:
    """
    Captura continuamente a tela de uma janela específica ou da área de trabalho,
    realiza detecção de placas com o modelo do YOLO e em seguida faz o OCR com o
    modelo do PaddleOCR e salva os resultados em um arquivo JSON.

    A função exibe os frames anotados com as detecções em uma janela OpenCV e salva os
    resultados de detecção e a imagem anotada a cada segundo, caso haja detecções.

    Args:
        window_title (Optional[str]): O título da janela a ser capturada. Se não for
            especificado, captura a área de trabalho.
        output_json_path (Path): O caminho do arquivo JSON onde os resultados das
            detecções serão salvos.
        image_folder_path (Path): O caminho da pasta onde as imagens anotadas serão salvas.
        type_of_camera (VehicleEnum): Indica se a câmera é uma entrada ou saída.
        camera_location (str): O local da câmera onde a detecção está sendo realizada.
        qty_frames_before_detection (int): A quantidade de frames antes da detecção.
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Prepara captura de janela
    window_id = setup_capture_window(window_title)

    # Load the plate detection model
    plate_detection_model_filename = PLATE_YOLO_DETECTION_MODEL_DROPBOX_PATH.split("/")[-1]
    plate_detection_model_path = MODELS_FOLDER_PATH / plate_detection_model_filename

    try:
        download_model(
            plate_detection_model_path,
            PLATE_YOLO_DETECTION_MODEL_DROPBOX_PATH,
            DROPBOX_ACCESS_TOKEN,
        )
    except NoModelAvailableException as e:
        logger.error(f"[PlateRecognition] {e} Detection cannot be performed.")
        return

    model = initialize_yolo_model(plate_detection_model_path)

    # Load the OCR model
    text_detection_model_filename = PLATE_PADDLE_DETECTION_MODEL_DROPBOX_PATH.split("/")[-1]
    text_detection_model_path = MODELS_FOLDER_PATH / text_detection_model_filename

    text_recognition_model_filename = PLATE_PADDLE_RECOGNITION_MODEL_DROPBOX_PATH.split("/")[-1]
    text_recognition_model_path = MODELS_FOLDER_PATH / text_recognition_model_filename

    text_cls_model_filename = PLATE_PADDLE_CLS_MODEL_DROPBOX_PATH.split("/")[-1]
    text_cls_model_path = MODELS_FOLDER_PATH / text_cls_model_filename

    try:
        download_paddle_folder_model(
            text_detection_model_path,
            PLATE_PADDLE_DETECTION_MODEL_DROPBOX_PATH,
            DROPBOX_ACCESS_TOKEN,
        )
        download_paddle_folder_model(
            text_recognition_model_path,
            PLATE_PADDLE_RECOGNITION_MODEL_DROPBOX_PATH,
            DROPBOX_ACCESS_TOKEN,
        )
        download_paddle_folder_model(
            text_cls_model_path, PLATE_PADDLE_CLS_MODEL_DROPBOX_PATH, DROPBOX_ACCESS_TOKEN
        )
    except NoModelAvailableException as e:
        logger.error(f"[PlateRecognition] {e} Detection cannot be performed.")
        return

    ocr = initialize_paddleocr_model(
        text_detection_model_path, text_recognition_model_path, text_cls_model_path
    )

    image_folder_path.mkdir(parents=True, exist_ok=True)

    tracking_data: dict[str, Any] = {
        "ocr_plates": [],
        "registered": False,
        "plate_type": None,
        "final_plate": None,
        "bbox": None,
    }

    frames_detected = 0
    while True:
        loop_time = time()

        screenshot = capture_window(window_id=window_id)

        # Run YOLOv8 inference on the frame
        results = model(source=screenshot, conf=0.87)

        # Display the annotated frame
        annotated_frame = results[0].plot()

        # Detectou uma placa
        if len(results[0].boxes) > 0:
            frames_detected += 1

            # Recorta imagens das placas
            cropped_images = experimental_extract_and_save_cropped_images(
                img=screenshot, results=results, save_images=False
            )

            # Placas OCR-izadas
            for cropped_image in cropped_images:
                ocr_plates = experimental_read_license_plate(cropped_image, ocr)
                experimental_add_or_update_ocr(
                    tracking_data=tracking_data,
                    ocrs=ocr_plates,
                    plate_type=cropped_image.plate_type,
                    bbox=cropped_image.bbox,
                )

            if (
                frames_detected > qty_frames_before_detection
                and tracking_data["ocr_plates"]
                and len(tracking_data["ocr_plates"]) > 6
            ):
                formatted_plates = []
                for ocr_plate in tracking_data["ocr_plates"]:
                    formatted_plate, success = format_license(
                        ocr_plate, tracking_data["plate_type"]
                    )
                    if success:
                        formatted_plates.append(formatted_plate)
                if formatted_plates:
                    tracking_data["final_plate"] = calculate_correct_plate(formatted_plates)

                    frame_path = (
                        save_annotated_image(screenshot, str(image_folder_path))
                        if tracking_data["final_plate"] not in validated_plates
                        else None
                    )

                    try:
                        if tracking_data["final_plate"] not in validated_plates:
                            save_plate_results_to_json(
                                file_path=str(output_json_path),
                                type_of_camera=type_of_camera,
                                plate_text=tracking_data["final_plate"],
                                plate_type=tracking_data["plate_type"],
                                camera_location=camera_location,
                                frame_path=frame_path,
                            )
                        tracking_data["registered"] = True

                        if (
                            tracking_data["bbox"] is not None
                            and tracking_data["final_plate"] is not None
                        ):
                            plot_only_label(
                                img=annotated_frame,
                                box=tracking_data["bbox"],
                                text=tracking_data["final_plate"],
                            )

                        tracking_data: dict[str, Any] = {
                            "ocr_plates": [],
                            "registered": False,
                            "plate_type": None,
                            "final_plate": None,
                        }

                    except Exception:
                        logger.exception(
                            "[PlateRecognition_VideoDetection] Error saving "
                            "plate results to JSON."
                        )
                        tracking_data["registered"] = False

        cv.imshow("Plate Recognition Inference", annotated_frame)

        # Debug da taxa de atualização
        logger.info(f"FPS: {1 / (time() - loop_time):.2f}")

        if cv.waitKey(1) == ord("q"):
            cv.destroyAllWindows()
            break

    print("Done.")


if __name__ == "__main__":
    main(
        window_title="Reprodutor Multimídia",
        type_of_camera=VehicleEnum.OUT,
        qty_frames_before_detection=3,
    )
