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
from config.paths import DATA_FOLDER_PATH, FRAMES_FOLDER_PATH, MODELS_FOLDER_PATH
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
    PlateType,
    VehicleEnum,
    calculate_correct_plate,
    extract_and_save_cropped_images,
    format_license,
    read_license_plate,
)
from infrastructure.utils.window_capture_utils import capture_window, setup_capture_window

logger = get_logger(__name__)

TrackingData = dict[int, dict[str, Any]]

validated_plates = [
    "QQV6O13",
]


def add_or_update_ocr(
    tracking_data: dict[int, dict[str, Any]],
    track_id: int,
    ocrs: list[str | None],
    plate_type: PlateType,
    registered: bool = False,
) -> None:
    """
    Adiciona ou atualiza OCRs e o estado 'registered' para um dado track_id.

    :param plate_type: Tipo da placa
    :param tracking_data: O dicionário que guarda os dados de rastreamento.
    :param track_id: O identificador único do rastreamento.
    :param ocrs: Lista de OCRs para adicionar.
    :param registered: O estado registrado (True ou False).
    """
    if track_id not in tracking_data:
        # Inicializa o track_id se não existir
        tracking_data[track_id] = {
            "ocr_plates": ocrs,
            "registered": registered,
            "plate_type": plate_type,
            "final_plate": None,
        }
    else:
        # Adiciona os novos OCRs à lista existente
        existing_ocrs: list[str] = tracking_data[track_id]["ocr_plates"]
        if existing_ocrs and len(existing_ocrs) >= 1:
            tracking_data[track_id]["ocr_plates"].extend(ocrs)
        else:
            tracking_data[track_id]["ocr_plates"] = ocrs

        # Atualiza o tipo da placa para caso tenha ocorrido um erro em distancia maior
        tracking_data[track_id]["plate_type"] = plate_type


def main(
    window_title: str | None = None,
    output_json_path: Path = DATA_FOLDER_PATH / "output_plates.json",
    image_folder_path: Path = FRAMES_FOLDER_PATH,
    type_of_camera: VehicleEnum = VehicleEnum.IN,
    camera_location: str = "Portaria 1 - Ondina",
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

    tracking_data: dict[int, dict[str, Any]] = {}

    last_run_time = time()

    while True:
        loop_time = time()

        screenshot = capture_window(window_id=window_id)

        # Run YOLOv8 inference on the frame
        results = list(model.track(source=screenshot, persist=True, stream=True, conf=0.8))

        # Display the annotated frame
        annotated_frame = results[0].plot()
        cv.imshow("Plate Detection Inference", annotated_frame)

        # Recorta imagens das placas
        cropped_images = extract_and_save_cropped_images(
            img=screenshot, results=results, save_images=False
        )

        # Placas OCR-izadas
        for cropped_image in cropped_images:
            ocr_plates = read_license_plate(cropped_image, ocr)
            add_or_update_ocr(
                tracking_data=tracking_data,
                track_id=cropped_image.track_id,
                ocrs=ocr_plates,
                plate_type=cropped_image.plate_type,
                registered=False,
            )

        # Verifica se passaram 10 segundos
        if time() - last_run_time >= 10:
            for track_id, track_data in tracking_data.items():
                # Se não estiver registrado e houver mais de 10 OCRs, tenta registrar
                if (
                    not track_data["registered"]
                    and track_data["ocr_plates"]
                    and len(track_data["ocr_plates"]) > 10
                ):
                    formatted_plates = []
                    for ocr_plate in track_data["ocr_plates"]:
                        formatted_plate, success = format_license(
                            ocr_plate, track_data["plate_type"]
                        )
                        if success:
                            formatted_plates.append(formatted_plate)
                    if formatted_plates:
                        track_data["final_plate"] = calculate_correct_plate(formatted_plates)

                        frame_path = (
                            save_annotated_image(screenshot, str(image_folder_path))
                            if track_data["final_plate"] not in validated_plates
                            else None
                        )

                        try:
                            save_plate_results_to_json(
                                file_path=str(output_json_path),
                                type_of_camera=type_of_camera,
                                plate_text=track_data["final_plate"],
                                plate_type=track_data["plate_type"],
                                camera_location=camera_location,
                                frame_path=frame_path,
                            )
                            track_data["registered"] = True
                        except Exception:
                            logger.exception("Error saving plate results to JSON.")
                            track_data["registered"] = False

            # Remove as placas já registradas
            keys_to_remove = [
                key for key, value in tracking_data.items() if value.get("registered", False)
            ]
            for key in keys_to_remove:
                del tracking_data[key]

            last_run_time = time()

        # Debug da taxa de atualização
        logger.info(f"FPS: {1 / (time() - loop_time):.2f}")

        if cv.waitKey(1) == ord("q"):
            cv.destroyAllWindows()
            break

    print("Done.")


if __name__ == "__main__":
    main(window_title="Reprodutor Multimídia", type_of_camera=VehicleEnum.OUT)
