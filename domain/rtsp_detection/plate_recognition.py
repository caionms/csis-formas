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
    MODELS_FOLDER_PATH,
    PLATES_FOLDER_PATH,
    RESOURCES_FOLDER_PATH,
)
from domain import TrackingData
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
    add_or_update_ocr,
    calculate_correct_plate,
    experimental_add_or_update_ocr,
    experimental_extract_and_save_cropped_images,
    experimental_read_license_plate,
    extract_and_save_cropped_images,
    format_license,
    load_valid_plates,
    read_license_plate,
)
from infrastructure.utils.plot_utils import plot_only_label

logger = get_logger(__name__)


def main(
    rtsp_url: str,
    show_video: bool = True,
    output_json_path: Path = DATA_FOLDER_PATH / "output_plates.json",
    image_folder_path: Path = PLATES_FOLDER_PATH,
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
        rtsp_url (str): A URL do stream RTSP a ser capturado.
        show_video (bool): Se True, exibe o vídeo anotado.
        output_json_path (Path): O caminho do arquivo JSON onde os resultados das
            detecções serão salvos.
        image_folder_path (Path): O caminho da pasta onde as imagens anotadas serão salvas.
        type_of_camera (VehicleEnum): Indica se a câmera é uma entrada ou saída.
        camera_location (str): O local da câmera onde a detecção está sendo realizada.
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Prepara captura via RTSP
    cap = cv.VideoCapture(rtsp_url)

    if cap is None:
        logger.error(
            f"[PlateDetection_RTSPDetection] Could not open RTSP stream at {rtsp_url}. "
            f"Detection cannot be performed."
        )
        return

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
        logger.error(f"[PlateDetection_RTSPDetection] {e} Detection cannot be performed.")
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
        logger.error(f"[PlateDetection_RTSPDetection] {e} Detection cannot be performed.")
        return

    ocr = initialize_paddleocr_model(
        text_detection_model_path, text_recognition_model_path, text_cls_model_path
    )

    image_folder_path.mkdir(parents=True, exist_ok=True)

    tracking_data: dict[int, dict[str, Any]] = TrackingData()

    validated_plates = list(load_valid_plates())

    last_run_time = time()

    while True:
        loop_time = time()

        # Read the current frame
        success, frame = cap.read()

        # Check if the read was successful and the frame is not None
        if not success or frame is None:
            logger.error("[PlateDetection_RTSPDetection] Could not read frame from RTSP stream.")
            break

        # Run YOLOv8 inference on the frame
        yaml_tracker = RESOURCES_FOLDER_PATH / "botsort.yaml"
        results = list(
            model.track(
                source=frame, persist=True, stream=True, conf=0.87, tracker=str(yaml_tracker)
            )
        )

        # Display the annotated frame
        annotated_frame = results[0].plot()
        if show_video:
            cv.imshow("Plate Detection Inference", annotated_frame)

        # Recorta imagens das placas
        cropped_images = extract_and_save_cropped_images(
            img=frame, results=results, save_images=False
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
        # TODO: Reduzido de 10 para 2 no desenvolvimento
        if time() - last_run_time >= 2:
            # Remove as placas já registradas
            keys_to_remove = [
                key for key, value in tracking_data.items() if value.get("registered", False)
            ]
            for key in keys_to_remove:
                del tracking_data[key]

            for track_id, track_data in tracking_data.items():
                # Se não estiver registrado e houver mais de 10 OCRs, tenta registrar
                if (
                    not track_data["registered"]
                    and track_data["ocr_plates"]
                    # TODO: Reduzido de 10 para 6 durante desenvolvimento
                    and len(track_data["ocr_plates"]) > 6
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
                            save_annotated_image(frame, str(image_folder_path))
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

            last_run_time = time()

        if len(results[0].boxes) > 0 and any([box.id for box in results[0].boxes]):
            for box, track_id, cls in zip(
                results[0].boxes.xyxy.cpu(),
                results[0].boxes.id.int().cpu().tolist(),
                results[0].boxes.cls.int(),
            ):
                if (
                    track_id in tracking_data.keys()
                    and tracking_data[track_id]["final_plate"] is not None
                ):
                    plotted_img = plot_only_label(
                        img=annotated_frame, box=box, text=tracking_data[track_id]["final_plate"]
                    )

                    save_annotated_image(plotted_img, str(image_folder_path))

        # Debug da taxa de atualização
        logger.info(f"FPS: {1 / (time() - loop_time):.2f}")

        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
    logger.info("[PlateDetection_RTSPDetection] Done.")


def main_without_tracking(
    rtsp_url: str,
    show_video: bool = True,
    output_json_path: Path = DATA_FOLDER_PATH / "output_plates.json",
    image_folder_path: Path = PLATES_FOLDER_PATH,
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
        rtsp_url (str): A URL do stream RTSP a ser capturado.
        show_video (bool): Se True, exibe o vídeo anotado.
        output_json_path (Path): O caminho do arquivo JSON onde os resultados das
            detecções serão salvos.
        image_folder_path (Path): O caminho da pasta onde as imagens anotadas serão salvas.
        type_of_camera (VehicleEnum): Indica se a câmera é uma entrada ou saída.
        camera_location (str): O local da câmera onde a detecção está sendo realizada.
        qty_frames_before_detection (int): A quantidade de frames antes de realizar a detecção.
    """
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Prepara captura via RTSP
    cap = cv.VideoCapture(rtsp_url)

    if cap is None:
        logger.error(
            f"[PlateDetection_RTSPDetection] Could not open RTSP stream at {rtsp_url}. "
            f"Detection cannot be performed."
        )
        return

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
        logger.error(f"[PlateDetection_RTSPDetection] {e} Detection cannot be performed.")
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
        logger.error(f"[PlateDetection_RTSPDetection] {e} Detection cannot be performed.")
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

    validated_plates = list(load_valid_plates())

    frames_detected = 0

    while True:
        loop_time = time()

        # Read the current frame
        success, frame = cap.read()

        # Check if the read was successful and the frame is not None
        if not success or frame is None:
            logger.error("[PlateDetection_RTSPDetection] Could not read frame from RTSP stream.")
            break

        # Run YOLOv8 inference on the frame
        results = model(source=frame, conf=0.87)

        # Display the annotated frame
        annotated_frame = results[0].plot()

        # Detectou uma placa
        if len(results[0].boxes) > 0:
            frames_detected += 1

            # Recorta imagens das placas
            cropped_images = experimental_extract_and_save_cropped_images(
                img=frame, results=results, save_images=False
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
                        save_annotated_image(frame, str(image_folder_path))
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
                            plotted_img = plot_only_label(
                                img=annotated_frame,
                                box=tracking_data["bbox"],
                                text=tracking_data["final_plate"],
                            )

                            save_annotated_image(plotted_img, str(image_folder_path))

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

        if show_video:
            cv.imshow("Plate Recognition Inference", annotated_frame)

        # Debug da taxa de atualização
        logger.info(f"FPS: {1 / (time() - loop_time):.2f}")

        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()
    logger.info("[PlateDetection_RTSPDetection] Done.")


if __name__ == "__main__":
    rtsp_url = "rtsp://"
    main_without_tracking(
        rtsp_url=rtsp_url,
        type_of_camera=VehicleEnum.IN,
        show_video=True,
        qty_frames_before_detection=3,
    )
