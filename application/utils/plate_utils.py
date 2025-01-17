"""Módulo de métodos utilitários para manipulação de placas veiculares."""

import string
from collections import Counter
from datetime import datetime
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics.engine.results import Results

from application.log_config import get_logger
from application.window_capture.wc_config import PLATES_FOLDER_PATH

logger = get_logger(__name__)


class VehicleEnum(Enum):
    """Enum que mapeia a entrada ou saída de veículos."""

    IN = "IN"
    """Entrada de veículo."""
    OUT = "OUT"
    """Saída de veículo."""


class PlateType(Enum):
    """Enum que mapeia o tipo de placa."""

    OLD = "old"
    MERCOSUL = "mercosul"


class CroppedPlate:
    """
    Classe que guarda a imagem recortada em 2 disposições de cores:
    1. RGB
    2. Tons de cinza
    """

    plate_text: str | None = None

    def __init__(
        self, track_id: int, rgb: np.ndarray, gray: np.ndarray, plate_type: PlateType, label: str
    ):
        self.track_id = track_id
        self.rgb = rgb
        self.gray = gray
        self.plate_type = plate_type
        self.label = label


# Mapping dictionaries for character conversion
DICT_CHAR_TO_INT = {
    PlateType.OLD: {"O": "0", "I": "1", "J": "3", "A": "4", "G": "6", "S": "5", "B": "8"},
    PlateType.MERCOSUL: {
        "O": "0",
        "J": "1",
        "A": "4",
        "P": "9",
        "G": "6",
        "S": "5",
        "B": "8",
        "Z": "7",
    },
}

DICT_INT_TO_CHAR = {
    PlateType.OLD: {"0": "O", "1": "I", "3": "J", "4": "A", "6": "G", "5": "S", "8": "B"},
    PlateType.MERCOSUL: {
        "0": "O",
        "1": "J",
        "4": "A",
        "9": "P",
        "6": "G",
        "5": "S",
        "8": "B",
        "2": "Z",
    },
}

PATTERN = {
    PlateType.OLD: [
        string.ascii_uppercase,
        string.ascii_uppercase,
        string.ascii_uppercase,
        "0123456789",
        "0123456789",
        "0123456789",
        "0123456789",
    ],
    PlateType.MERCOSUL: [
        string.ascii_uppercase,
        string.ascii_uppercase,
        string.ascii_uppercase,
        "0123456789",
        string.ascii_uppercase,
        "0123456789",
        "0123456789",
    ],
}


def replace_char_at_position(original_string: str, char: str, position: int) -> str:
    """
    Substitui um caractere em uma posição específica em uma string.

    Args:
        original_string (str): A string original onde o caractere será substituído.
        char (str): O novo caractere a ser inserido na posição especificada.
        position (int): A posição do caractere que será substituído.

    Returns:
        str: A nova string com o caractere substituído.
    """
    # Verifica se a posição é válida
    if position < 0 or position >= len(original_string):
        raise ValueError("A posição deve estar dentro do intervalo da string.")

    # Substitui o caractere na posição desejada
    new_string = original_string[:position] + char + original_string[position + 1 :]
    return new_string


def insert_char_at_position(original_string: str, char: str, position: int) -> str:
    """
    Insere um caractere em uma posição específica em uma string.

    Args:
        original_string (str): A string original onde o caractere será inserido.
        char (str): O caractere a ser inserido.
        position (int): A posição onde o caractere será inserido.

    Returns:
        str: A nova string com o caractere inserido.
    """
    # Verifica se a posição é válida
    if position < 0 or position > len(original_string):
        raise ValueError("A posição deve estar dentro do intervalo da string.")

    # Divide a string e insere o caractere na posição desejada
    new_string = original_string[:position] + char + original_string[position:]
    return new_string


def is_valid_character(char: str) -> bool:
    """
    Verifica se o caractere é uma letra ou um número.
    """
    return char in string.ascii_uppercase or char.isdigit()


def clean_text(text: str) -> str:
    """
    Limpa o texto da OCR removendo espaços, convertendo para maiúsculas e, se necessário,
    removendo caracteres indesejados no início ou no final.
    Lida com o cenário de adicionar [(,),{,}] no inicio e fim de placas.
    """
    text = text.upper().replace(" ", "")

    # Remover caracteres no início ou no final que não são letras ou números
    if text[0] != "#" and not is_valid_character(text[0]):
        text = text[1:]
    if text[0] != "#" and not is_valid_character(text[-1]):
        text = text[:-1]

    return text


def calculate_correct_plate(plates: list[str]) -> str:
    """
    Calculate the correct license plate based on the most frequent character in each position.

    Args:
        plates (list[str]): list of license plates to analyze.

    Returns:
        str: The most likely correct license plate.
    """
    # Verifica se todas as placas têm o mesmo comprimento
    if not all(len(plate) == len(plates[0]) for plate in plates):
        raise ValueError("Todas as placas devem ter o mesmo comprimento.")

    correct_plate = []

    # Iterar sobre cada posição dos caracteres nas placas
    for i in range(len(plates[0])):
        # Coletar todos os caracteres na posição i
        chars_at_position = [plate[i] for plate in plates]

        # Determinar o caractere mais comum na posição i
        most_common_char = Counter(chars_at_position).most_common(1)[0][0]

        # Adicionar o caractere mais comum à placa correta
        correct_plate.append(most_common_char)

    # Combinar a lista de caracteres para formar a placa final
    return "".join(correct_plate)


def _char_matches_pattern(char: str, pattern: str, conversion_dict: dict) -> bool:
    """
    Helper function to determine if a character matches the expected pattern.

    Args:
        char (str): The character to check.
        pattern (str): The pattern string it should match.
        conversion_dict (dict): The conversion dictionary for character replacements.

    Returns:
        bool: True if the character matches the pattern, False otherwise.
    """
    return char in pattern or char in conversion_dict and conversion_dict[char] in pattern


def license_complies_format(text: str, plate_type: PlateType) -> bool:
    """
    Check if the license plate text complies with the specified format.

    Args:
        text (str): License plate text.
        plate_type (PlateType): Indicates whether the plate is Mercosul or old Brazilian format.

    Returns:
        bool: True if the license plate complies with the specified format, False otherwise.
    """
    text = text.replace("-", "").replace(".", "").replace(":", "").replace(" ", "").upper()
    print(f"texto {text}")
    if len(text) < 7:
        return False
    elif len(text) == 8:
        if text[0] == "1" or text[0] == "l" or text[0] == "|" or text[0] == "I":
            # Se o primeiro caractere for um 1, l, | ou I, remova-o
            text = text[1:]
        elif text[7] == "1" or text[7] == "l" or text[7] == "|" or text[7] == "I":
            # Se o último caractere for um 1, l, | ou I, remova-o
            text = text[:-1]
    elif (
        len(text) == 9
        and (text[0] == "1" or text[0] == "l" or text[0] == "|" or text[0] == "I")
        and (text[8] == "1" or text[8] == "l" or text[8] == "|" or text[8] == "I")
    ):
        # Se o primeiro e último caracteres forem 1, l, | ou I, remova-os
        text = text[1:-1]

    pattern = PATTERN[plate_type]
    char_to_int = DICT_CHAR_TO_INT[plate_type]
    int_to_char = DICT_INT_TO_CHAR[plate_type]

    for i, char in enumerate(text):
        if i in [0, 1, 2]:  # Common for both OLD and MERCOSUL
            if not _char_matches_pattern(char, pattern[i], int_to_char):
                return False
        elif i == 4 and plate_type == PlateType.MERCOSUL:  # Special case for MERCOSUL
            if not _char_matches_pattern(char, pattern[i], int_to_char):
                return False
        else:  # Numeric positions for both formats
            if not _char_matches_pattern(char, pattern[i], char_to_int):
                return False

    return True


def format_license(text: str, plate_type: PlateType) -> tuple[str, bool]:
    """
    Format the license plate text by converting characters using the appropriate mapping dictionary.

    Args:
        text (str): License plate text.
        plate_type (PlateType): Indicates whether the plate is Mercosul or old Brazilian format.

    Returns:
        tuple[str, bool]: A tuple containing the formatted license plate text and a boolean
         indicating whether the formatting was successful.
    """
    # Primeiro, verificamos se o formato da placa está correto
    if not license_complies_format(text, plate_type):
        return text, False

    # Obtemos os padrões e dicionários de conversão para o tipo de placa especificado
    char_to_int = DICT_CHAR_TO_INT[plate_type]
    int_to_char = DICT_INT_TO_CHAR[plate_type]

    formatted_license = []

    for i, char in enumerate(text):
        if i in [0, 1, 2]:  # Comum para ambos os tipos de placa
            formatted_license.append(int_to_char.get(char, char))
        elif i == 4 and plate_type == PlateType.MERCOSUL:  # Caso especial para MERCOSUL
            formatted_license.append(int_to_char.get(char, char))
        else:  # Posições numéricas para ambos os formatos
            formatted_license.append(char_to_int.get(char, char))

    # Combina a lista de caracteres formatados em uma string
    formatted_license_str = "".join(formatted_license)

    return formatted_license_str, True


def read_license_plate(license_plate_crop: CroppedPlate, ocr: PaddleOCR) -> list[str | None]:
    """
    Read the license plate text from the given cropped image(s) and format it according to the
    plate type. Returns the first plate that is in the correct format.

    Args:
        license_plate_crop (CroppedPlate): Cropped images and the plate information.
        ocr (PaddleOCR): The OCR model to use for text recognition.

    Returns:
        list[str | None]: A list of license plate texts or None if no valid plate is found.
    """
    rgb_text = run_ocr_inference(license_plate_crop.rgb, ocr)
    gray_text = run_ocr_inference(license_plate_crop.gray, ocr)

    plates = []
    if rgb_text[0]:
        plates.append(rgb_text[1])
    if gray_text[0]:
        plates.append(gray_text[1])

    return plates


def extract_and_save_cropped_images(
    img: np.ndarray,
    results: list[Results],
    save_images: bool,
    output_dir: Path = PLATES_FOLDER_PATH,
) -> list[CroppedPlate]:
    """
    Extracts and optionally saves cropped images based on detection results.

    Args:
        img (np.ndarray): The original image from which to extract cropped images.
        results (list): Detection results containing bounding boxes and labels.
        save_images (bool): Flag indicating whether to save the cropped images.
        output_dir (Path): The directory where the cropped images will be saved.

    Returns:
        list[CroppedPlate]: A list of cropped images and the plate information.
    """
    cropped_images: list[CroppedPlate] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(results[0].boxes) > 0 and any([box.id for box in results[0].boxes]):
        for box, track_id, class_id, confidence in zip(
            results[0].boxes.xyxy.cpu(),
            results[0].boxes.id.int().cpu().tolist(),
            results[0].boxes.cls.int(),
            results[0].boxes.conf.tolist(),
        ):
            if track_id is None:
                continue

            left, top, right, bottom = map(int, box.numpy())
            cropped_img = img[top:bottom, left:right]
            license_plate_crop_gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

            if save_images:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                cropped_image_name = (
                    f"{results[0].names[class_id.item()]}_{float(confidence):.2f}".replace(".", "-")
                    + f"_{timestamp}"
                )
                cropped_image_path = output_dir / cropped_image_name

                cv2.imwrite(str(cropped_image_path) + ".jpg", cropped_img)
                cv2.imwrite(str(cropped_image_path) + "_gray.jpg", license_plate_crop_gray)

            plate_type = PlateType.MERCOSUL if class_id.item() in [0, 1] else PlateType.OLD

            cropped_plate = CroppedPlate(
                track_id=track_id,
                rgb=cropped_img,
                gray=license_plate_crop_gray,
                plate_type=plate_type,
                label=f"{track_id} {results[0].names[class_id.item()]}: {float(confidence):.2f}\n",
            )

            cropped_images.append(cropped_plate)

    return cropped_images


# Função para executar inferência de OCR com tratamento para IndexError
def run_ocr_inference(image: np.ndarray, ocr: PaddleOCR) -> tuple[bool, str | None]:
    """
    Executa a inferência de OCR na imagem especificada e retorna o texto da placa.

    Args:
        image (np.ndarray): A imagem da placa a ser processada.
        ocr (PaddleOCR): O modelo OCR para usar na inferência.

    Returns:
        tuple[bool, Optional[str]]: Uma tupla contendo um booleano indicando se a inferência
        foi bem-sucedida e o texto da placa, ou None se a inferência falhar.
    """
    try:
        # Executar OCR na imagem
        result = ocr.ocr(image, cls=True)
        logger.debug(f"Debug do resultado bruto: {result}")

        # Verificar o resultado retornado
        if result is None or len(result) == 0 or (len(result) == 1 and result[0] is None):
            logger.warning("Nenhum resultado foi retornado pelo OCR...")
            return False, None

        # Exibir resultados de inferência
        result_ocr = []
        for item in result[0]:  # Cada conjunto de texto (na moto terá 2)
            # box = item[0]  # Coordenadas da caixa delimitadora
            text_info = item[1]

            # Tratamento dos resultados
            if isinstance(text_info, tuple):
                text = text_info[0]
                confidence = text_info[1]
                logger.debug(f"Texto: {text}, Confiança: {confidence:.2f}")
            else:
                text = text_info
                logger.debug(f"Texto: {text}, Confiança: N/A")

            result_ocr.append(text)
        logger.info(f'Processei a placa: {"-".join(result_ocr)}')

        return True, "".join(result_ocr).replace("-", "")

    except IndexError:
        logger.exception(
            "Erro: Índice fora do intervalo. "
            "Verifique se o dicionário de caracteres é compatível com o modelo."
        )
        return False, None
    except Exception as e:
        logger.exception(f"Erro ao executar a inferência de OCR: {e}")
        return False, None
