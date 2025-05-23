"""Módulo para verificar a resolução de um stream RTSP."""

import time

import cv2

from infrastructure.logging.log_config import get_logger

logger = get_logger(__name__)


def verificar_stream_rtsp(
    rtsp_url: str, timeout: int = 10, mostrar_video: bool = True, max_frames: int = 10
) -> tuple[int, int] | None:
    if not rtsp_url or not rtsp_url.startswith("rtsp://"):
        logger.error("URL RTSP inválida ou vazia")
        return None

    logger.info(f"[RESOLUTION_CHECKER] Conectando ao stream RTSP: {rtsp_url}")

    try:
        cap = cv2.VideoCapture(rtsp_url)
        inicio = time.time()

        while not cap.isOpened():
            if time.time() - inicio > timeout:
                logger.error(f"[RESOLUTION_CHECKER] Timeout de {timeout}s ao abrir.")
                return None
            time.sleep(0.1)

        # Coleta metadados
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_name = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

        # Medição do primeiro frame
        t0 = time.time()
        ret, frame = cap.read()
        delay_ms = (time.time() - t0) * 1000

        if not ret or frame is None:
            logger.error("[RESOLUTION_CHECKER] Não foi possível ler o primeiro frame.")
            cap.release()
            return None

        altura, largura = frame.shape[:2]
        logger.info(f"[RESOLUTION_CHECKER] Resolução: {largura}×{altura}")
        logger.info(
            f"[RESOLUTION_CHECKER] FPS reportado: {fps:.2f}"
            if fps > 0
            else "[RESOLUTION_CHECKER] FPS não informado"
        )
        logger.info(f"[RESOLUTION_CHECKER] Codec: {codec_name}")
        logger.info(
            f"[RESOLUTION_CHECKER] Total de frames: {int(total_frames)}"
            if total_frames > 0
            else "[RESOLUTION_CHECKER] Total de frames: N/A (stream ao vivo?)"
        )
        logger.info(f"[RESOLUTION_CHECKER] Delay do frame inicial: {delay_ms:.1f} ms")

        if mostrar_video:
            logger.info("[RESOLUTION_CHECKER] Exibindo vídeo. Pressione 'q' para sair.")
            count = 1
            while ret and count < max_frames:
                cv2.imshow("Stream RTSP", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                ret, frame = cap.read()
                if ret:  # Só incrementa se obteve um frame válido
                    count += 1
            cv2.destroyAllWindows()

        cap.release()
        return largura, altura

    except Exception as e:
        logger.error(f"[RESOLUTION_CHECKER] Erro inesperado: {str(e)}")
        if "cap" in locals():
            cap.release()
        cv2.destroyAllWindows()
        return None


if __name__ == "__main__":
    rtsp_url = "rtsp://807e9439d5ca.entrypoint.cloud.wowza.com:1935/app-rC94792j/068b9c9a_stream2"
    resultado = verificar_stream_rtsp(rtsp_url)
    if resultado:
        logger.info(f"Resolução do stream: {resultado[0]}x{resultado[1]}")
    else:
        logger.error("Falha ao obter dados do stream.")
