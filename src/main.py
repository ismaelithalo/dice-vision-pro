"""Loop principal e interface visual OpenCV do Dice Vision Pro (MVP)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DiceConfig, detect_gpu_available, get_hardware_info
from src.detector import DiceROI, DiceSegmenter, MotionStabilityDetector
from src.ocr_engine import DiceOCREngine, OCRResult


def build_arg_parser() -> argparse.ArgumentParser:
    """Constrói o parser de argumentos de linha de comando (CLI)."""
    parser = argparse.ArgumentParser(
        prog="dice-vision-pro",
        description="Leitor em tempo real da face superior de dados poliédricos (d6, d8, d10, d12, d20) via OpenCV + EasyOCR.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Índice da webcam utilizada na captura em tempo real (padrão: 0).",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Caminho para uma imagem estática para validação/teste offline sem câmera.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Caminho opcional para um arquivo de vídeo gravado.",
    )
    parser.add_argument(
        "--stability-time",
        type=float,
        default=0.6,
        help="Tempo mínimo (em segundos) que o dado deve permanecer imóvel antes de disparar o OCR (padrão: 0.6).",
    )
    parser.add_argument(
        "--rotation-step",
        type=int,
        default=30,
        help="Passo angular (em graus) para a varredura rotacionada do EasyOCR de 0 a 360 graus (padrão: 30).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Força o uso de GPU (CUDA) no EasyOCR se disponível.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Força o uso de CPU no EasyOCR mesmo quando CUDA estiver disponível.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Exibe janelas intermediárias (Threshold, ROI da Face Superior e Pré-processamento CLAHE).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Executa sem abrir janelas gráficas (útil para testes automatizados com --image).",
    )
    return parser


def draw_hud(
    frame: np.ndarray,
    status_text: str,
    status_color: tuple[int, int, int],
    roi: Optional[DiceROI],
    ocr_result: Optional[OCRResult],
) -> np.ndarray:
    """Desenha o bounding box do dado, a região central da face superior e o cabeçalho de status."""
    canvas = frame.copy()

    if roi is not None:
        x, y, w, h = roi.bbox
        cv2.rectangle(canvas, (x, y), (x + w, y + h), status_color, 2)

        # Destaca o círculo central correspondente à face superior analisada
        cx, cy = x + w // 2, y + h // 2
        radius = max(6, int(min(w, h) * 0.32))
        cv2.circle(canvas, (cx, cy), radius, (255, 215, 0), 1)

        label = f"Dado ROI ({w}x{h})"
        if ocr_result is not None and ocr_result.is_valid:
            label = f"Face: {ocr_result.value} ({ocr_result.confidence * 100:.0f}% @ {ocr_result.best_angle_deg}deg)"

        cv2.putText(
            canvas,
            label,
            (x, max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
            cv2.LINE_AA,
        )

    # Faixa superior semitransparente para o status geral
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 56), (20, 20, 20), thickness=-1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

    cv2.putText(
        canvas,
        status_text,
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        status_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "'m' stats | 'r' reler | 'q' sair",
        (max(16, canvas.shape[1] - 290), 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_stats_overlay(canvas: np.ndarray, stats_lines: list[str]) -> np.ndarray:
    """Desenha um painel semitransparente com informações de GPU/CPU e estatísticas de debug."""
    if not stats_lines:
        return canvas

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 22
    padding_x = 12
    padding_y = 10

    max_text_w = max(cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in stats_lines)
    panel_w = min(canvas.shape[1] - 24, max_text_w + padding_x * 2)
    panel_h = len(stats_lines) * line_height + padding_y * 2

    x0, y0 = 12, 68
    x1 = min(canvas.shape[1] - 12, x0 + panel_w)
    y1 = min(canvas.shape[0] - 12, y0 + panel_h)
    if x1 <= x0 or y1 <= y0:
        return canvas

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (15, 15, 15), thickness=-1)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (90, 90, 90), thickness=1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

    for idx, line in enumerate(stats_lines):
        ty = y0 + padding_y + (idx + 1) * line_height - 6
        if ty >= y1:
            break
        color = (110, 235, 180) if idx == 0 else (230, 230, 230)
        cv2.putText(canvas, line, (x0 + padding_x, ty), font, font_scale, color, thickness, cv2.LINE_AA)

    return canvas


def show_debug_windows(
    roi: Optional[DiceROI],
    ocr_result: Optional[OCRResult],
    motion_mask: Optional[np.ndarray] = None,
) -> None:
    """Exibe janelas auxiliares de depuração com as máscaras de threshold, movimento e ROI."""
    if motion_mask is not None:
        cv2.imshow("Debug - Motion Diff", motion_mask)

    if roi is not None:
        cv2.imshow("Debug - Foreground Mask", roi.threshold_mask)
        cv2.imshow("Debug - Top Face ROI", roi.top_face_crop)

    if ocr_result is not None:
        cv2.imshow("Debug - CLAHE Preprocessed", ocr_result.preprocessed_image)


def run_static_image_mode(
    image_path: str,
    config: DiceConfig,
    debug: bool = False,
    headless: bool = False,
) -> int:
    """Processa uma única imagem estática para validação sem webcam."""
    path = Path(image_path)
    if not path.is_file():
        print(f"[ERRO] Arquivo de imagem não encontrado: {image_path}", file=sys.stderr)
        return 1

    frame = cv2.imread(str(path))
    if frame is None:
        print(f"[ERRO] Não foi possível decodificar a imagem: {image_path}", file=sys.stderr)
        return 1

    hardware_info = get_hardware_info(config.use_gpu)
    print(f"[INFO] Dispositivo ativo: {hardware_info}")

    segmenter = DiceSegmenter(config)
    ocr_engine = DiceOCREngine(config)

    roi = segmenter.detect_dice_roi(frame)
    if roi is None:
        # Fallback: se a imagem já for um recorte fechado do dado, usa o frame inteiro
        top_crop = segmenter.extract_top_face_crop(frame)
        roi = DiceROI(
            bbox=(0, 0, frame.shape[1], frame.shape[0]),
            contour=np.array([]),
            contour_area=float(frame.shape[0] * frame.shape[1]),
            full_crop=frame,
            top_face_crop=top_crop,
            threshold_mask=segmenter.compute_foreground_mask(frame),
        )

    t0 = time.monotonic()
    ocr_result = ocr_engine.recognize_top_face(roi.top_face_crop)
    ocr_elapsed_ms = (time.monotonic() - t0) * 1000.0
    if ocr_result.is_valid:
        status_text = f"Resultado: {ocr_result.value} ({ocr_result.confidence * 100:.1f}%)"
        status_color = (40, 220, 80)
        print(
            f"[OK] Dado reconhecido -> Valor: {ocr_result.value} | "
            f"Confiança: {ocr_result.confidence * 100:.1f}% | "
            f"Ângulo: {ocr_result.best_angle_deg}° | Tempo OCR: {ocr_elapsed_ms:.0f} ms"
        )
    else:
        status_text = "Resultado: Indefinido (0.0%)"
        status_color = (0, 165, 255)
        print(f"[AVISO] Nenhum valor válido (1..20) foi reconhecido na imagem (Tempo OCR: {ocr_elapsed_ms:.0f} ms).")

    if not headless:
        hud = draw_hud(frame, status_text, status_color, roi, ocr_result)
        if debug:
            draw_stats_overlay(
                hud,
                [
                    f"Dispositivo: {hardware_info}",
                    f"Frame: {frame.shape[1]}x{frame.shape[0]} | ROI Area: {int(roi.contour_area)} px",
                    f"OCR: {ocr_elapsed_ms:.0f} ms | Candidatos: {len(ocr_result.candidates)}",
                ],
            )
            show_debug_windows(roi, ocr_result)
        cv2.imshow("Dice Vision Pro", hud)
        print("Pressione 'q' ou qualquer tecla na janela da imagem para encerrar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


def run_stream_mode(
    source: int | str,
    config: DiceConfig,
    debug: bool = False,
) -> int:
    """Executa o loop principal em tempo real sobre webcam ou arquivo de vídeo."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir a fonte de vídeo: {source}", file=sys.stderr)
        return 1

    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    motion_detector = MotionStabilityDetector(config)
    segmenter = DiceSegmenter(config)
    ocr_engine = DiceOCREngine(config)

    hardware_info = get_hardware_info(config.use_gpu)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    locked_ocr_result: Optional[OCRResult] = None
    ocr_already_triggered_for_stop: bool = False
    show_stats_overlay: bool = debug
    last_ocr_ms: Optional[float] = None
    smoothed_fps: float = 0.0
    prev_frame_time = time.monotonic()

    print(
        f"[INFO] Dice Vision Pro iniciado | {hardware_info} | "
        f"Resolucao: {actual_w}x{actual_h} | "
        f"Estabilidade: {config.stability_duration_sec}s | Passo: {config.rotation_step_deg}°"
    )
    print(
        "[INFO] Atalhos na janela: 'm' = alternar overlay de stats | "
        "'r' = forcar nova leitura | 'q'/ESC = sair"
    )

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            now = time.monotonic()
            dt = max(1e-4, now - prev_frame_time)
            prev_frame_time = now
            inst_fps = 1.0 / dt
            smoothed_fps = inst_fps if smoothed_fps == 0.0 else (0.9 * smoothed_fps + 0.1 * inst_fps)

            stability = motion_detector.update(frame)
            roi = segmenter.detect_dice_roi(frame)

            if stability.is_moving:
                locked_ocr_result = None
                ocr_already_triggered_for_stop = False
                status_text = "Rolando..."
                status_color = (50, 80, 255)  # Vermelho/Laranja em BGR
            elif not stability.is_stable:
                locked_ocr_result = None
                ocr_already_triggered_for_stop = False
                progress = min(
                    100.0,
                    (stability.stable_elapsed_sec / max(0.01, config.stability_duration_sec))
                    * 100.0,
                )
                status_text = f"Estabilizando... ({progress:.0f}%)"
                status_color = (0, 215, 255)  # Amarelo em BGR
            else:
                if roi is None:
                    status_text = "Aguardando dado..."
                    status_color = (180, 180, 180)
                else:
                    if not ocr_already_triggered_for_stop:
                        t_ocr = time.monotonic()
                        locked_ocr_result = ocr_engine.recognize_top_face(roi.top_face_crop)
                        last_ocr_ms = (time.monotonic() - t_ocr) * 1000.0
                        ocr_already_triggered_for_stop = True

                        if locked_ocr_result.is_valid:
                            print(
                                f"[OCR] Valor: {locked_ocr_result.value} | "
                                f"Confianca: {locked_ocr_result.confidence * 100:.1f}% | "
                                f"Angulo: {locked_ocr_result.best_angle_deg}° | "
                                f"Tempo: {last_ocr_ms:.0f} ms | "
                                f"Candidatos: {len(locked_ocr_result.candidates)}"
                            )
                        else:
                            print(
                                f"[OCR] Nenhum valor valido (1..20) | "
                                f"Tempo: {last_ocr_ms:.0f} ms | ROI: {roi.bbox[2]}x{roi.bbox[3]}"
                            )

                    if locked_ocr_result is not None and locked_ocr_result.is_valid:
                        conf_pct = locked_ocr_result.confidence * 100.0
                        status_text = f"Resultado: {locked_ocr_result.value} ({conf_pct:.1f}%)"
                        status_color = (40, 220, 80)  # Verde em BGR
                    else:
                        status_text = "Resultado: Nao identificado"
                        status_color = (0, 140, 255)

            hud = draw_hud(frame, status_text, status_color, roi, locked_ocr_result)
            if show_stats_overlay:
                roi_desc = (
                    f"Area: {int(roi.contour_area)} px | Box: {roi.bbox[2]}x{roi.bbox[3]}"
                    if roi is not None
                    else "Nenhum contorno"
                )
                ocr_desc = (
                    f"{last_ocr_ms:.0f} ms ({len(locked_ocr_result.candidates)} cand.)"
                    if last_ocr_ms is not None and locked_ocr_result is not None
                    else "Aguardando"
                )
                draw_stats_overlay(
                    hud,
                    [
                        f"Dispositivo: {hardware_info}",
                        f"FPS: {smoothed_fps:.1f} | Frame: {frame.shape[1]}x{frame.shape[0]}",
                        f"Movimento: {stability.motion_ratio * 100:.2f}% (lim: {config.motion_changed_ratio_threshold * 100:.2f}%)",
                        f"ROI: {roi_desc}",
                        f"Ultimo OCR: {ocr_desc} | Passo: {config.rotation_step_deg} deg",
                    ],
                )

            cv2.imshow("Dice Vision Pro", hud)

            if debug:
                show_debug_windows(roi, locked_ocr_result, stability.diff_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("r"):
                # Permite forçar uma nova leitura manual com a tecla 'r'
                ocr_already_triggered_for_stop = False
            if key in (ord("m"), ord("M")):
                show_stats_overlay = not show_stats_overlay
                estado = "ATIVADO" if show_stats_overlay else "DESATIVADO"
                print(f"[INFO] Overlay de estatisticas: {estado}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Ponto de entrada principal da CLI."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    use_gpu = detect_gpu_available()
    if args.gpu:
        use_gpu = True
    if args.cpu:
        use_gpu = False

    config = DiceConfig(
        camera_index=args.camera,
        stability_duration_sec=args.stability_time,
        rotation_step_deg=args.rotation_step,
        use_gpu=use_gpu,
    )

    if args.image:
        return run_static_image_mode(
            image_path=args.image,
            config=config,
            debug=args.debug,
            headless=args.headless,
        )

    source: int | str = args.video if args.video else args.camera
    return run_stream_mode(source=source, config=config, debug=args.debug)


if __name__ == "__main__":
    sys.exit(main())
