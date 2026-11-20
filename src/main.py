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
        "--min-area",
        type=int,
        default=1200,
        help="Área mínima em pixels² para considerar um contorno como dado (padrão: 1200).",
    )
    parser.add_argument(
        "--min-diff",
        type=int,
        default=28,
        help="Limiar mínimo de contraste/cor (0-255); valores maiores reduzem falsos positivos de sombras (padrão: 28).",
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
    parser.add_argument(
        "--play-area",
        type=str,
        default=None,
        help="Região fixa da bandeja: 'x,y,w,h' ou polígono 'x1,y1;x2,y2;x3,y3;...'. Se omitido, permite desenhar com o mouse.",
    )
    parser.add_argument(
        "--no-roi-select",
        action="store_true",
        help="Pula a seleção interativa da área da bandeja com o mouse e usa o quadro inteiro.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Inicia no modo de visão limpa (oculta contornos da visão computacional e mostra apenas tipo e valor).",
    )
    return parser


def parse_play_area(raw: Optional[str]) -> Optional[np.ndarray]:
    """Converte 'x,y,w,h' ou 'x1,y1;x2,y2;...' em um polígono OpenCV (N, 1, 2)."""
    if not raw:
        return None
    text = raw.strip()
    if ";" in text:
        pts: list[list[int]] = []
        for token in text.split(";"):
            token = token.strip()
            if not token:
                continue
            coords = [int(c.strip()) for c in token.split(",")]
            if len(coords) != 2:
                raise ValueError(f"Vértice inválido em --play-area: '{token}'. Use 'x,y'.")
            pts.append(coords)
        if len(pts) < 3:
            raise ValueError("O polígono de --play-area precisa de pelo menos 3 vértices.")
        return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"Formato inválido para --play-area: '{raw}'. Use 'x,y,w,h' ou 'x1,y1;x2,y2;x3,y3'."
        )
    x, y, w, h = (int(p) for p in parts)
    if w <= 0 or h <= 0:
        raise ValueError(f"Largura e altura de --play-area devem ser positivas: '{raw}'.")
    rect_pts = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
    return np.array(rect_pts, dtype=np.int32).reshape((-1, 1, 2))


def mask_frame_to_polygon(
    frame: np.ndarray, polygon: np.ndarray
) -> tuple[np.ndarray, int, int, np.ndarray]:
    """Recorta o bounding box do polígono e preenche o exterior com a cor mediana interna."""
    fh, fw = frame.shape[:2]
    x, y, w, h = cv2.boundingRect(polygon)
    ax0 = max(0, x)
    ay0 = max(0, y)
    ax1 = min(fw, x + w)
    ay1 = min(fh, y + h)
    if ax1 <= ax0 or ay1 <= ay0:
        full_poly = np.array([[[0, 0]], [[fw, 0]], [[fw, fh]], [[0, fh]]], dtype=np.int32)
        return frame, 0, 0, full_poly

    crop = frame[ay0:ay1, ax0:ax1].copy()
    local_poly = polygon - np.array([[[ax0, ay0]]], dtype=np.int32)

    poly_mask = np.zeros((ay1 - ay0, ax1 - ax0), dtype=np.uint8)
    cv2.fillPoly(poly_mask, [local_poly], 255)

    inside_pixels = crop[poly_mask == 255]
    if inside_pixels.size > 0:
        if crop.ndim == 3:
            fill_color = np.median(inside_pixels, axis=0).astype(crop.dtype)
            result = np.empty_like(crop)
            result[:, :] = fill_color
        else:
            fill_val = int(np.median(inside_pixels))
            result = np.full_like(crop, fill_val)
        result[poly_mask == 255] = inside_pixels
        return result, ax0, ay0, local_poly

    return crop, ax0, ay0, local_poly


def offset_rois(rois: list[DiceROI], dx: int, dy: int) -> list[DiceROI]:
    """Desloca as coordenadas das ROIs recortadas da bandeja para o frame completo da câmera."""
    if dx == 0 and dy == 0:
        return rois
    shifted: list[DiceROI] = []
    offset_arr = np.array([[[dx, dy]]], dtype=np.int32)
    for r in rois:
        rx, ry, rw, rh = r.bbox
        poly = r.approx_polygon + offset_arr if r.approx_polygon is not None else None
        fc = (r.face_center[0] + dx, r.face_center[1] + dy) if r.face_center is not None else None
        shifted.append(
            DiceROI(
                bbox=(rx + dx, ry + dy, rw, rh),
                contour=r.contour + offset_arr if r.contour.size > 0 else r.contour,
                contour_area=r.contour_area,
                full_crop=r.full_crop,
                top_face_crop=r.top_face_crop,
                threshold_mask=r.threshold_mask,
                approx_polygon=poly,
                shape_label=r.shape_label,
                face_center=fc,
                face_radius=r.face_radius,
            )
        )
    return shifted


def select_play_area_interactive(frame: np.ndarray) -> Optional[np.ndarray]:
    """Permite desenhar um polígono com cliques do mouse para delimitar a bandeja."""
    points: list[tuple[int, int]] = []
    cursor_pos: list[Optional[tuple[int, int]]] = [None]

    def _on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()
        elif event == cv2.EVENT_MOUSEMOVE:
            cursor_pos[0] = (x, y)

    win_name = "Desenhe a Area da Bandeja"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, _on_mouse)

    try:
        while True:
            preview = frame.copy()
            if len(points) >= 3:
                poly_np = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
                overlay = preview.copy()
                cv2.fillPoly(overlay, [poly_np], (60, 180, 90))
                cv2.addWeighted(overlay, 0.25, preview, 0.75, 0, preview)
                cv2.polylines(preview, [poly_np], True, (0, 230, 120), 2, cv2.LINE_AA)
            elif len(points) == 2:
                cv2.line(preview, points[0], points[1], (0, 230, 120), 2, cv2.LINE_AA)

            if points and cursor_pos[0] is not None:
                cv2.line(preview, points[-1], cursor_pos[0], (0, 220, 255), 1, cv2.LINE_AA)
                if len(points) >= 2:
                    cv2.line(preview, cursor_pos[0], points[0], (0, 220, 255), 1, cv2.LINE_AA)

            for idx, pt in enumerate(points):
                cv2.circle(preview, pt, 5, (0, 230, 255), -1)
                cv2.putText(
                    preview,
                    str(idx + 1),
                    (pt[0] + 6, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 230, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.rectangle(preview, (0, 0), (preview.shape[1], 44), (20, 20, 20), -1)
            cv2.putText(
                preview,
                "Clique esq: add vertice | dir/Backspace: desfazer | ENTER/ESPACO: confirmar | C: cancelar",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 230, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(win_name, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32):  # ENTER ou ESPACO
                if len(points) >= 3:
                    poly_str = ";".join(f"{px},{py}" for px, py in points)
                    print(f"[INFO] Poligono da bandeja definido ({len(points)} vertices): --play-area \"{poly_str}\"")
                    return np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            elif key in (ord("c"), ord("C"), 27):  # C ou ESC
                print("[INFO] Selecao cancelada; usando o quadro inteiro.")
                return None
            elif key in (8, ord("z"), ord("Z")) and points:  # Backspace ou Z
                points.pop()
    finally:
        cv2.destroyWindow(win_name)


def extract_dice_type(shape_label: str) -> str:
    """Extrai o tipo resumido do dado (ex: 'd6', 'd8/d10', 'd12/d20') a partir de shape_label."""
    if "(" in shape_label and ")" in shape_label:
        return shape_label.split("(", 1)[1].split(")", 1)[0].strip()
    return shape_label.strip() or "Dado"


def draw_hud(
    frame: np.ndarray,
    status_text: str,
    status_color: tuple[int, int, int],
    rois: list[DiceROI],
    ocr_results: list[OCRResult],
    play_polygon: Optional[np.ndarray] = None,
    clean_view: bool = False,
) -> np.ndarray:
    """Desenha a área da bandeja, bounding boxes, polígonos e cabeçalho de status (ou apenas tipo+valor no modo limpo)."""
    canvas = frame.copy()

    if not clean_view and play_polygon is not None and len(play_polygon) >= 3:
        cv2.polylines(canvas, [play_polygon], True, (90, 200, 120), 2, cv2.LINE_AA)

    for idx, roi in enumerate(rois):
        x, y, w, h = roi.bbox
        ocr_result = ocr_results[idx] if idx < len(ocr_results) else None
        dice_type = extract_dice_type(roi.shape_label)

        if clean_view:
            if ocr_result is not None and ocr_result.is_valid:
                clean_label = f"{dice_type}: {ocr_result.value}"
                pill_color = (40, 220, 80)
            else:
                clean_label = f"{dice_type}: ..."
                pill_color = (200, 200, 200)

            cx, cy = roi.face_center if roi.face_center is not None else (x + w // 2, y + h // 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(clean_label, font, 0.72, 2)
            px0 = max(8, cx - tw // 2 - 10)
            py0 = max(64, cy - th // 2 - 8)
            px1 = min(canvas.shape[1] - 8, px0 + tw + 20)
            py1 = min(canvas.shape[0] - 8, py0 + th + 16)
            overlay_pill = canvas.copy()
            cv2.rectangle(overlay_pill, (px0, py0), (px1, py1), (18, 18, 18), -1)
            cv2.addWeighted(overlay_pill, 0.78, canvas, 0.22, 0, canvas)
            cv2.putText(
                canvas,
                clean_label,
                (px0 + 10, py1 - 8),
                font,
                0.72,
                pill_color,
                2,
                cv2.LINE_AA,
            )
            continue

        box_color = status_color
        if ocr_result is not None:
            box_color = (40, 220, 80) if ocr_result.is_valid else (0, 140, 255)

        cv2.rectangle(canvas, (x, y), (x + w, y + h), box_color, 2)
        if roi.approx_polygon is not None and len(roi.approx_polygon) >= 3:
            cv2.polylines(canvas, [roi.approx_polygon], True, (255, 255, 0), 2, cv2.LINE_AA)

        # Destaca o círculo central exato calculado a partir do centroide e raio inscrito do polígono
        if roi.face_center is not None and roi.face_radius > 0:
            cx, cy = roi.face_center
            radius = roi.face_radius
        else:
            cx, cy = x + w // 2, y + h // 2
            radius = max(6, int(min(w, h) * 0.28))
        cv2.circle(canvas, (cx, cy), radius, (0, 215, 255), 2)
        cv2.circle(canvas, (cx, cy), 3, (0, 215, 255), -1)

        shape_tag = f" [{roi.shape_label}]" if roi.shape_label else ""
        label = f"#{idx + 1}{shape_tag}"
        if ocr_result is not None:
            if ocr_result.is_valid:
                label = f"#{idx + 1}{shape_tag}: {ocr_result.value} ({ocr_result.confidence * 100:.0f}%)"
            else:
                label = f"#{idx + 1}{shape_tag}: ?"

        cv2.putText(
            canvas,
            label,
            (x, max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            box_color,
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
        0.78,
        status_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "'v' limpo | 'a' area | 'b' fundo | 'm' stats | 'r' ler | 'q' sair",
        (max(16, canvas.shape[1] - 490), 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
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


def _hconcat_tiles(images: list[np.ndarray], target_height: int = 160) -> Optional[np.ndarray]:
    """Empilha recortes lado a lado na mesma altura para visualização em modo debug."""
    valid = [img for img in images if img is not None and img.size > 0]
    if not valid:
        return None
    resized: list[np.ndarray] = []
    for img in valid[:6]:
        h, w = img.shape[:2]
        scale = target_height / max(1, h)
        new_w = max(8, int(round(w * scale)))
        resized.append(cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA))
    return cv2.hconcat(resized)


def show_debug_windows(
    rois: list[DiceROI],
    ocr_results: list[OCRResult],
    motion_mask: Optional[np.ndarray] = None,
) -> None:
    """Exibe janelas auxiliares de depuração com as máscaras de threshold, movimento e ROIs."""
    if motion_mask is not None:
        cv2.imshow("Debug - Motion Diff", motion_mask)

    if rois:
        cv2.imshow("Debug - Foreground Mask", rois[0].threshold_mask)
        roi_strip = _hconcat_tiles([r.top_face_crop for r in rois])
        if roi_strip is not None:
            cv2.imshow("Debug - Top Face ROI", roi_strip)

    if ocr_results:
        clahe_strip = _hconcat_tiles([res.preprocessed_image for res in ocr_results])
        if clahe_strip is not None:
            cv2.imshow("Debug - CLAHE Preprocessed", clahe_strip)


def _format_results_summary(
    ocr_results: list[OCRResult],
    rois: Optional[list[DiceROI]] = None,
    clean_view: bool = False,
) -> tuple[str, tuple[int, int, int]]:
    """Formata o texto e a cor da barra de status para um ou múltiplos dados lidos."""
    valid_pairs = [
        (idx, res)
        for idx, res in enumerate(ocr_results)
        if res.is_valid and res.value is not None
    ]
    if not valid_pairs:
        return "Resultado: Nao identificado", (0, 140, 255)

    if clean_view:
        items: list[str] = []
        for idx, res in valid_pairs:
            dice_type = (
                extract_dice_type(rois[idx].shape_label)
                if rois is not None and idx < len(rois)
                else "Dado"
            )
            items.append(f"{dice_type}: {res.value}")
        return " | ".join(items), (40, 220, 80)

    if len(valid_pairs) == 1 and len(ocr_results) == 1:
        _, res = valid_pairs[0]
        return f"Resultado: {res.value} ({res.confidence * 100:.1f}%)", (40, 220, 80)

    values = [int(res.value) for _, res in valid_pairs if res.value is not None]
    values_str = ", ".join(str(v) for v in values)
    return f"Dados: [{values_str}] | Soma: {sum(values)}", (40, 220, 80)


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

    rois = segmenter.detect_dice_rois(frame)
    if not rois:
        # Fallback: se a imagem já for um recorte fechado do dado, usa o frame inteiro
        top_crop = segmenter.extract_top_face_crop(frame)
        rois = [
            DiceROI(
                bbox=(0, 0, frame.shape[1], frame.shape[0]),
                contour=np.array([]),
                contour_area=float(frame.shape[0] * frame.shape[1]),
                full_crop=frame,
                top_face_crop=top_crop,
                threshold_mask=segmenter.compute_foreground_mask(frame),
            )
        ]

    t0 = time.monotonic()
    ocr_results = [ocr_engine.recognize_top_face(roi.top_face_crop) for roi in rois]
    ocr_elapsed_ms = (time.monotonic() - t0) * 1000.0

    for idx, res in enumerate(ocr_results, start=1):
        if res.is_valid:
            print(
                f"[OK] Dado #{idx} -> Valor: {res.value} | "
                f"Confiança: {res.confidence * 100:.1f}% | "
                f"Ângulo: {res.best_angle_deg}°"
            )
        else:
            print(f"[AVISO] Dado #{idx} -> Nenhum valor válido (1..20) reconhecido.")

    valid_vals = [int(r.value) for r in ocr_results if r.is_valid and r.value is not None]
    if len(valid_vals) > 1:
        print(f"[OK] Soma ({len(valid_vals)} dados): {sum(valid_vals)} | Tempo OCR total: {ocr_elapsed_ms:.0f} ms")
    else:
        print(f"[INFO] Tempo OCR total: {ocr_elapsed_ms:.0f} ms")

    status_text, status_color = _format_results_summary(ocr_results)

    if not headless:
        hud = draw_hud(frame, status_text, status_color, rois, ocr_results)
        if debug:
            total_cands = sum(len(r.candidates) for r in ocr_results)
            draw_stats_overlay(
                hud,
                [
                    f"Dispositivo: {hardware_info}",
                    f"Frame: {frame.shape[1]}x{frame.shape[0]} | Dados: {len(rois)}",
                    f"OCR total: {ocr_elapsed_ms:.0f} ms | Candidatos: {total_cands}",
                ],
            )
            show_debug_windows(rois, ocr_results)
        cv2.imshow("Dice Vision Pro", hud)
        print("Pressione 'q' ou qualquer tecla na janela da imagem para encerrar.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return 0


def run_stream_mode(
    source: int | str,
    config: DiceConfig,
    debug: bool = False,
    play_polygon: Optional[np.ndarray] = None,
    interactive_roi: bool = True,
    clean_view: bool = False,
) -> int:
    """Executa o loop principal em tempo real sobre webcam ou arquivo de vídeo."""
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir a fonte de vídeo: {source}", file=sys.stderr)
        return 1

    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)

    if play_polygon is None and interactive_roi:
        # Descarta os primeiros frames para estabilizar exposição automática antes da seleção
        preview_frame: Optional[np.ndarray] = None
        for _ in range(5):
            ok_p, f_p = cap.read()
            if ok_p and f_p is not None:
                preview_frame = f_p
        if preview_frame is not None:
            play_polygon = select_play_area_interactive(preview_frame)

    motion_detector = MotionStabilityDetector(config)
    segmenter = DiceSegmenter(config)
    ocr_engine = DiceOCREngine(config)

    hardware_info = get_hardware_info(config.use_gpu)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    locked_ocr_results: list[OCRResult] = []
    ocr_already_triggered_for_stop: bool = False
    show_stats_overlay: bool = debug
    last_ocr_ms: Optional[float] = None
    smoothed_fps: float = 0.0
    prev_frame_time = time.monotonic()
    bg_reference: Optional[np.ndarray] = None

    print(
        f"[INFO] Dice Vision Pro iniciado | {hardware_info} | "
        f"Resolucao: {actual_w}x{actual_h} | "
        f"Estabilidade: {config.stability_duration_sec}s | Passo: {config.rotation_step_deg}°"
    )
    print(
        "[INFO] Atalhos na janela: 'a' = desenhar area | 'b' = calibrar fundo vazio | "
        "'m' = stats | 'r' = ler | 'q'/ESC = sair"
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

            if play_polygon is not None and len(play_polygon) >= 3:
                work_frame, ax0, ay0, local_poly = mask_frame_to_polygon(frame, play_polygon)
            else:
                ax0, ay0 = 0, 0
                work_frame = frame
                local_poly = None

            stability = motion_detector.update(work_frame)
            bg_crop = bg_reference if (bg_reference is not None and bg_reference.shape == work_frame.shape) else None
            local_rois = segmenter.detect_dice_rois(work_frame, background=bg_crop)
            if local_poly is not None:
                local_rois = [
                    r
                    for r in local_rois
                    if cv2.pointPolygonTest(
                        local_poly,
                        (r.bbox[0] + r.bbox[2] / 2.0, r.bbox[1] + r.bbox[3] / 2.0),
                        False,
                    )
                    >= 0
                ]
            rois = offset_rois(local_rois, ax0, ay0)

            if stability.is_moving:
                locked_ocr_results = []
                ocr_already_triggered_for_stop = False
                status_text = "Rolando..."
                status_color = (50, 80, 255)  # Vermelho/Laranja em BGR
            elif not stability.is_stable:
                locked_ocr_results = []
                ocr_already_triggered_for_stop = False
                progress = min(
                    100.0,
                    (stability.stable_elapsed_sec / max(0.01, config.stability_duration_sec))
                    * 100.0,
                )
                status_text = f"Estabilizando... ({progress:.0f}%)"
                status_color = (0, 215, 255)  # Amarelo em BGR
            else:
                if not rois:
                    status_text = "Aguardando dado..."
                    status_color = (180, 180, 180)
                else:
                    if not ocr_already_triggered_for_stop:
                        t_ocr = time.monotonic()
                        locked_ocr_results = [
                            ocr_engine.recognize_top_face(roi.top_face_crop) for roi in rois
                        ]
                        last_ocr_ms = (time.monotonic() - t_ocr) * 1000.0
                        ocr_already_triggered_for_stop = True

                        for idx, (roi, res) in enumerate(zip(rois, locked_ocr_results), start=1):
                            if res.is_valid:
                                print(
                                    f"[OCR] Dado #{idx}: {res.value} | "
                                    f"Confianca: {res.confidence * 100:.1f}% | "
                                    f"Angulo: {res.best_angle_deg}°"
                                )
                            else:
                                print(
                                    f"[OCR] Dado #{idx}: Nao identificado | "
                                    f"ROI: {roi.bbox[2]}x{roi.bbox[3]}"
                                )

                        valid_vals = [
                            int(r.value)
                            for r in locked_ocr_results
                            if r.is_valid and r.value is not None
                        ]
                        if len(valid_vals) > 1:
                            print(
                                f"[OCR] Soma ({len(valid_vals)} dados): {sum(valid_vals)} | "
                                f"Tempo total: {last_ocr_ms:.0f} ms"
                            )
                        else:
                            print(f"[OCR] Tempo total: {last_ocr_ms:.0f} ms")

                    status_text, status_color = _format_results_summary(
                        locked_ocr_results, rois=rois, clean_view=clean_view
                    )

            hud = draw_hud(
                frame,
                status_text,
                status_color,
                rois,
                locked_ocr_results,
                play_polygon=play_polygon,
                clean_view=clean_view,
            )
            if show_stats_overlay:
                if rois:
                    total_area = sum(int(r.contour_area) for r in rois)
                    roi_desc = f"{len(rois)} dado(s) | Area total: {total_area} px"
                else:
                    roi_desc = "Nenhum contorno"

                if last_ocr_ms is not None and locked_ocr_results:
                    total_cands = sum(len(r.candidates) for r in locked_ocr_results)
                    ocr_desc = f"{last_ocr_ms:.0f} ms ({total_cands} cand.)"
                else:
                    ocr_desc = "Aguardando"

                draw_stats_overlay(
                    hud,
                    [
                        f"Dispositivo: {hardware_info}",
                        f"FPS: {smoothed_fps:.1f} | Frame: {frame.shape[1]}x{frame.shape[0]}",
                        f"Movimento: {stability.motion_ratio * 100:.2f}% (lim: {config.motion_changed_ratio_threshold * 100:.2f}%)",
                        f"ROIs: {roi_desc}",
                        f"Ultimo OCR: {ocr_desc} | Passo: {config.rotation_step_deg} deg",
                    ],
                )

            cv2.imshow("Dice Vision Pro", hud)

            if debug:
                show_debug_windows(rois, locked_ocr_results, stability.diff_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("r"):
                # Permite forçar uma nova leitura manual (inclusive em dados já parados)
                motion_detector.clear_motion_history()
                ocr_already_triggered_for_stop = False
            if key in (ord("m"), ord("M")):
                show_stats_overlay = not show_stats_overlay
                estado = "ATIVADO" if show_stats_overlay else "DESATIVADO"
                print(f"[INFO] Overlay de estatisticas: {estado}")
            if key in (ord("a"), ord("A")):
                play_polygon = select_play_area_interactive(frame)
                bg_reference = None
                motion_detector.reset()
                locked_ocr_results = []
                ocr_already_triggered_for_stop = False
            if key in (ord("v"), ord("V")):
                clean_view = not clean_view
                estado = "LIMPO" if clean_view else "COMPLETO"
                print(f"[INFO] Modo de visualizacao: {estado}")
            if key in (ord("b"), ord("B")):
                if bg_reference is None:
                    bg_reference = work_frame.copy()
                    print("[INFO] Fundo vazio da bandeja calibrado (sombras fixas ignoradas).")
                else:
                    bg_reference = None
                    print("[INFO] Calibracao de fundo vazio removida.")
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
        min_contour_area=args.min_area,
        min_foreground_diff=args.min_diff,
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
    play_polygon = parse_play_area(args.play_area)
    return run_stream_mode(
        source=source,
        config=config,
        debug=args.debug,
        play_polygon=play_polygon,
        interactive_roi=not args.no_roi_select,
        clean_view=args.clean,
    )


if __name__ == "__main__":
    sys.exit(main())
