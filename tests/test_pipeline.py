"""Testes unitários do pipeline de visão computacional e OCR do Dice Vision Pro."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_CONFIG, DiceConfig, detect_gpu_available, get_hardware_info
from src.detector import DiceSegmenter, MotionStabilityDetector
from src.main import (
    build_arg_parser,
    draw_hud,
    draw_stats_overlay,
    mask_frame_to_polygon,
    offset_rois,
    parse_play_area,
)
from src.ocr_engine import DiceOCREngine, OCRResult


class FakeEasyOCRReader:
    """Mock leve do EasyOCR Reader para validar a lógica de varredura angular e filtro de intervalo."""

    def __init__(self, responses_by_call: list[list[tuple[Any, str, float]]]) -> None:
        self.responses_by_call = responses_by_call
        self.call_count = 0

    def readtext(
        self,
        image: np.ndarray,
        allowlist: str = "0123456789",
        detail: int = 1,
        paragraph: bool = False,
    ) -> list[tuple[Any, str, float]]:
        assert image.ndim == 2
        assert allowlist == "0123456789"
        idx = self.call_count
        self.call_count += 1
        if idx < len(self.responses_by_call):
            return self.responses_by_call[idx]
        return []


def _create_synthetic_dice_frame(text: str = "20") -> np.ndarray:
    """Gera um frame sintético 640x480 com fundo uniforme escuro e um dado claro centralizado."""
    frame = np.full((480, 640, 3), 25, dtype=np.uint8)
    # Desenha um polígono octogonal simulando o corpo do dado no centro
    center_x, center_y = 320, 240
    radius = 85
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    pts = np.array(
        [
            [int(center_x + radius * np.cos(a)), int(center_y + radius * np.sin(a))]
            for a in angles
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [pts], (235, 235, 235))
    cv2.putText(
        frame,
        text,
        (center_x - 38, center_y + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (20, 20, 20),
        4,
        cv2.LINE_AA,
    )
    return frame


def test_config_defaults_and_gpu_detection() -> None:
    """Verifica se a configuração padrão e a detecção de GPU retornam tipos consistentes."""
    assert isinstance(detect_gpu_available(), bool)
    assert DEFAULT_CONFIG.stability_duration_sec == pytest.approx(0.6)
    assert DEFAULT_CONFIG.rotation_step_deg == 30
    assert DEFAULT_CONFIG.min_dice_value == 1
    assert DEFAULT_CONFIG.max_dice_value == 20
    assert DEFAULT_CONFIG.ocr_allowlist == "0123456789"


def test_motion_stability_detector_transitions() -> None:
    """Verifica a transição de estados: frame inicial -> movimento -> estabilizando -> estável após 0.6s."""
    config = DiceConfig(stability_duration_sec=0.6)
    detector = MotionStabilityDetector(config)

    frame_a = _create_synthetic_dice_frame("12")
    frame_b = frame_a.copy()
    # Simula movimento significativo deslocando um bloco branco
    cv2.rectangle(frame_b, (50, 50), (260, 260), (255, 255, 255), thickness=-1)

    status_0 = detector.update(frame_a, timestamp=10.0)
    assert not status_0.is_moving
    assert not status_0.is_stable

    # Movimento brusco entre t=10.0 e t=10.2
    status_move = detector.update(frame_b, timestamp=10.2)
    assert status_move.is_moving
    assert not status_move.is_stable
    assert status_move.stable_elapsed_sec == pytest.approx(0.0)

    # Dado para em t=10.3 (inicia contagem de estabilidade)
    status_stop_start = detector.update(frame_b, timestamp=10.3)
    assert not status_stop_start.is_moving
    assert not status_stop_start.is_stable
    assert status_stop_start.stable_elapsed_sec == pytest.approx(0.0)

    # Ainda estabilizando em t=10.7 (0.4s < 0.6s)
    status_wait = detector.update(frame_b, timestamp=10.7)
    assert not status_wait.is_moving
    assert not status_wait.is_stable
    assert status_wait.stable_elapsed_sec == pytest.approx(0.4)

    # Estável em t=10.95 (0.65s >= 0.6s)
    status_stable = detector.update(frame_b, timestamp=10.95)
    assert not status_stable.is_moving
    assert status_stable.is_stable
    assert status_stable.stable_elapsed_sec >= 0.6


def test_dice_segmenter_detects_roi_and_crops_top_face() -> None:
    """Garante que o segmentador isola o dado sobre fundo uniforme e recorta a região central."""
    config = DiceConfig(center_crop_ratio=0.65)
    segmenter = DiceSegmenter(config)
    frame = _create_synthetic_dice_frame("20")

    roi = segmenter.detect_dice_roi(frame)
    assert roi is not None
    x, y, w, h = roi.bbox
    assert w > 80 and h > 80
    assert roi.contour_area >= config.min_contour_area

    # O recorte da face superior deve ser menor que o bounding box completo do dado
    full_h, full_w = roi.full_crop.shape[:2]
    top_h, top_w = roi.top_face_crop.shape[:2]
    assert top_w < full_w
    assert top_h < full_h
    assert top_w == pytest.approx(full_w * 0.65, abs=2)


def test_ocr_engine_preprocessing_rotation_and_validation() -> None:
    """Valida CLAHE, geração de ângulos, rotação geométrica e filtro de valores entre 1 e 20."""
    config = DiceConfig(
        rotation_step_deg=30,
        ocr_target_size=(200, 200),
        min_ocr_confidence=0.3,
    )

    # Simula leituras em diferentes ângulos:
    # - ângulo 0°: valor inválido "0" (fora de 1..20) e "25" (fora de 1..20)
    # - ângulo 30°: leitura "18" com confiança 0.62
    # - ângulo 60°: leitura "20" com confiança 0.94 (deve vencer)
    fake_responses: list[list[tuple[Any, str, float]]] = [
        [([], "0", 0.99), ([], "25", 0.95)],
        [([], "18", 0.62)],
        [([], "20", 0.94)],
    ]
    fake_reader = FakeEasyOCRReader(fake_responses)
    engine = DiceOCREngine(config=config, reader_factory=lambda: fake_reader)

    angles = engine.get_rotation_angles()
    assert angles == [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

    # Validação unitária de números
    assert engine.parse_and_validate_number("1") == 1
    assert engine.parse_and_validate_number("20") == 20
    assert engine.parse_and_validate_number(" 14 ") == 14
    assert engine.parse_and_validate_number("0") is None
    assert engine.parse_and_validate_number("21") is None
    assert engine.parse_and_validate_number("abc") is None

    frame = _create_synthetic_dice_frame("20")
    preprocessed = engine.preprocess_roi(frame)
    assert preprocessed.shape == (200, 200)
    assert preprocessed.ndim == 2

    rotated = engine.rotate_image(preprocessed, 45.0)
    assert rotated.shape == preprocessed.shape

    result = engine.recognize_top_face(frame)
    assert result.is_valid
    assert result.value == 20
    assert result.confidence == pytest.approx(0.94)
    assert result.best_angle_deg == 60


def test_cli_argument_parser() -> None:
    """Verifica se main.py aceita os argumentos --camera, --image e --debug exigidos pela especificação."""
    parser = build_arg_parser()
    args = parser.parse_args(["--camera", "1", "--image", "teste.jpg", "--debug"])
    assert args.camera == 1
    assert args.image == "teste.jpg"
    assert args.debug is True


def test_segmenter_ignores_lighting_gradient_and_corner_artifacts() -> None:
    """Garante que gradientes de iluminação nos cantos não sequestrem a detecção do dado."""
    # Cria fundo 640x480 com gradiente forte (escuro em cima/esquerda, claro em baixo/direita)
    y_grad = np.linspace(30, 170, 480, dtype=np.float32)[:, None]
    x_grad = np.linspace(0, 50, 640, dtype=np.float32)[None, :]
    bg = np.clip(y_grad + x_grad, 0, 255).astype(np.uint8)
    empty_frame = cv2.merge([bg, bg, bg])

    segmenter = DiceSegmenter()
    # Em quadro vazio com gradiente, nenhum dado deve ser detectado no canto
    assert segmenter.detect_dice_roi(empty_frame) is None

    # Adiciona um dado colorido (vermelho) próximo ao centro com luminância parecida à do fundo
    frame_with_dice = empty_frame.copy()
    cv2.rectangle(frame_with_dice, (280, 200), (350, 270), (40, 40, 210), thickness=-1)
    roi = segmenter.detect_dice_roi(frame_with_dice)
    assert roi is not None
    x, y, w, h = roi.bbox
    cx, cy = x + w / 2.0, y + h / 2.0
    assert 260 <= cx <= 370
    assert 180 <= cy <= 290


def test_segmenter_detects_close_large_dice() -> None:
    """Garante que um dado grande (aproximado da câmera) seja detectado corretamente."""
    frame = np.full((480, 640, 3), 35, dtype=np.uint8)
    # Dado grande (240x240) centralizado
    cv2.rectangle(frame, (200, 120), (440, 360), (225, 225, 225), thickness=-1)
    cv2.putText(frame, "15", (270, 260), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (20, 20, 20), 5)

    segmenter = DiceSegmenter()
    roi = segmenter.detect_dice_roi(frame)
    assert roi is not None
    _, _, w, h = roi.bbox
    assert w >= 200 and h >= 200


def test_hardware_info_and_stats_overlay() -> None:
    """Verifica a descrição de hardware (CPU/GPU) e a renderização do overlay de estatísticas."""
    cpu_info = get_hardware_info(False)
    gpu_info = get_hardware_info(True)
    assert "CPU" in cpu_info
    assert "GPU" in gpu_info

    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered = draw_stats_overlay(
        canvas,
        [
            f"Dispositivo: {cpu_info}",
            "FPS: 30.0 | Frame: 640x480",
            "Movimento: 0.00% | ROI: 100x100",
        ],
    )
    assert rendered.shape == canvas.shape
    assert np.any(rendered > 0)


def test_dice_segmenter_detects_multiple_dice() -> None:
    """Garante que múltiplos dados simultâneos no mesmo frame sejam detectados separadamente."""
    frame = np.full((480, 640, 3), 30, dtype=np.uint8)
    # Dado 1 (branco, esquerda)
    cv2.rectangle(frame, (80, 180), (150, 250), (235, 235, 235), thickness=-1)
    # Dado 2 (azul, centro)
    cv2.rectangle(frame, (280, 190), (350, 260), (220, 90, 40), thickness=-1)
    # Dado 3 (verde, direita)
    cv2.rectangle(frame, (470, 200), (540, 270), (60, 210, 60), thickness=-1)

    segmenter = DiceSegmenter()
    rois = segmenter.detect_dice_rois(frame)
    assert len(rois) == 3

    centers_x = [roi.bbox[0] + roi.bbox[2] / 2.0 for roi in rois]
    assert centers_x[0] < centers_x[1] < centers_x[2]


def test_segmenter_filters_small_artifacts_and_soft_shadows() -> None:
    """Garante que sombras suaves e artefatos pequenos sejam ignorados pelo segmentador."""
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    # Artefato pequeno (25x25 = 625 px < min_contour_area=1500)
    cv2.rectangle(frame, (100, 100), (125, 125), (220, 220, 220), thickness=-1)
    # Sombra suave (contraste ~20 < min_foreground_diff=32)
    cv2.rectangle(frame, (250, 180), (380, 300), (100, 100, 100), thickness=-1)

    segmenter = DiceSegmenter()
    assert segmenter.detect_dice_rois(frame) == []


def test_segmenter_rejects_blurry_dark_shadow() -> None:
    """Garante que sombras escuras com bordas difusas sejam rejeitadas pelo filtro de nitidez."""
    frame = np.full((480, 640, 3), 160, dtype=np.uint8)
    shadow_layer = frame.copy()
    cv2.circle(shadow_layer, (320, 240), 55, (70, 70, 70), thickness=-1)
    # Borra fortemente a sombra para simular penumbra real
    frame = cv2.GaussianBlur(shadow_layer, (51, 51), 0)

    segmenter = DiceSegmenter()
    assert segmenter.detect_dice_rois(frame) == []


def test_dice_segmenter_splits_touching_dice() -> None:
    """Garante que dois dados encostados lado a lado sejam separados em dois contornos."""
    frame = np.full((480, 640, 3), 30, dtype=np.uint8)
    # Dois dados octogonais (raio 45) encostados lado a lado
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    for cx in (240, 326):
        pts = np.array(
            [[int(cx + 45 * np.cos(a)), int(220 + 45 * np.sin(a))] for a in angles],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [pts], (230, 230, 230))

    segmenter = DiceSegmenter()
    rois = segmenter.detect_dice_rois(frame)
    assert len(rois) == 2
    assert rois[0].bbox[0] < rois[1].bbox[0]


def test_play_area_parsing_and_roi_offset() -> None:
    """Verifica o parser de --play-area (retângulo/polígono), máscara e deslocamento de ROIs."""
    assert parse_play_area(None) is None
    rect_poly = parse_play_area("100, 50, 400, 300")
    assert rect_poly is not None and rect_poly.shape == (4, 1, 2)

    trap_poly = parse_play_area("100,50; 300,50; 340,250; 60,250")
    assert trap_poly is not None and trap_poly.shape == (4, 1, 2)

    with pytest.raises(ValueError):
        parse_play_area("100,50,0,300")

    frame = np.full((300, 400, 3), 30, dtype=np.uint8)
    # Desenha artefato fora do trapézio (canto superior esquerdo) e dado dentro do trapézio
    cv2.rectangle(frame, (5, 5), (50, 50), (240, 240, 240), thickness=-1)
    cv2.rectangle(frame, (165, 110), (235, 180), (230, 230, 230), thickness=-1)

    masked_crop, ax0, ay0, _ = mask_frame_to_polygon(frame, trap_poly)
    assert ax0 == 60 and ay0 == 50

    segmenter = DiceSegmenter()
    rois = segmenter.detect_dice_rois(masked_crop)
    assert len(rois) == 1

    shifted = offset_rois(rois, ax0, ay0)
    assert shifted[0].bbox[0] == rois[0].bbox[0] + ax0
    assert shifted[0].bbox[1] == rois[0].bbox[1] + ay0


def test_segmenter_ignores_large_tray_shadow_and_supports_bg_reference() -> None:
    """Garante que uma sombra grande na bandeja não seja fatiada em múltiplos dados falsos."""
    empty_tray = np.full((480, 640, 3), 150, dtype=np.uint8)
    shadow_layer = empty_tray.copy()
    # Sombra grande ocupando ~40% da bandeja no lado esquerdo
    cv2.rectangle(shadow_layer, (20, 40), (300, 440), (65, 65, 65), thickness=-1)
    empty_tray = cv2.GaussianBlur(shadow_layer, (45, 45), 0)

    segmenter = DiceSegmenter()
    # Sem referência de fundo, a sombra grande não deve ser fatiada em dados falsos
    assert segmenter.detect_dice_rois(empty_tray) == []

    # Com referência de fundo, um dado colocado dentro da própria sombra é detectado
    with_dice = empty_tray.copy()
    cv2.rectangle(with_dice, (130, 190), (200, 260), (235, 235, 235), thickness=-1)
    rois = segmenter.detect_dice_rois(with_dice, background=empty_tray)
    assert len(rois) == 1


def test_ocr_merges_adjacent_digits_and_clean_view_hides_contours() -> None:
    """Valida a junção de dígitos adjacentes ('1'+'2'->12) e o modo de visão limpa sem contornos."""
    # Simula EasyOCR retornando '1' e '2' lado a lado na mesma linha horizontal
    box_1 = [[40, 80], [70, 80], [70, 130], [40, 130]]
    box_2 = [[76, 82], [108, 82], [108, 132], [76, 132]]
    fake_reader = FakeEasyOCRReader([[(box_1, "1", 0.91), (box_2, "2", 0.89)]])
    engine = DiceOCREngine(reader_factory=lambda: fake_reader)
    frame = _create_synthetic_dice_frame("12")
    result = engine.recognize_top_face(frame)
    assert result.is_valid
    assert result.value == 12

    # No modo limpo (clean_view=True), nenhum contorno verde do polígono da bandeja é desenhado
    blank = np.zeros((300, 400, 3), dtype=np.uint8)
    poly = parse_play_area("20,20;380,20;380,280;20,280")
    hud_clean = draw_hud(
        blank,
        "d12/d20: 12",
        (40, 220, 80),
        rois=[],
        ocr_results=[OCRResult(12, 0.9, 0, np.zeros((10, 10), dtype=np.uint8), [])],
        play_polygon=poly,
        clean_view=True,
    )
    # A borda inferior fora da faixa superior (y=280) deve permanecer preta (sem polígono verde)
    assert np.all(hud_clean[280, 100] == 0)
