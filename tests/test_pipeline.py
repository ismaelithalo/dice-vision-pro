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

from src.config import DEFAULT_CONFIG, DiceConfig, detect_gpu_available
from src.detector import DiceSegmenter, MotionStabilityDetector
from src.main import build_arg_parser
from src.ocr_engine import DiceOCREngine


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
