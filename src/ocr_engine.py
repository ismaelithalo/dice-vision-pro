"""Motor de pré-processamento de imagem, varredura angular e leitura via EasyOCR."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import cv2
import numpy as np

from src.config import DEFAULT_CONFIG, DiceConfig


@dataclass
class OCRCandidate:
    """Representa uma leitura candidata obtida em um determinado ângulo de rotação."""

    value: int
    raw_text: str
    confidence: float
    angle_deg: int


@dataclass
class OCRResult:
    """Resultado consolidado da varredura angular de OCR."""

    value: Optional[int]
    confidence: float
    best_angle_deg: int
    preprocessed_image: np.ndarray
    candidates: list[OCRCandidate]

    @property
    def is_valid(self) -> bool:
        """Indica se um número válido de dado (1..20) foi reconhecido."""
        return self.value is not None


class DiceOCREngine:
    """Realiza pré-processamento CLAHE, rotações de 0° a 360° e leitura do valor do dado via EasyOCR."""

    def __init__(
        self,
        config: DiceConfig = DEFAULT_CONFIG,
        reader_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.config = config
        self._reader_factory = reader_factory
        self._reader: Optional[Any] = None

    def _get_reader(self) -> Any:
        """Inicializa o EasyOCR Reader sob demanda (lazy loading) para acelerar imports e testes."""
        if self._reader is None:
            if self._reader_factory is not None:
                self._reader = self._reader_factory()
            else:
                import easyocr  # type: ignore[import-untyped]

                self._reader = easyocr.Reader(
                    ["en"],
                    gpu=bool(self.config.use_gpu),
                    verbose=False,
                )
        return self._reader

    def preprocess_roi(self, roi_image: np.ndarray) -> np.ndarray:
        """Aplica conversão para escala de cinza, redimensionamento, CLAHE e leve nitidez."""
        if roi_image.size == 0:
            raise ValueError("A imagem de ROI fornecida para o pré-processamento está vazia.")

        if roi_image.ndim == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image.copy()

        resized = cv2.resize(
            gray,
            self.config.ocr_target_size,
            interpolation=cv2.INTER_CUBIC,
        )

        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size,
        )
        enhanced = clahe.apply(resized)
        denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
        return denoised

    @staticmethod
    def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
        """Rotaciona a imagem em torno do seu centro preservando as dimensões e preenchendo a borda com a mediana."""
        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        border_val = int(np.median(image))
        rotated = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_val,
        )
        return rotated

    def get_rotation_angles(self) -> list[int]:
        """Retorna a lista de ângulos da varredura de 0° a 360° (excluindo 360° duplicado)."""
        step = max(1, int(self.config.rotation_step_deg))
        return list(range(0, 360, step))

    def parse_and_validate_number(self, raw_text: str) -> Optional[int]:
        """Extrai dígitos da string lida e valida se o número pertence ao intervalo [1, 20]."""
        cleaned = re.sub(r"[^0-9]", "", raw_text.strip())
        if not cleaned:
            return None

        try:
            value = int(cleaned)
        except ValueError:
            return None

        if self.config.min_dice_value <= value <= self.config.max_dice_value:
            return value
        return None

    def _collect_candidates_from_detections(
        self,
        detections: Sequence[Any],
        angle_deg: int,
    ) -> list[OCRCandidate]:
        """Converte a saída do EasyOCR em candidatos validados no intervalo permitido."""
        candidates: list[OCRCandidate] = []
        for item in detections:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            _, raw_text, confidence = item[0], str(item[1]), float(item[2])
            if confidence < self.config.min_ocr_confidence:
                continue

            value = self.parse_and_validate_number(raw_text)
            if value is not None:
                candidates.append(
                    OCRCandidate(
                        value=value,
                        raw_text=raw_text,
                        confidence=confidence,
                        angle_deg=angle_deg,
                    )
                )
        return candidates

    def recognize_top_face(self, roi_image: np.ndarray) -> OCRResult:
        """Executa o pré-processamento e varre múltiplos ângulos para encontrar a leitura de maior confiança."""
        preprocessed = self.preprocess_roi(roi_image)
        reader = self._get_reader()
        angles = self.get_rotation_angles()

        all_candidates: list[OCRCandidate] = []

        # Testamos a polaridade original do CLAHE e sua versão invertida caso nenhuma leitura de alta confiança surja
        variants = [preprocessed, cv2.bitwise_not(preprocessed)]

        for variant_idx, base_img in enumerate(variants):
            for angle in angles:
                rotated = self.rotate_image(base_img, angle) if angle != 0 else base_img
                detections = reader.readtext(
                    rotated,
                    allowlist=self.config.ocr_allowlist,
                    detail=1,
                    paragraph=False,
                )
                all_candidates.extend(
                    self._collect_candidates_from_detections(detections, angle)
                )

            # Se a polaridade padrão já encontrou uma leitura sólida, evitamos dobrar o custo na imagem invertida
            if variant_idx == 0 and any(c.confidence >= 0.75 for c in all_candidates):
                break

        if not all_candidates:
            return OCRResult(
                value=None,
                confidence=0.0,
                best_angle_deg=0,
                preprocessed_image=preprocessed,
                candidates=[],
            )

        best_candidate = max(all_candidates, key=lambda c: c.confidence)
        return OCRResult(
            value=best_candidate.value,
            confidence=best_candidate.confidence,
            best_angle_deg=best_candidate.angle_deg,
            preprocessed_image=preprocessed,
            candidates=all_candidates,
        )

