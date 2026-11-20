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
                    recog_network="english_g2",
                    quantize=False,
                    verbose=False,
                )
        return self._reader

    def preprocess_roi(self, roi_image: np.ndarray) -> np.ndarray:
        """Aplica escala de cinza, suavização preservando bordas, normalização de contraste e margem silenciosa."""
        if roi_image.size == 0:
            raise ValueError("A imagem de ROI fornecida para o pré-processamento está vazia.")

        if roi_image.ndim == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image.copy()

        target_w, target_h = self.config.ocr_target_size
        inner_w = max(32, int(target_w * 0.72))
        inner_h = max(32, int(target_h * 0.72))

        resized = cv2.resize(gray, (inner_w, inner_h), interpolation=cv2.INTER_CUBIC)
        denoised = cv2.bilateralFilter(resized, d=5, sigmaColor=40, sigmaSpace=40)

        # Normalização linear preserva traços dos dígitos sem amplificar granulação do dado como CLAHE forte
        normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)

        border_val = int(np.median(normalized))
        canvas = np.full((target_h, target_w), border_val, dtype=np.uint8)
        x0 = (target_w - inner_w) // 2
        y0 = (target_h - inner_h) // 2
        canvas[y0 : y0 + inner_h, x0 : x0 + inner_w] = normalized
        return canvas

    @staticmethod
    def _build_adaptive_binary_variant(preprocessed: np.ndarray) -> np.ndarray:
        """Gera versão binarizada limpa (texto escuro em fundo branco) para iluminação difícil."""
        binary = cv2.adaptiveThreshold(
            preprocessed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            8,
        )
        # Se a maior parte da imagem ficou preta, inverte para manter texto escuro sobre fundo branco
        if np.mean(binary) < 127.0:
            binary = cv2.bitwise_not(binary)
        return binary

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
        """Converte a saída do EasyOCR em candidatos validados, unindo dígitos adjacentes (ex: '1' + '2' -> 12)."""
        candidates: list[OCRCandidate] = []
        raw_boxes: list[tuple[float, float, float, float, str, float]] = []

        for item in detections:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            box, raw_text, confidence = item[0], str(item[1]), float(item[2])
            if confidence < self.config.min_ocr_confidence:
                continue

            cleaned = re.sub(r"[^0-9]", "", raw_text.strip())
            if not cleaned:
                continue

            if isinstance(box, (list, tuple)) and len(box) >= 4:
                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]
                cx = sum(xs) / len(xs)
                cy = sum(ys) / len(ys)
                bw = max(1.0, max(xs) - min(xs))
                bh = max(1.0, max(ys) - min(ys))
                raw_boxes.append((cx, cy, bw, bh, cleaned, confidence))
            else:
                value = self.parse_and_validate_number(cleaned)
                if value is not None:
                    candidates.append(
                        OCRCandidate(
                            value=value,
                            raw_text=raw_text,
                            confidence=confidence,
                            angle_deg=angle_deg,
                        )
                    )

        # Une dígitos alinhados horizontalmente (ex: '1' e '2' formando '12') antes de emitir fragmentos isolados
        used_indices: set[int] = set()
        if len(raw_boxes) >= 2:
            raw_boxes.sort(key=lambda b: b[0])
            for i in range(len(raw_boxes) - 1):
                if i in used_indices:
                    continue
                cx1, cy1, bw1, bh1, t1, c1 = raw_boxes[i]
                cx2, cy2, bw2, bh2, t2, c2 = raw_boxes[i + 1]
                avg_h = (bh1 + bh2) * 0.5
                if abs(cy1 - cy2) <= avg_h * 0.45 and (cx2 - cx1) <= (bw1 + bw2) * 1.1:
                    merged_val = self.parse_and_validate_number(t1 + t2)
                    if merged_val is not None:
                        candidates.append(
                            OCRCandidate(
                                value=merged_val,
                                raw_text=t1 + t2,
                                confidence=(c1 + c2) * 0.5,
                                angle_deg=angle_deg,
                            )
                        )
                        used_indices.add(i)
                        used_indices.add(i + 1)

        for idx, (_, _, _, _, cleaned, confidence) in enumerate(raw_boxes):
            if idx in used_indices:
                continue
            value = self.parse_and_validate_number(cleaned)
            if value is not None:
                candidates.append(
                    OCRCandidate(
                        value=value,
                        raw_text=cleaned,
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

        # Testa polaridade original, invertida e binarização adaptativa se ainda não houver leitura sólida
        variants = [
            preprocessed,
            255 - preprocessed,
            self._build_adaptive_binary_variant(preprocessed),
        ]

        for variant_idx, base_img in enumerate(variants):
            for angle in angles:
                rotated = self.rotate_image(base_img, angle) if angle != 0 else base_img
                try:
                    detections = reader.readtext(
                        rotated,
                        decoder="beamsearch",
                        beamWidth=5,
                        allowlist=self.config.ocr_allowlist,
                        detail=1,
                        paragraph=False,
                        min_size=10,
                        text_threshold=0.65,
                        low_text=0.35,
                        mag_ratio=1.5,
                    )
                except TypeError:
                    detections = reader.readtext(
                        rotated,
                        allowlist=self.config.ocr_allowlist,
                        detail=1,
                        paragraph=False,
                    )

                all_candidates.extend(
                    self._collect_candidates_from_detections(detections, angle)
                )

            # Se já encontrou uma leitura sólida, evita rodar passagens extras
            if any(c.confidence >= 0.75 for c in all_candidates):
                break

        if not all_candidates:
            return OCRResult(
                value=None,
                confidence=0.0,
                best_angle_deg=0,
                preprocessed_image=preprocessed,
                candidates=[],
            )

        # Consenso multi-ângulo: favorece valores consistentes em múltiplos ângulos em vez de ruídos isolados
        by_value: dict[int, list[OCRCandidate]] = {}
        for cand in all_candidates:
            by_value.setdefault(cand.value, []).append(cand)

        def _value_score(cands: list[OCRCandidate]) -> float:
            best_conf = max(c.confidence for c in cands)
            return best_conf + 0.12 * min(3, len(cands) - 1)

        winning_cands = max(by_value.values(), key=_value_score)
        best_candidate = max(winning_cands, key=lambda c: c.confidence)
        return OCRResult(
            value=best_candidate.value,
            confidence=best_candidate.confidence,
            best_angle_deg=best_candidate.angle_deg,
            preprocessed_image=preprocessed,
            candidates=all_candidates,
        )

