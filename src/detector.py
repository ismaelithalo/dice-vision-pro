"""Módulo de detecção de estabilidade (anti-blur) e segmentação de dados poliédricos via OpenCV."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.config import DEFAULT_CONFIG, DiceConfig


@dataclass
class StabilityStatus:
    """Estado de movimento e estabilidade do quadro atual."""

    is_moving: bool
    is_stable: bool
    stable_elapsed_sec: float
    motion_ratio: float
    diff_mask: np.ndarray


@dataclass
class DiceROI:
    """Resultado da segmentação e recorte do dado detectado."""

    bbox: tuple[int, int, int, int]  # (x, y, w, h) no frame original
    contour: np.ndarray
    contour_area: float
    full_crop: np.ndarray
    top_face_crop: np.ndarray
    threshold_mask: np.ndarray


class MotionStabilityDetector:
    """Analisa frames consecutivos via absdiff para disparar o OCR apenas após o dado estabilizar."""

    def __init__(self, config: DiceConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._prev_gray: Optional[np.ndarray] = None
        self._stable_since: Optional[float] = None

    def reset(self) -> None:
        """Reinicia o histórico de frames e o cronômetro de estabilidade."""
        self._prev_gray = None
        self._stable_since = None

    def update(self, frame: np.ndarray, timestamp: Optional[float] = None) -> StabilityStatus:
        """Compara o frame atual com o anterior e atualiza o temporizador de estabilidade.

        Args:
            frame: Imagem BGR ou em escala de cinza capturada pela câmera.
            timestamp: Instante atual em segundos (usa time.monotonic() por padrão).

        Returns:
            Objeto StabilityStatus indicando se há movimento e se o tempo de estabilidade foi atingido.
        """
        now = time.monotonic() if timestamp is None else timestamp
        gray = self._to_blurred_gray(frame)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            self._stable_since = now
            empty_mask = np.zeros_like(gray)
            return StabilityStatus(
                is_moving=False,
                is_stable=False,
                stable_elapsed_sec=0.0,
                motion_ratio=0.0,
                diff_mask=empty_mask,
            )

        diff = cv2.absdiff(self._prev_gray, gray)
        _, thresh = cv2.threshold(
            diff,
            self.config.motion_pixel_diff_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        changed_pixels = int(np.count_nonzero(thresh))
        total_pixels = int(thresh.size) if thresh.size > 0 else 1
        motion_ratio = changed_pixels / float(total_pixels)

        self._prev_gray = gray
        is_moving = motion_ratio > self.config.motion_changed_ratio_threshold

        if is_moving:
            self._stable_since = None
            elapsed = 0.0
            is_stable = False
        else:
            if self._stable_since is None:
                self._stable_since = now
            elapsed = max(0.0, now - self._stable_since)
            is_stable = elapsed >= self.config.stability_duration_sec

        return StabilityStatus(
            is_moving=is_moving,
            is_stable=is_stable,
            stable_elapsed_sec=elapsed,
            motion_ratio=motion_ratio,
            diff_mask=thresh,
        )

    def _to_blurred_gray(self, frame: np.ndarray) -> np.ndarray:
        """Converte frame para escala de cinza e aplica suavização Gaussiana."""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        return cv2.GaussianBlur(gray, self.config.blur_kernel_size, 0)


class DiceSegmenter:
    """Segmenta o dado sobre fundo uniforme e extrai a região central da face superior."""

    def __init__(self, config: DiceConfig = DEFAULT_CONFIG) -> None:
        self.config = config

    def compute_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """Gera máscara binária do dado em relação ao fundo uniforme via Otsu ou Adaptive Threshold.

        Seleciona automaticamente a polaridade (dado claro em fundo escuro ou dado escuro em fundo claro)
        observando a intensidade predominante das bordas do frame.
        """
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        blurred = cv2.GaussianBlur(gray, self.config.blur_kernel_size, 0)

        if self.config.use_adaptive_threshold:
            block_size = max(3, self.config.adaptive_block_size | 1)
            mask = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                self.config.adaptive_c,
            )
        else:
            _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Se as bordas da imagem estiverem majoritariamente brancas (255), invertemos para que o fundo seja 0
            border_pixels = np.concatenate(
                [mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]]
            )
            if np.mean(border_pixels) > 127.0:
                mask = cv2.bitwise_not(mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def extract_top_face_crop(self, full_crop: np.ndarray) -> np.ndarray:
        """Aplica recorte centralizado e máscara circular opcional para isolar a face superior do dado."""
        h, w = full_crop.shape[:2]
        if h == 0 or w == 0:
            return full_crop.copy()

        ratio = float(np.clip(self.config.center_crop_ratio, 0.25, 1.0))
        crop_w = max(8, int(round(w * ratio)))
        crop_h = max(8, int(round(h * ratio)))

        x0 = max(0, (w - crop_w) // 2)
        y0 = max(0, (h - crop_h) // 2)
        center_crop = full_crop[y0 : y0 + crop_h, x0 : x0 + crop_w].copy()

        if not self.config.apply_circular_mask:
            return center_crop

        ch, cw = center_crop.shape[:2]
        circle_mask = np.zeros((ch, cw), dtype=np.uint8)
        center = (cw // 2, ch // 2)
        radius = max(4, int(min(cw, ch) * 0.48))
        cv2.circle(circle_mask, center, radius, 255, thickness=-1)

        # Preenche o exterior da máscara circular com a cor mediana da borda interna para evitar bordas falsas no OCR
        bg_value = int(np.median(center_crop))
        masked = np.full_like(center_crop, bg_value)
        masked[circle_mask == 255] = center_crop[circle_mask == 255]
        return masked

    def detect_dice_roi(self, frame: np.ndarray) -> Optional[DiceROI]:
        """Localiza o contorno principal do dado e retorna seu bounding box e o recorte da face superior."""
        mask = self.compute_foreground_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_h, frame_w = frame.shape[:2]
        frame_center = np.array([frame_w / 2.0, frame_h / 2.0])

        best_contour: Optional[np.ndarray] = None
        best_area: float = 0.0
        best_score: float = -1.0
        best_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.config.min_contour_area or area > self.config.max_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 0.0
            if aspect_ratio < 0.35 or aspect_ratio > 2.85:
                continue

            # Prioriza contornos grandes e próximos ao centro da imagem
            cx = x + w / 2.0
            cy = y + h / 2.0
            dist = float(np.linalg.norm(np.array([cx, cy]) - frame_center))
            max_dist = max(1.0, float(np.hypot(frame_w, frame_h)))
            centrality = 1.0 - (dist / max_dist)
            score = area * (0.5 + 0.5 * centrality)

            if score > best_score:
                best_score = score
                best_area = area
                best_contour = contour
                best_bbox = (x, y, w, h)

        if best_contour is None:
            return None

        x, y, w, h = best_bbox
        pad = self.config.roi_padding_px
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(frame_w, x + w + pad)
        y1 = min(frame_h, y + h + pad)

        full_crop = frame[y0:y1, x0:x1].copy()
        top_face_crop = self.extract_top_face_crop(full_crop)

        return DiceROI(
            bbox=(x0, y0, x1 - x0, y1 - y0),
            contour=best_contour,
            contour_area=best_area,
            full_crop=full_crop,
            top_face_crop=top_face_crop,
            threshold_mask=mask,
        )

