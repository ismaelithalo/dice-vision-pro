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
        """Gera máscara binária do dado sobre o fundo usando contraste local e distância de cor (Lab).

        A estimativa de fundo suavizada (em baixa resolução) remove gradientes de iluminação e
        vinhetas nos cantos da câmera, enquanto os canais de crominância (a, b) permitem detectar
        dados coloridos mesmo quando possuem luminância parecida com a da mesa.
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
            h, w = blurred.shape[:2]
            # Estima o fundo localmente em escala reduzida para ser imune a sombras/vinhetas nos cantos
            bg_w, bg_h = max(16, w // 8), max(16, h // 8)
            small_gray = cv2.resize(blurred, (bg_w, bg_h), interpolation=cv2.INTER_AREA)
            small_bg = cv2.GaussianBlur(small_gray, (21, 21), 0)
            local_bg = cv2.resize(small_bg, (w, h), interpolation=cv2.INTER_LINEAR)
            fg_score = cv2.absdiff(blurred, local_bg)

            if frame.ndim == 3 and frame.shape[2] == 3:
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
                small_lab = cv2.resize(lab, (bg_w, bg_h), interpolation=cv2.INTER_AREA)
                med_a = int(np.median(small_lab[:, :, 1]))
                med_b = int(np.median(small_lab[:, :, 2]))
                diff_a = cv2.absdiff(lab[:, :, 1], np.full((h, w), med_a, dtype=np.uint8))
                diff_b = cv2.absdiff(lab[:, :, 2], np.full((h, w), med_b, dtype=np.uint8))
                chroma_diff = cv2.add(diff_a, diff_b)
                fg_score = cv2.max(fg_score, chroma_diff)

            fg_score = cv2.GaussianBlur(fg_score, self.config.blur_kernel_size, 0)
            min_diff = max(1, int(self.config.min_foreground_diff))

            if int(np.max(fg_score)) < min_diff:
                return np.zeros((h, w), dtype=np.uint8)

            otsu_thresh, mask = cv2.threshold(
                fg_score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            if otsu_thresh < min_diff:
                _, mask = cv2.threshold(fg_score, min_diff, 255, cv2.THRESH_BINARY)

        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
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
        effective_max_area = min(
            float(self.config.max_contour_area),
            float(frame_w * frame_h) * 0.55,
        )
        border_margin = 2

        best_contour: Optional[np.ndarray] = None
        best_area: float = 0.0
        best_score: float = -1.0
        best_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.config.min_contour_area or area > effective_max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 0 or h <= 0:
                continue

            # Descarta artefatos ancorados nos cantos/bordas da imagem (sombras, vinhetas, borda da mesa)
            edges_touched = (
                int(x <= border_margin)
                + int(y <= border_margin)
                + int((x + w) >= (frame_w - border_margin))
                + int((y + h) >= (frame_h - border_margin))
            )
            if edges_touched >= 2:
                continue

            aspect_ratio = w / float(h)
            if aspect_ratio < 0.40 or aspect_ratio > 2.50:
                continue

            extent = area / float(w * h)
            if extent < 0.35:
                continue

            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < 0.60:
                continue

            # Pontuação equilibrada: favorece objetos convexos, centralizados e sem dominar por área excessiva
            cx = x + w / 2.0
            cy = y + h / 2.0
            dist = float(np.linalg.norm(np.array([cx, cy]) - frame_center))
            max_dist = max(1.0, float(np.hypot(frame_w, frame_h)))
            centrality = 1.0 - (dist / max_dist)
            score = (area ** 0.5) * (0.35 + 0.65 * centrality) * solidity
            if edges_touched == 1:
                score *= 0.25

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

