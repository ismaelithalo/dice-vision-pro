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
    approx_polygon: Optional[np.ndarray] = None
    shape_label: str = ""
    face_center: Optional[tuple[int, int]] = None
    face_radius: int = 0


class MotionStabilityDetector:
    """Analisa frames consecutivos via absdiff para disparar o OCR apenas após o dado estabilizar."""

    def __init__(self, config: DiceConfig = DEFAULT_CONFIG) -> None:
        self.config = config
        self._prev_gray: Optional[np.ndarray] = None
        self._stable_since: Optional[float] = None
        self._motion_accumulator: Optional[np.ndarray] = None
        self.has_recorded_motion: bool = False

    def reset(self) -> None:
        """Reinicia o histórico de frames e o cronômetro de estabilidade."""
        self._prev_gray = None
        self._stable_since = None
        self._motion_accumulator = None
        self.has_recorded_motion = False

    def clear_motion_history(self) -> None:
        """Limpa a máscara acumulada de movimento após concluir uma leitura."""
        if self._motion_accumulator is not None:
            self._motion_accumulator.fill(0)
        self.has_recorded_motion = False

    def has_recent_motion_at(self, bbox: tuple[int, int, int, int]) -> bool:
        """Verifica se a região do objeto teve movimento recente antes de parar."""
        if not self.has_recorded_motion or self._motion_accumulator is None:
            return True
        x, y, w, h = bbox
        ah, aw = self._motion_accumulator.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(aw, x + w), min(ah, y + h)
        if x1 <= x0 or y1 <= y0:
            return False
        region = self._motion_accumulator[y0:y1, x0:x1]
        active_ratio = float(np.count_nonzero(region)) / float(region.size)
        return active_ratio >= 0.08

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

        if self._motion_accumulator is None or self._motion_accumulator.shape != thresh.shape:
            self._motion_accumulator = np.zeros_like(thresh)

        if is_moving:
            self._stable_since = None
            elapsed = 0.0
            is_stable = False
            dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            dilated_motion = cv2.dilate(thresh, dilate_k, iterations=1)
            self._motion_accumulator = cv2.bitwise_or(self._motion_accumulator, dilated_motion)
            self.has_recorded_motion = True
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

    def compute_foreground_mask(
        self, frame: np.ndarray, background: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Gera máscara binária sólida usando fundo de referência (se disponível) ou fundo mediano local + Lab.

        Mantém a silhueta preenchida mesmo quando uma das arestas do dado tem baixo contraste,
        permitindo que a validação geométrica por polígonos filtre sombras e artefatos depois.
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
        elif background is not None and background.shape == frame.shape:
            bg_gray = (
                cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
                if background.ndim == 3
                else background.copy()
            )
            bg_blurred = cv2.GaussianBlur(bg_gray, self.config.blur_kernel_size, 0)
            fg_score = cv2.absdiff(blurred, bg_blurred)
            if frame.ndim == 3 and frame.shape[2] == 3:
                lab_cur = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
                lab_bg = cv2.cvtColor(background, cv2.COLOR_BGR2Lab)
                diff_a = cv2.absdiff(lab_cur[:, :, 1], lab_bg[:, :, 1])
                diff_b = cv2.absdiff(lab_cur[:, :, 2], lab_bg[:, :, 2])
                fg_score = cv2.max(fg_score, cv2.add(diff_a, diff_b))

            fg_score = cv2.GaussianBlur(fg_score, self.config.blur_kernel_size, 0)
            min_diff = max(1, int(self.config.min_foreground_diff))
            if int(np.max(fg_score)) < min_diff:
                return np.zeros(blurred.shape[:2], dtype=np.uint8)
            otsu_thresh, mask = cv2.threshold(
                fg_score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            if otsu_thresh < min_diff:
                _, mask = cv2.threshold(fg_score, min_diff, 255, cv2.THRESH_BINARY)
        else:
            h, w = blurred.shape[:2]
            bg_w, bg_h = 64, 48
            small_gray = cv2.resize(blurred, (bg_w, bg_h), interpolation=cv2.INTER_AREA)
            small_bg = cv2.medianBlur(small_gray, 39)
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

        # Abertura primeiro para remover poeira/ruído antes do fechamento
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
        return mask

    def _split_touching_contours(self, mask: np.ndarray) -> list[np.ndarray]:
        """Separa dados que caíram encostados usando transformada de distância e picos locais."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_contours: list[np.ndarray] = []

        mask_area = float(mask.shape[0] * mask.shape[1])
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.config.min_contour_area:
                continue

            # Evita fatiar manchas gigantes (ex: sombra grande da bandeja) em vários polígonos falsos
            if area > mask_area * 0.35:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h > 0 else 1.0
            # Só tenta dividir se o contorno for grande ou alongado o suficiente para conter 2+ dados
            if area < self.config.min_contour_area * 1.8 and 0.72 <= aspect_ratio <= 1.38:
                result_contours.append(contour)
                continue

            roi_mask = np.zeros((h, w), dtype=np.uint8)
            shifted = contour - np.array([[[x, y]]])
            cv2.drawContours(roi_mask, [shifted], -1, 255, thickness=-1)

            dist = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
            max_dist = float(np.max(dist))
            if max_dist < 10.0:
                result_contours.append(contour)
                continue

            peak_kernel_size = max(15, int(max_dist * 1.1) | 1)
            peak_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (peak_kernel_size, peak_kernel_size)
            )
            dilated = cv2.dilate(dist, peak_kernel)
            peaks_mask = (dist >= max_dist * 0.45) & (dist == dilated)
            num_labels, _, _, centroids = cv2.connectedComponentsWithStats(
                peaks_mask.astype(np.uint8)
            )

            # Filtra picos muito próximos entre si
            peaks: list[tuple[float, float]] = []
            min_peak_sep = max_dist * 0.9
            for idx in range(1, num_labels):
                px, py = float(centroids[idx][0]), float(centroids[idx][1])
                if all(np.hypot(px - qx, py - qy) >= min_peak_sep for qx, qy in peaks):
                    peaks.append((px, py))

            if len(peaks) < 2 or len(peaks) > 4:
                result_contours.append(contour)
                continue

            # Atribui cada pixel ativo do contorno ao pico mais próximo (partição Voronoi)
            ys, xs = np.nonzero(roi_mask)
            dists_to_peaks = np.stack(
                [(xs - px) ** 2 + (ys - py) ** 2 for px, py in peaks],
                axis=0,
            )
            nearest = np.argmin(dists_to_peaks, axis=0)

            for peak_idx in range(len(peaks)):
                part_mask = np.zeros((h, w), dtype=np.uint8)
                sel = nearest == peak_idx
                part_mask[ys[sel], xs[sel]] = 255
                part_contours, _ = cv2.findContours(
                    part_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                for pc in part_contours:
                    if cv2.contourArea(pc) >= self.config.min_contour_area:
                        result_contours.append(pc + np.array([[[x, y]]]))

        return result_contours

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

        # Preenche o exterior da máscara circular com a mediana por canal para não criar anel artificial
        if center_crop.ndim == 3:
            bg_color = np.median(center_crop.reshape(-1, center_crop.shape[2]), axis=0).astype(
                center_crop.dtype
            )
            masked = np.empty_like(center_crop)
            masked[:, :] = bg_color
        else:
            bg_value = int(np.median(center_crop))
            masked = np.full_like(center_crop, bg_value)

        masked[circle_mask == 255] = center_crop[circle_mask == 255]
        return masked

    def extract_polygon_center_crop(
        self,
        frame: np.ndarray,
        approx_polygon: np.ndarray,
        full_crop_shape: Optional[tuple[int, int]] = None,
    ) -> tuple[np.ndarray, tuple[int, int], int]:
        """Extrai o recorte centralizado no centroide do polígono e limitado ao seu raio inscrito."""
        if approx_polygon is None or len(approx_polygon) < 3:
            fallback = self.extract_top_face_crop(frame)
            fh, fw = fallback.shape[:2]
            return fallback, (fw // 2, fh // 2), max(4, min(fw, fh) // 2)

        moments = cv2.moments(approx_polygon)
        if moments["m00"] > 0:
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
        else:
            pts = approx_polygon.reshape(-1, 2)
            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))

        r_in = float(cv2.pointPolygonTest(approx_polygon, (cx, cy), True))
        if r_in <= 4.0:
            x, y, w, h = cv2.boundingRect(approx_polygon)
            fallback = self.extract_top_face_crop(frame[y : y + h, x : x + w])
            return fallback, (x + w // 2, y + h // 2), max(4, min(w, h) // 4)

        ratio = float(np.clip(self.config.center_crop_ratio, 0.25, 1.0))
        # Em câmeras anguladas, a face superior fica deslocada ligeiramente para cima do centroide 2D
        face_radius = max(6, int(round(r_in * max(ratio, 0.72))))
        upward_shift = int(round(r_in * 0.10))

        if full_crop_shape is not None:
            ref_h, ref_w = full_crop_shape
            crop_w = max(8, int(round(ref_w * ratio)))
            crop_h = max(8, int(round(ref_h * ratio)))
        else:
            crop_w = max(8, face_radius * 2)
            crop_h = max(8, face_radius * 2)

        frame_h, frame_w = frame.shape[:2]
        icx = int(round(cx))
        icy = max(0, int(round(cy)) - upward_shift)
        x0 = max(0, icx - crop_w // 2)
        y0 = max(0, icy - crop_h // 2)
        x1 = min(frame_w, x0 + crop_w)
        y1 = min(frame_h, y0 + crop_h)

        if x1 <= x0 or y1 <= y0:
            fallback = self.extract_top_face_crop(frame)
            return fallback, (icx, icy), face_radius

        center_crop = frame[y0:y1, x0:x1].copy()
        if not self.config.apply_circular_mask:
            return center_crop, (icx, icy), face_radius

        ch, cw = center_crop.shape[:2]
        circle_mask = np.zeros((ch, cw), dtype=np.uint8)
        mask_radius = max(4, min(int(min(cw, ch) * 0.48), face_radius))
        cv2.circle(circle_mask, (cw // 2, ch // 2), mask_radius, 255, thickness=-1)

        if center_crop.ndim == 3:
            bg_color = np.median(center_crop.reshape(-1, center_crop.shape[2]), axis=0).astype(
                center_crop.dtype
            )
            masked = np.empty_like(center_crop)
            masked[:, :] = bg_color
        else:
            bg_value = int(np.median(center_crop))
            masked = np.full_like(center_crop, bg_value)

        masked[circle_mask == 255] = center_crop[circle_mask == 255]
        return masked, (icx, icy), mask_radius

    def _score_and_collect_rois(
        self, frame: np.ndarray, background: Optional[np.ndarray] = None
    ) -> list[tuple[float, DiceROI]]:
        """Segmenta e pontua todos os contornos candidatos a dados no frame."""
        mask = self.compute_foreground_mask(frame, background=background)
        contours = self._split_touching_contours(mask)

        # Mapa de nitidez de borda (Lab/cinza) para rejeitar sombras suaves com contorno difuso
        grad_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        if frame.ndim == 3 and frame.shape[2] == 3:
            lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
            grad_channels = cv2.morphologyEx(lab_frame, cv2.MORPH_GRADIENT, grad_kernel)
            edge_map = np.max(grad_channels, axis=2)
        else:
            edge_map = cv2.morphologyEx(frame, cv2.MORPH_GRADIENT, grad_kernel)

        frame_h, frame_w = frame.shape[:2]
        frame_center = np.array([frame_w / 2.0, frame_h / 2.0])
        effective_max_area = min(
            float(self.config.max_contour_area),
            float(frame_w * frame_h) * 0.55,
        )
        border_margin = 2
        pad = self.config.roi_padding_px

        scored_rois: list[tuple[float, DiceROI]] = []

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
            if aspect_ratio < 0.50 or aspect_ratio > 2.00:
                continue

            extent = area / float(w * h)
            if extent < 0.45:
                continue

            hull = cv2.convexHull(contour)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < 0.78:
                continue

            # Validação geométrica: polígonos convexos fechados (4 a 10 vértices) e boa circularidade
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0:
                continue
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            if circularity < 0.52:
                continue

            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            num_vertices = len(approx)
            if num_vertices < 4 or num_vertices > 10:
                continue

            (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
            if rw <= 0 or rh <= 0:
                continue
            rect_aspect = max(rw, rh) / min(rw, rh)
            fill_ratio = area / (rw * rh)
            if rect_aspect > 2.0 or fill_ratio < 0.50:
                continue

            if num_vertices == 4:
                shape_label = "4v (d6)"
            elif num_vertices <= 6:
                shape_label = f"{num_vertices}v (d8/d10)"
            else:
                shape_label = f"{num_vertices}v (d12/d20)"

            # Sombras difusas têm gradiente baixo ao longo do próprio contorno; dados têm bordas nítidas
            perim_mask = np.zeros((h, w), dtype=np.uint8)
            shifted_cnt = contour - np.array([[[x, y]]])
            cv2.drawContours(perim_mask, [shifted_cnt], -1, 255, thickness=7)
            perim_edges = edge_map[y : y + h, x : x + w][perim_mask == 255]
            if perim_edges.size > 0 and float(np.percentile(perim_edges, 90)) < 14.0:
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

            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(frame_w, x + w + pad)
            y1 = min(frame_h, y + h + pad)

            full_crop = frame[y0:y1, x0:x1].copy()
            top_face_crop, face_center, face_radius = self.extract_polygon_center_crop(
                frame, approx, full_crop_shape=(y1 - y0, x1 - x0)
            )

            roi = DiceROI(
                bbox=(x0, y0, x1 - x0, y1 - y0),
                contour=contour,
                contour_area=area,
                full_crop=full_crop,
                top_face_crop=top_face_crop,
                threshold_mask=mask,
                approx_polygon=approx,
                shape_label=shape_label,
                face_center=face_center,
                face_radius=face_radius,
            )
            scored_rois.append((score, roi))

        scored_rois.sort(key=lambda item: item[0], reverse=True)
        return scored_rois

    def detect_dice_rois(
        self, frame: np.ndarray, background: Optional[np.ndarray] = None
    ) -> list[DiceROI]:
        """Localiza todos os dados válidos no frame, ordenados espacialmente (cima->baixo, esquerda->direita)."""
        scored_rois = self._score_and_collect_rois(frame, background=background)
        if not scored_rois:
            return []

        limit = max(1, int(self.config.max_dice_count))
        selected = [roi for _, roi in scored_rois[:limit]]
        # Ordena espacialmente em linhas horizontais para numeração estável (#1, #2, ...)
        selected.sort(key=lambda r: (r.bbox[1] // 40, r.bbox[0]))
        return selected

    def detect_dice_roi(
        self, frame: np.ndarray, background: Optional[np.ndarray] = None
    ) -> Optional[DiceROI]:
        """Localiza o contorno principal do dado e retorna seu bounding box e o recorte da face superior."""
        scored_rois = self._score_and_collect_rois(frame, background=background)
        if not scored_rois:
            return None
        return scored_rois[0][1]

