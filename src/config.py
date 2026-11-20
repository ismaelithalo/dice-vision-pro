"""Configurações globais e parâmetros ajustáveis do Dice Vision Pro (MVP)."""

from __future__ import annotations

from dataclasses import dataclass, field


def detect_gpu_available() -> bool:
    """Detecta automaticamente se há aceleração CUDA disponível via PyTorch.

    Retorna False como fallback seguro caso o PyTorch não esteja instalado
    ou nenhuma GPU compatível com CUDA seja encontrada.
    """
    try:
        import torch  # type: ignore[import-not-found]

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def get_hardware_info(use_gpu: bool) -> str:
    """Retorna uma descrição legível do dispositivo de inferência ativo (GPU CUDA ou CPU)."""
    if not use_gpu:
        if detect_gpu_available():
            return "CPU (forcado via CLI)"
        return "CPU (CUDA indisponivel)"

    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"GPU: {name} (CUDA)"
    except Exception:
        pass
    return "GPU (CUDA)"


# Flag padrão de GPU com detecção automática e fallback seguro para CPU (False)
AUTO_DETECTED_GPU: bool = detect_gpu_available()
USE_GPU: bool = AUTO_DETECTED_GPU if AUTO_DETECTED_GPU else False


@dataclass
class DiceConfig:
    """Parâmetros ajustáveis de captura, segmentação, estabilidade e OCR."""

    # --- Captura de Vídeo ---
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720

    # --- Detecção de Estabilidade (Anti-Blur) ---
    stability_duration_sec: float = 0.6
    motion_pixel_diff_threshold: int = 25
    motion_changed_ratio_threshold: float = 0.004  # Fração máxima de pixels alterados para considerar parado

    # --- Segmentação e Contorno do Dado ---
    blur_kernel_size: tuple[int, int] = (5, 5)
    use_adaptive_threshold: bool = False  # Se False, utiliza contraste local + distância de cor (Lab) + Otsu
    adaptive_block_size: int = 19
    adaptive_c: int = 4
    min_foreground_diff: int = 28  # Limiar mínimo de contraste (maior = menos sensível a sombras e ruídos)
    min_contour_area: int = 1200
    max_contour_area: int = 280000
    max_dice_count: int = 12  # Limite máximo de dados segmentados simultaneamente por quadro
    roi_padding_px: int = 8

    # --- Recorte Central da Face Superior (Redução de Faces Laterais) ---
    center_crop_ratio: float = 0.55  # Fração central do polígono preservada para isolar apenas a face superior
    apply_circular_mask: bool = True

    # --- Pré-processamento & OCR Rotacionado ---
    use_gpu: bool = field(default_factory=lambda: USE_GPU)
    clahe_clip_limit: float = 2.5
    clahe_tile_grid_size: tuple[int, int] = (8, 8)
    ocr_target_size: tuple[int, int] = (220, 220)
    rotation_step_deg: int = 30
    ocr_allowlist: str = "0123456789"
    min_dice_value: int = 1
    max_dice_value: int = 20
    min_ocr_confidence: float = 0.25


DEFAULT_CONFIG = DiceConfig()

