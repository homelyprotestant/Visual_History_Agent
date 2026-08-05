"""Device resolution and OpenAI CLIP load (same pattern as NEHM ``nehm_pipeline.device``)."""

from __future__ import annotations

import gc
import logging
from typing import Any, Callable, Tuple

import torch

logger = logging.getLogger(__name__)


def mps_is_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def resolve_device(preference: str = "mps") -> torch.device:
    pref = preference.lower().strip()
    if pref == "cpu":
        logger.info("Using device: cpu (requested)")
        return torch.device("cpu")
    if pref == "cuda":
        if torch.cuda.is_available():
            logger.info("Using device: cuda:0")
            return torch.device("cuda:0")
        logger.warning("CUDA requested but not available; falling back.")
    elif pref == "mps":
        if mps_is_available():
            logger.info("Using device: mps")
            return torch.device("mps")
        logger.warning("MPS requested but not available; falling back.")

    if mps_is_available():
        logger.info("Using device: mps (fallback)")
        return torch.device("mps")
    if torch.cuda.is_available():
        logger.info("Using device: cuda:0 (fallback)")
        return torch.device("cuda:0")
    logger.info("Using device: cpu (fallback)")
    return torch.device("cpu")


def configure_training_runtime(device: torch.device) -> None:
    """Tune PyTorch for stability on Apple Silicon / CUDA before heavy training."""
    gc.collect()
    if device.type == "mps":
        try:
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        # timm EVA/DINOv3 uses F.scaled_dot_product_attention when fused attn is on; MPS often
        # errors or misbehaves there. Disable before timm.create_model (runs in main() after imports).
        try:
            from timm.layers import set_fused_attn

            set_fused_attn(False)
            logger.info(
                "MPS: timm fused attention disabled (explicit attention matmul path; avoids SDPA issues).",
            )
        except ModuleNotFoundError:
            # timm is optional for workflows that do not instantiate timm models.
            pass
        except Exception as e:
            logger.warning("MPS: could not disable timm fused attention: %s", e)
    elif device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass


def load_openai_clip(model_name: str, device: torch.device) -> Tuple[Any, Callable]:
    import clip

    if device.type == "mps":
        model, preprocess = clip.load(model_name, device="cpu")
        model = model.float().to(device)
        return model, preprocess

    model, preprocess = clip.load(model_name, device=str(device))
    model = model.float()
    return model, preprocess
