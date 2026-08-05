"""LongCLIP-B student model from the curriculum trainer notebook.

Loads the patched LongCLIP-B visual tower (1120² dynamic positional grid), the
decoder-style student head (768 → 2304 + logits), and training checkpoints
written by ``Stable/Colab/Copy of LongCLIP_Student_Trainer_Curriculum.ipynb``.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import os
import sys
import types
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from visual_history_agent.models.artifact_paths import resolve_existing_file
from visual_history_agent.models.vhm_visual import build_vhm_visual_from_state

logger = logging.getLogger(__name__)

# The painting corpus includes legitimate very large TIFF/JPEG files. Downstream
# callers can still cap decode size with ``max_decode_side``; this disables only
# PIL's pre-decode pixel-count guard.
Image.MAX_IMAGE_PIXELS = None

EXPECTED_IMAGE_DIM = 768
EXPECTED_TEACHER_DIM = 2304
DEFAULT_GLOBAL_IMAGE_SIZE = 1120
DEFAULT_TILE_IMAGE_SIZE = 224

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)


def _vit_amp_context(device: torch.device, use_autocast: bool):
    """Autocast wrapper; MPS and disabled paths use a no-op context."""
    if not use_autocast:
        return nullcontext()
    dev = str(device.type).lower()
    if dev == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if dev == "cpu":
        return torch.autocast(device_type="cpu", dtype=torch.float16)
    return nullcontext()


def resolve_longclip_root(explicit: Optional[Path] = None) -> Path:
    """Find the vendored LongCLIP package (Stable/Colab copy or LONGCLIP_ROOT)."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(Path(explicit).expanduser().resolve())
    env = (os.environ.get("LONGCLIP_ROOT") or "").strip()
    if env:
        candidates.append(Path(env).expanduser().resolve())

    # visual_history_agent/models → project root is parents[2]
    project_root = Path(__file__).resolve().parents[2]
    study_root = project_root.parent
    # Prefer in-repo / env locations only (no machine-local absolute paths).
    candidates.extend(
        [
            project_root / "third_party" / "LongCLIP",
            project_root / "vendor" / "LongCLIP",
            project_root / "Colab" / "LongCLIP",
            study_root / "Stable" / "Colab" / "LongCLIP",
        ]
    )
    for root in candidates:
        if (root / "model" / "longclip.py").is_file():
            return root
    raise FileNotFoundError(
        "LongCLIP package not found. Set LONGCLIP_ROOT or pass --longclip-root "
        "pointing at a folder containing model/longclip.py"
    )


def resolve_longclip_checkpoint(longclip_root: Path, explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    default = longclip_root / "checkpoints" / "longclip-B.pt"
    if default.is_file():
        return default
    raise FileNotFoundError(
        f"LongCLIP-B checkpoint not found at {default}. Pass --longclip-checkpoint."
    )


def load_longclip_module(longclip_root: Path):
    pkg_dir = longclip_root / "model"
    init_py = pkg_dir / "__init__.py"
    longclip_py = pkg_dir / "longclip.py"
    model_longclip_py = pkg_dir / "model_longclip.py"
    simple_tok_py = pkg_dir / "simple_tokenizer.py"
    missing = [p for p in (pkg_dir, init_py, longclip_py, model_longclip_py, simple_tok_py) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "LongCLIP package incomplete; missing:\n  " + "\n  ".join(str(m) for m in missing)
        )
    if str(longclip_root) not in sys.path:
        sys.path.insert(0, str(longclip_root))
    for key in [k for k in list(sys.modules) if k == "model" or k.startswith("model.")]:
        del sys.modules[key]
    if "pkg_resources" not in sys.modules:
        try:
            import pkg_resources  # noqa: F401
        except ModuleNotFoundError:
            import packaging
            import packaging.version as _packaging_version

            if not hasattr(packaging, "version"):
                packaging.version = _packaging_version  # type: ignore[attr-defined]
            shim = types.ModuleType("pkg_resources")
            shim.packaging = packaging
            sys.modules["pkg_resources"] = shim
    # Register ``model`` as a package without executing __init__.py (it eagerly
    # imports longclip, which needs pkg_resources on some vendored copies).
    pkg_mod = types.ModuleType("model")
    pkg_mod.__path__ = [str(pkg_dir)]  # type: ignore[attr-defined]
    sys.modules["model"] = pkg_mod
    sub_spec = importlib.util.spec_from_file_location("model.longclip", str(longclip_py))
    if sub_spec is None or sub_spec.loader is None:
        raise ImportError(f"Could not build module spec for {longclip_py}")
    sub_mod = importlib.util.module_from_spec(sub_spec)
    sys.modules["model.longclip"] = sub_mod
    sub_spec.loader.exec_module(sub_mod)
    return sub_mod


def pil_to_tensor_norm(img: Image.Image, *, mean: torch.Tensor = CLIP_MEAN, std: torch.Tensor = CLIP_STD) -> torch.Tensor:
    """RGB PIL → CLIP-normalized CHW float tensor without importing torchvision."""
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)
    return (x - mean) / std


def load_rgb_image_capped(path: Path, *, max_decode_side: int | None = None) -> Image.Image:
    """Load RGB. If ``max_decode_side`` is set, downscale during decode when possible."""
    path = resolve_existing_file(path)
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as im:
        if max_decode_side is not None and int(max_decode_side) > 0:
            im.thumbnail((int(max_decode_side), int(max_decode_side)), Image.BILINEAR)
        return im.convert("RGB")


def center_crop_square_pil(
    img: Image.Image, size: int, *, pre_crop_margin_px: int = 1
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Return the center ``size×size`` PIL crop and its box on the source image."""
    img = img.convert("RGB")
    ox, oy = 0, 0
    w, h = img.size
    if pre_crop_margin_px > 0 and w > 2 * pre_crop_margin_px and h > 2 * pre_crop_margin_px:
        ox, oy = pre_crop_margin_px, pre_crop_margin_px
        img = img.crop((ox, oy, w - pre_crop_margin_px, h - pre_crop_margin_px))
        w, h = img.size
    short = min(w, h)
    scale = size / short
    new_w = max(size, int(round(w * scale)))
    new_h = max(size, int(round(h * scale)))
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    crop = resized.crop((left, top, left + size, top + size))
    src_left = ox + int(round(left / scale))
    src_top = oy + int(round(top / scale))
    src_right = ox + int(round((left + size) / scale))
    src_bottom = oy + int(round((top + size) / scale))
    return crop, (src_left, src_top, src_right, src_bottom)


def top_crop_square_pil(
    img: Image.Image, size: int, *, pre_crop_margin_px: int = 1
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """Scale the short side to ``size`` (up or down), then top-left ``size×size`` crop."""
    img = img.convert("RGB")
    ox, oy = 0, 0
    w, h = img.size
    if pre_crop_margin_px > 0 and w > 2 * pre_crop_margin_px and h > 2 * pre_crop_margin_px:
        ox, oy = pre_crop_margin_px, pre_crop_margin_px
        img = img.crop((ox, oy, w - pre_crop_margin_px, h - pre_crop_margin_px))
        w, h = img.size
    short = min(w, h)
    scale = size / short
    new_w = max(size, int(round(w * scale)))
    new_h = max(size, int(round(h * scale)))
    working = img.resize((new_w, new_h), Image.BICUBIC)
    left = 0
    top = 0
    crop = working.crop((left, top, left + size, top + size))
    src_left = ox + int(round(left / scale))
    src_top = oy + int(round(top / scale))
    src_right = ox + int(round((left + size) / scale))
    src_bottom = oy + int(round((top + size) / scale))
    return crop, (src_left, src_top, src_right, src_bottom)


def overlay_activation_on_image(
    base: Image.Image,
    activation: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    *,
    cmap_name: str = "magma",
    alpha: float = 0.82,
    clip_percentile: float = 99.0,
) -> Tuple[Image.Image, np.ndarray]:
    """Upsample a patch grid onto ``crop_box`` of ``base`` and alpha-blend."""
    import matplotlib.cm as cm

    act = np.asarray(activation, dtype=np.float32)
    v = float(np.percentile(np.abs(act), clip_percentile))
    heat = np.clip(act, -v, v)
    heat_norm = (heat - heat.min()) / max(heat.max() - heat.min(), 1e-12)
    l, t, r, b = (int(v) for v in crop_box)
    box_w = max(1, r - l)
    box_h = max(1, b - t)
    heat_up = Image.fromarray((heat_norm * 255).astype(np.uint8)).resize((box_w, box_h), Image.BILINEAR)
    rgba = cm.get_cmap(cmap_name)(np.asarray(heat_up, dtype=np.float32) / 255.0)
    overlay = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))
    mask = Image.fromarray((rgba[..., 3] * alpha * 255).astype(np.uint8))
    out = base.convert("RGB").copy()
    out.paste(overlay, (l, t), mask)
    return out, heat_norm


def fit_patch_activations(
    patch_tokens: np.ndarray,
    atom_image: np.ndarray,
    *,
    method: str = "projection",
) -> Tuple[np.ndarray, float]:
    """
    Map an atom image-block vector onto per-patch activations.

    ``patch_tokens``: ``(n_patches, 768)``
    ``method``: ``projection`` (stable default), ``lstsq``, or ``nnls``.
    """
    p = np.asarray(patch_tokens, dtype=np.float32)
    target = np.asarray(atom_image, dtype=np.float32).reshape(-1)
    if p.ndim != 2 or p.shape[1] != target.shape[0]:
        raise ValueError(f"patch_tokens {p.shape} incompatible with atom_image {target.shape}")

    method = str(method).lower().strip()
    if method == "projection":
        denom = float(np.dot(target, target) + 1e-12)
        coef = (p @ target) / denom
        residual = float(np.linalg.norm(p.T @ coef - target))
    elif method == "lstsq":
        coef, _, _, _ = np.linalg.lstsq(p.T, target, rcond=None)
        residual = float(np.linalg.norm(p.T @ coef - target))
    elif method == "nnls":
        try:
            from scipy.optimize import nnls
        except ImportError as exc:
            raise ImportError("scipy is required for method='nnls'") from exc
        coef, residual = nnls(p.T, target)
        residual = float(residual)
    else:
        raise ValueError(f"Unknown fit method {method!r}; use projection, lstsq, or nnls.")
    return coef.astype(np.float32, copy=False), residual


def fit_atoms_per_atom(
    patch_embeddings: np.ndarray,
    dictionary_rows: np.ndarray,
    *,
    method: str = "projection",
    show_progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit each dictionary atom independently onto the patch embedding matrix.

    For atom ``j``, finds spatial weights ``w[:, j]`` such that
    ``patch_embeddings.T @ w[:, j] ≈ dictionary_rows[j]`` using
    ``fit_patch_activations`` (``projection``, ``lstsq``, or ``nnls``).

    Returns ``weights`` ``(n_patches, n_atoms)`` and per-atom residuals.
    """
    z = np.asarray(patch_embeddings, dtype=np.float32)
    d = np.asarray(dictionary_rows, dtype=np.float32)
    if z.ndim != 2 or d.ndim != 2:
        raise ValueError(f"expected 2-D arrays, got Z={z.shape} D={d.shape}")
    if z.shape[1] != d.shape[1]:
        raise ValueError(f"feature dim mismatch: Z={z.shape} D={d.shape}")

    n_patches, n_atoms = z.shape[0], int(d.shape[0])
    weights = np.zeros((n_patches, n_atoms), dtype=np.float32)
    residuals = np.zeros(n_atoms, dtype=np.float32)

    atom_iter = range(n_atoms)
    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = None  # type: ignore[assignment,misc]
        if tqdm is not None:
            atom_iter = tqdm(atom_iter, desc="per-atom fit", unit="atom", dynamic_ncols=True)

    for j in atom_iter:
        coef, residual = fit_patch_activations(z, d[j], method=method)
        weights[:, j] = coef
        residuals[j] = residual
    return weights, residuals


def patch_grid_for_image_size(image_size: int, *, patch_size: int = 16) -> Tuple[int, int, int]:
    """Return ``(grid_h, grid_w, n_patches)`` for a square ViT input."""
    size = int(image_size)
    if size <= 0 or size % patch_size != 0:
        raise ValueError(f"image_size must be a positive multiple of {patch_size}, got {image_size!r}")
    grid = size // patch_size
    return grid, grid, grid * grid


def release_torch_memory(device: torch.device, *, aggressive: bool = False) -> None:
    """Best-effort VRAM cleanup after a heavy forward pass.

    ``aggressive=False`` (default): only empty the device cache. Avoid calling
    ``gc.collect()`` every batch — on MPS that pattern often causes severe slowdowns.
    """
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and torch.backends.mps.is_available():
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    if aggressive:
        import gc

        gc.collect()


def center_crop_square(img: Image.Image, size: int, *, pre_crop_margin_px: int = 1) -> torch.Tensor:
    """Match the curriculum trainer: optional 1px border trim, short-side scale, center crop."""
    crop, _ = center_crop_square_pil(img, size, pre_crop_margin_px=pre_crop_margin_px)
    return pil_to_tensor_norm(crop)


def top_crop_square(img: Image.Image, size: int, *, pre_crop_margin_px: int = 1) -> torch.Tensor:
    """Scale short side to ``size`` (up or down), then top-left ``size×size`` crop."""
    crop, _ = top_crop_square_pil(img, size, pre_crop_margin_px=pre_crop_margin_px)
    return pil_to_tensor_norm(crop)


def _interpolate_visual_positional_embedding(
    vit: nn.Module, grid_h: int, grid_w: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    cache = getattr(vit, "_interpolated_pos_cache", None)
    if cache is None:
        cache = {}
        vit._interpolated_pos_cache = cache
    cache_key = (int(grid_h), int(grid_w), str(device), str(dtype))
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    pos = vit.positional_embedding.to(device=device, dtype=dtype)
    cls_pos = pos[:1]
    patch_pos = pos[1:]
    old_grid = int(round(math.sqrt(patch_pos.shape[0])))
    if old_grid * old_grid != patch_pos.shape[0]:
        raise ValueError(f"Cannot infer square visual positional grid from {patch_pos.shape}")
    if grid_h == old_grid and grid_w == old_grid:
        out = pos.unsqueeze(0)
        cache[cache_key] = out
        return out
    patch_pos = patch_pos.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2).float()
    # MPS has no bicubic upsample; bilinear stays on-device and is fine for pos-embed resize.
    interp_mode = "bilinear" if device.type == "mps" else "bicubic"
    patch_pos = F.interpolate(
        patch_pos, size=(grid_h, grid_w), mode=interp_mode, align_corners=False
    )
    patch_pos = patch_pos.to(dtype=dtype).permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
    out = torch.cat([cls_pos.reshape(1, 1, -1), patch_pos], dim=1)
    cache[cache_key] = out
    return out


def patch_longclip_visual_for_dynamic_grids(
    visual: nn.Module,
    *,
    use_dynamic_position_interpolation: bool = True,
    use_experimental_2d_rope: bool = False,
) -> int:
    patch_size = int(visual.conv1.kernel_size[0])

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def _apply_token_space_2d_rope(tokens: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        cls, patch = tokens[:, :1], tokens[:, 1:]
        b, _, d = patch.shape
        d_axis = (d // 4) * 2
        if d_axis < 2:
            return tokens
        patch = patch.reshape(b, grid_h, grid_w, d)
        y_part = patch[..., :d_axis]
        x_part = patch[..., d_axis : 2 * d_axis]
        rest = patch[..., 2 * d_axis :]
        inv_freq = 1.0 / (10000 ** (torch.arange(0, d_axis, 2, device=patch.device, dtype=torch.float32) / d_axis))
        y_pos = torch.arange(grid_h, device=patch.device, dtype=torch.float32)
        x_pos = torch.arange(grid_w, device=patch.device, dtype=torch.float32)
        y_freq = torch.einsum("i,j->ij", y_pos, inv_freq)
        x_freq = torch.einsum("i,j->ij", x_pos, inv_freq)
        y_emb = torch.repeat_interleave(y_freq, 2, dim=-1).view(1, grid_h, 1, d_axis).to(dtype=patch.dtype)
        x_emb = torch.repeat_interleave(x_freq, 2, dim=-1).view(1, 1, grid_w, d_axis).to(dtype=patch.dtype)
        y_part = (y_part * y_emb.cos()) + (_rotate_half(y_part) * y_emb.sin())
        x_part = (x_part * x_emb.cos()) + (_rotate_half(x_part) * x_emb.sin())
        patch = torch.cat([y_part, x_part, rest], dim=-1).reshape(b, grid_h * grid_w, d)
        return torch.cat([cls, patch], dim=1)

    def _dynamic_vit_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        grid_h, grid_w = int(x.shape[-2]), int(x.shape[-1])
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype).to(x.device) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([cls, x], dim=1)
        if use_dynamic_position_interpolation:
            x = x + _interpolate_visual_positional_embedding(self, grid_h, grid_w, x.dtype, x.device)
        else:
            x = x + self.positional_embedding.to(dtype=x.dtype, device=x.device)
        if use_experimental_2d_rope:
            x = _apply_token_space_2d_rope(x, grid_h, grid_w)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        return self.ln_post(x[:, 0, :])

    visual.forward = types.MethodType(_dynamic_vit_forward, visual)
    return patch_size


def _prepare_vit_ln_pre_nld(
    visual: nn.Module,
    x: torch.Tensor,
    *,
    use_dynamic_position_interpolation: bool = True,
    use_experimental_2d_rope: bool = False,
) -> Tuple[torch.Tensor, int, int]:
    """Build ViT token sequence and return ``(L, N, D)`` after ``ln_pre``."""
    x = visual.conv1(x)
    grid_h, grid_w = int(x.shape[-2]), int(x.shape[-1])
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(x.dtype).to(x.device) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls, x], dim=1)
    if use_dynamic_position_interpolation:
        x = x + _interpolate_visual_positional_embedding(visual, grid_h, grid_w, x.dtype, x.device)
    else:
        x = x + visual.positional_embedding.to(dtype=x.dtype, device=x.device)
    if use_experimental_2d_rope:
        inv_freq = 1.0 / (
            10000
            ** (
                torch.arange(0, (x.shape[-1] // 4) * 2, 2, device=x.device, dtype=torch.float32)
                / max((x.shape[-1] // 4) * 2, 2)
            )
        )
        y_pos = torch.arange(grid_h, device=x.device, dtype=torch.float32)
        x_pos = torch.arange(grid_w, device=x.device, dtype=torch.float32)
        cls_tok, patch = x[:, :1], x[:, 1:]
        b, _, d = patch.shape
        d_axis = (d // 4) * 2
        if d_axis >= 2:
            patch = patch.reshape(b, grid_h, grid_w, d)
            y_part = patch[..., :d_axis]
            x_part = patch[..., d_axis : 2 * d_axis]
            rest = patch[..., 2 * d_axis :]
            y_freq = torch.einsum("i,j->ij", y_pos, inv_freq)
            x_freq = torch.einsum("i,j->ij", x_pos, inv_freq)
            y_emb = torch.repeat_interleave(y_freq, 2, dim=-1).view(1, grid_h, 1, d_axis).to(dtype=patch.dtype)
            x_emb = torch.repeat_interleave(x_freq, 2, dim=-1).view(1, 1, grid_w, d_axis).to(dtype=patch.dtype)

            def _rotate_half(t: torch.Tensor) -> torch.Tensor:
                t_even = t[..., 0::2]
                t_odd = t[..., 1::2]
                return torch.stack((-t_odd, t_even), dim=-1).flatten(-2)

            y_part = (y_part * y_emb.cos()) + (_rotate_half(y_part) * y_emb.sin())
            x_part = (x_part * x_emb.cos()) + (_rotate_half(x_part) * x_emb.sin())
            patch = torch.cat([y_part, x_part, rest], dim=-1).reshape(b, grid_h * grid_w, d)
            x = torch.cat([cls_tok, patch], dim=1)
    x = visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    return x, grid_h, grid_w


def _extract_cls_patch_attn_per_head(
    mha: nn.MultiheadAttention,
    x: torch.Tensor,
    *,
    patch_start: int = 1,
) -> np.ndarray:
    """
    CLS→patch softmax attention for each head without materializing ``L×L`` weights.

    ``x``: ``(L, N, E)`` — typically ``block.ln_1(tokens)``.
    Returns float32 ``(num_heads, n_patches)`` on CPU (batch index 0 when ``N >= 1``).
    """
    embed_dim = int(mha.embed_dim)
    num_heads = int(mha.num_heads)
    head_dim = embed_dim // num_heads
    if head_dim * num_heads != embed_dim:
        raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")

    seq_len, batch_size, _ = x.shape
    q, k, _v = F.linear(x, mha.in_proj_weight, mha.in_proj_bias).chunk(3, dim=-1)
    q_cls = q[patch_start - 1 : patch_start].contiguous().view(1, batch_size * num_heads, head_dim).transpose(0, 1)
    k_all = k.contiguous().view(seq_len, batch_size * num_heads, head_dim).transpose(0, 1)
    scale = head_dim ** -0.5
    logits = torch.bmm(q_cls, k_all.transpose(1, 2)) * scale
    probs = F.softmax(logits, dim=-1)[:, 0, patch_start:]
    return probs.reshape(batch_size, num_heads, -1)[0].detach().float().cpu().numpy().astype(np.float32, copy=False)


@torch.inference_mode()
def forward_vit_cls_head_attention_and_patch_student(
    visual: nn.Module,
    head: StudentHead,
    x: torch.Tensor,
    *,
    use_dynamic_position_interpolation: bool = True,
    use_experimental_2d_rope: bool = False,
    use_autocast: bool = True,
    head_batch_size: int = 64,
) -> Tuple[list[Dict[str, Any]], np.ndarray, int, int]:
    """
    One ViT forward with memory-efficient per-layer/per-head CLS→patch attention maps
    plus batched student-head patch embeddings.

    Avoids ``need_weights=True`` on 70×70 grids (~4901 tokens), which would allocate
    ~1 GB per layer on MPS/CUDA.
    """
    x_tokens, grid_h, grid_w = _prepare_vit_ln_pre_nld(
        visual,
        x,
        use_dynamic_position_interpolation=use_dynamic_position_interpolation,
        use_experimental_2d_rope=use_experimental_2d_rope,
    )
    attn_records: list[Dict[str, Any]] = []
    with _vit_amp_context(x.device, use_autocast):
        for layer_i, block in enumerate(visual.transformer.resblocks):
            q = block.ln_1(x_tokens)
            cls_patch = _extract_cls_patch_attn_per_head(block.attn, q)
            for head_i in range(cls_patch.shape[0]):
                attn_records.append(
                    {
                        "layer": int(layer_i),
                        "head": int(head_i),
                        "attention": cls_patch[head_i].reshape(int(grid_h), int(grid_w)),
                    }
                )
            attn_out = block.attn(q, q, q, need_weights=False)[0]
            x_tokens = x_tokens + attn_out
            x_tokens = x_tokens + block.mlp(block.ln_2(x_tokens))

    x_bld = x_tokens.permute(1, 0, 2)
    patch_tokens = visual.ln_post(x_bld[:, 1:, :]).float()[0]
    patch_chunks: list[np.ndarray] = []
    batch_size = max(1, int(head_batch_size))
    for start in range(0, patch_tokens.shape[0], batch_size):
        batch = patch_tokens[start : start + batch_size]
        batch = F.normalize(batch, dim=-1)
        z, _ = head(batch)
        patch_chunks.append(z.detach().cpu().numpy().astype(np.float32, copy=False))
    patch_student = np.concatenate(patch_chunks, axis=0)
    return attn_records, patch_student, int(grid_h), int(grid_w)


@torch.inference_mode()
def encode_visual_patch_tokens(
    visual: nn.Module,
    x: torch.Tensor,
    *,
    use_dynamic_position_interpolation: bool = True,
    use_experimental_2d_rope: bool = False,
    use_autocast: bool = True,
) -> Tuple[torch.Tensor, int, int]:
    """Extract post-transformer ViT patch tokens, shape ``(B, grid_h*grid_w, width)``."""
    x, grid_h, grid_w = _prepare_vit_ln_pre_nld(
        visual,
        x,
        use_dynamic_position_interpolation=use_dynamic_position_interpolation,
        use_experimental_2d_rope=use_experimental_2d_rope,
    )
    with _vit_amp_context(x.device, use_autocast):
        x = visual.transformer(x)
    x = x.permute(1, 0, 2)
    patch_tokens = visual.ln_post(x[:, 1:, :]).float()
    return patch_tokens, grid_h, grid_w


@torch.inference_mode()
def encode_visual_cls_spatial_maps(
    visual: nn.Module,
    x: torch.Tensor,
    *,
    use_dynamic_position_interpolation: bool = True,
    use_experimental_2d_rope: bool = False,
    use_autocast: bool = True,
    attn_last_n_avg: int = 4,
) -> Dict[str, Any]:
    """
    DINO-style CLS→patch attention maps plus CLS–patch cosine similarity at 768-D.

    Returns float32 numpy arrays shaped ``(grid_h, grid_w)``:
    - ``attention_last``: last transformer layer
    - ``attention_last_n_mean``: mean of the last ``attn_last_n_avg`` layers
    - ``cosine_768``: ``cos(ln_post(CLS), ln_post(patch_i))``
    """
    x, grid_h, grid_w = _prepare_vit_ln_pre_nld(
        visual,
        x,
        use_dynamic_position_interpolation=use_dynamic_position_interpolation,
        use_experimental_2d_rope=use_experimental_2d_rope,
    )
    attn_layers: list[torch.Tensor] = []
    with _vit_amp_context(x.device, use_autocast):
        for block in visual.transformer.resblocks:
            q = block.ln_1(x)
            attn_out, attn_w = block.attn(q, q, q, need_weights=True, average_attn_weights=True)
            x = x + attn_out
            x = x + block.mlp(block.ln_2(x))
            attn_layers.append(attn_w)

    x = x.permute(1, 0, 2)
    cls_token = visual.ln_post(x[:, 0, :]).float()
    patch_tokens = visual.ln_post(x[:, 1:, :]).float()

    cls_attn = [attn_layers[i][0, 0, 1:].float().cpu().numpy() for i in range(len(attn_layers))]
    n_avg = max(1, min(int(attn_last_n_avg), len(cls_attn)))
    attention_last = cls_attn[-1].reshape(grid_h, grid_w).astype(np.float32, copy=False)
    attention_last_n_mean = (
        np.mean([cls_attn[i].reshape(grid_h, grid_w) for i in range(-n_avg, 0)], axis=0)
        .astype(np.float32, copy=False)
    )
    cosine_768 = (
        F.cosine_similarity(cls_token, patch_tokens, dim=-1)[0]
        .detach()
        .cpu()
        .numpy()
        .reshape(grid_h, grid_w)
        .astype(np.float32, copy=False)
    )

    return {
        "attention_last": attention_last,
        "attention_last_n_mean": attention_last_n_mean,
        "cosine_768": cosine_768,
        "grid_h": int(grid_h),
        "grid_w": int(grid_w),
        "n_layers": int(len(attn_layers)),
        "attn_last_n_avg": int(n_avg),
    }


class StudentHead(nn.Module):
    """Decoder-style head: 768-D visual features → 2304-D teacher space + logits."""

    def __init__(self, in_dim: int, teacher_dim: int, num_classes: int, dropout: float = 0.20):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1536),
            nn.LayerNorm(1536),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1536, teacher_dim),
        )
        self.classifier = nn.Linear(teacher_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.net(x)
        logits = self.classifier(z)
        return z, logits


class LongCLIPStudent(nn.Module):
    def __init__(self, visual_module: nn.Module, visual_dim: int, teacher_dim: int, num_classes: int):
        super().__init__()
        self.visual = visual_module
        self.head = StudentHead(visual_dim, teacher_dim, num_classes)

    def _encode_visual(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.visual(x)
        if isinstance(feats, (tuple, list)):
            feats = feats[0]
        return feats.float()

    def forward(
        self, x_global: torch.Tensor, x_tiles: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (visual_feats_l2, z_2304, logits)."""
        global_feats = self._encode_visual(x_global)
        feats_to_average = [global_feats]
        if x_tiles is not None and x_tiles.numel() > 0:
            b, t, c, h, w = x_tiles.shape
            tile_feats = self._encode_visual(x_tiles.reshape(b * t, c, h, w)).reshape(b, t, -1).mean(dim=1)
            feats_to_average.append(tile_feats)
        feats = torch.stack(feats_to_average, dim=1).mean(dim=1)
        feats = F.normalize(feats.float(), dim=-1)
        z, logits = self.head(feats)
        return feats, z, logits


def _num_classes_from_state(state: Dict[str, torch.Tensor]) -> int:
    key = "head.classifier.weight"
    if key not in state:
        raise KeyError(f"Checkpoint state missing {key!r}; cannot infer num_classes")
    return int(state[key].shape[0])


def _read_student_checkpoint(checkpoint_path: Path) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state") or ckpt.get("state_dict")
    if state is None:
        raise KeyError(f"Checkpoint has no model_state/state_dict: {ckpt_path}")
    return ckpt, state


def _student_checkpoint_meta(
    ckpt: Dict[str, Any],
    *,
    ckpt_path: Path,
    patch_size: int,
) -> Dict[str, Any]:
    cfg = ckpt.get("config") or {}
    return {
        "checkpoint_path": str(ckpt_path),
        "stage": ckpt.get("stage"),
        "epoch": ckpt.get("epoch"),
        "run_id": ckpt.get("run_id"),
        "target_column": ckpt.get("target_column"),
        "label_to_id": ckpt.get("label_to_id"),
        "teacher_dim": int(ckpt.get("teacher_dim", EXPECTED_TEACHER_DIM)),
        "visual_dim": int(ckpt.get("visual_dim", EXPECTED_IMAGE_DIM)),
        "num_classes": _num_classes_from_state(ckpt.get("model_state") or ckpt.get("state_dict") or {}),
        "global_image_size": int(cfg.get("global_image_size", DEFAULT_GLOBAL_IMAGE_SIZE)),
        "use_dynamic_position_interpolation": bool(cfg.get("use_dynamic_position_interpolation", True)),
        "use_experimental_2d_rope": bool(cfg.get("use_experimental_2d_rope", False)),
        "patch_size": int(patch_size),
        "architecture": "VHM-B/16",
    }


def _build_visual_tower(
    ckpt: Dict[str, Any],
    state: Dict[str, torch.Tensor],
    *,
    longclip_root: Optional[Path] = None,
    longclip_checkpoint: Optional[Path] = None,
    device: torch.device,
    strict: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Build the vendored VHM visual tower and load weights from the student checkpoint.

    ``longclip_root`` / ``longclip_checkpoint`` are accepted for API compatibility but
    ignored — architecture is in-repo and weights come only from ``VHM-B-16.pth``.
    """
    del longclip_root, longclip_checkpoint  # unused; kept for call-site compatibility

    cfg = ckpt.get("config") or {}
    global_image_size = int(cfg.get("global_image_size", DEFAULT_GLOBAL_IMAGE_SIZE))
    use_dynamic = bool(cfg.get("use_dynamic_position_interpolation", True))
    use_rope = bool(cfg.get("use_experimental_2d_rope", False))

    visual_state = {k: v for k, v in state.items() if k.startswith("visual.")}
    if not visual_state:
        raise KeyError("Checkpoint state has no visual.* weights")
    visual = build_vhm_visual_from_state(visual_state, strict=bool(strict))
    visual = visual.float().eval()

    patch_size = patch_longclip_visual_for_dynamic_grids(
        visual,
        use_dynamic_position_interpolation=use_dynamic,
        use_experimental_2d_rope=use_rope,
    )
    if global_image_size % patch_size != 0:
        raise ValueError(
            f"global_image_size={global_image_size} must be divisible by patch_size={patch_size}"
        )

    visual = visual.to(device).eval()
    ckpt_path = Path(ckpt.get("checkpoint_path", "")) if ckpt.get("checkpoint_path") else Path(".")
    meta = _student_checkpoint_meta(
        ckpt,
        ckpt_path=ckpt_path,
        patch_size=patch_size,
    )
    return visual, meta


def load_student_head_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    strict: bool = True,
) -> Tuple[StudentHead, Dict[str, Any]]:
    """Load only the decoder head (768 -> 2304 + logits)."""
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    ckpt, state = _read_student_checkpoint(ckpt_path)
    teacher_dim = int(ckpt.get("teacher_dim", EXPECTED_TEACHER_DIM))
    visual_dim = int(ckpt.get("visual_dim", EXPECTED_IMAGE_DIM))
    num_classes = _num_classes_from_state(state)
    head = StudentHead(visual_dim, teacher_dim, num_classes)
    head_state = {k[len("head.") :]: v for k, v in state.items() if k.startswith("head.")}
    if not head_state:
        raise KeyError("Checkpoint state has no head.* weights")
    missing, unexpected = head.load_state_dict(head_state, strict=bool(strict))
    if missing or unexpected:
        logger.warning(
            "Head load strict=%s: missing=%s unexpected=%s",
            strict,
            missing,
            unexpected,
        )
    head = head.to(device).eval()
    meta = {
        "checkpoint_path": str(ckpt_path),
        "teacher_dim": teacher_dim,
        "visual_dim": visual_dim,
        "num_classes": num_classes,
    }
    return head, meta


def load_visual_and_head_from_checkpoint(
    checkpoint_path: Path,
    *,
    longclip_root: Optional[Path] = None,
    longclip_checkpoint: Optional[Path] = None,
    device: torch.device,
    strict: bool = True,
) -> Tuple[nn.Module, StudentHead, Dict[str, Any]]:
    """Load visual tower and student head from one checkpoint read."""
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    ckpt, state = _read_student_checkpoint(ckpt_path)
    ckpt["checkpoint_path"] = str(ckpt_path)
    visual, visual_meta = _build_visual_tower(
        ckpt,
        state,
        longclip_root=longclip_root,
        longclip_checkpoint=longclip_checkpoint,
        device=device,
        strict=strict,
    )
    teacher_dim = int(ckpt.get("teacher_dim", EXPECTED_TEACHER_DIM))
    visual_dim = int(ckpt.get("visual_dim", EXPECTED_IMAGE_DIM))
    num_classes = _num_classes_from_state(state)
    head = StudentHead(visual_dim, teacher_dim, num_classes)
    head_state = {k[len("head.") :]: v for k, v in state.items() if k.startswith("head.")}
    if not head_state:
        raise KeyError("Checkpoint state has no head.* weights")
    missing, unexpected = head.load_state_dict(head_state, strict=bool(strict))
    if missing or unexpected:
        logger.warning(
            "Head load strict=%s: missing=%s unexpected=%s",
            strict,
            missing,
            unexpected,
        )
    head = head.to(device).eval()
    meta = {
        **visual_meta,
        "teacher_dim": teacher_dim,
        "visual_dim": visual_dim,
        "num_classes": num_classes,
    }
    return visual, head, meta


@torch.inference_mode()
def encode_image_patch_student_embeddings(
    visual: nn.Module,
    head: StudentHead,
    x: torch.Tensor,
    *,
    use_dynamic_position_interpolation: bool = True,
    use_experimental_2d_rope: bool = False,
    use_autocast: bool = True,
    batch_size: int = 64,
    normalize_visual: bool = True,
    show_progress: bool = False,
    progress_desc: str = "ViT + student head",
) -> Tuple[np.ndarray, int, int]:
    """
    One forward pass: image -> ViT patch tokens -> student head -> 2304-D per patch.

    Keeps patch tokens on device between ViT and head (no full CPU round-trip).
    Returns ``(n_patches, teacher_dim)`` float32 numpy and ``(grid_h, grid_w)``.
    """
    patch_tokens, grid_h, grid_w = encode_visual_patch_tokens(
        visual,
        x,
        use_dynamic_position_interpolation=use_dynamic_position_interpolation,
        use_experimental_2d_rope=use_experimental_2d_rope,
        use_autocast=use_autocast,
    )
    patches = patch_tokens[0]
    del patch_tokens

    teacher_dim = int(head.net[-1].out_features)
    n_patches = int(patches.shape[0])
    batch_size = max(1, int(batch_size))
    embeddings: list[np.ndarray] = []
    pbar = None
    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = None  # type: ignore[assignment,misc]
        if tqdm is not None:
            pbar = tqdm(total=n_patches, desc=progress_desc, unit="patch", dynamic_ncols=True)

    try:
        for start in range(0, n_patches, batch_size):
            batch = patches[start : start + batch_size]
            if normalize_visual:
                batch = F.normalize(batch, dim=-1)
            z, _ = head(batch)
            embeddings.append(z.detach().cpu().numpy().astype(np.float32, copy=False))
            if pbar is not None:
                pbar.update(int(batch.shape[0]))
    finally:
        if pbar is not None:
            pbar.close()

    out = np.concatenate(embeddings, axis=0)
    if out.shape != (n_patches, teacher_dim):
        raise RuntimeError(f"expected {(n_patches, teacher_dim)} patch embeddings, got {out.shape}")
    return out, int(grid_h), int(grid_w)


@torch.inference_mode()
def encode_patches_through_student_head(
    patch_tokens: np.ndarray,
    head: StudentHead,
    *,
    batch_size: int = 64,
    normalize_visual: bool = True,
    show_progress: bool = False,
    progress_desc: str = "student head",
) -> np.ndarray:
    """
    Run each ViT patch token through the student head individually.

    Per patch: L2-normalize 768-D token -> head -> 2304-D.
    Returns ``(n_patches, teacher_dim)``. Batching is only for speed.
    """
    patches = np.asarray(patch_tokens, dtype=np.float32)
    if patches.ndim != 2:
        raise ValueError(f"patch_tokens must be 2-D, got {patches.shape}")
    teacher_dim = int(head.net[-1].out_features)
    if patches.shape[1] != head.net[0].in_features:
        raise ValueError(f"patch_tokens {patches.shape} incompatible with head in_dim {head.net[0].in_features}")

    device = next(head.parameters()).device
    embeddings: list[np.ndarray] = []
    n_patches = int(patches.shape[0])
    batch_size = max(1, int(batch_size))
    pbar = None
    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            tqdm = None  # type: ignore[assignment,misc]
        if tqdm is not None:
            pbar = tqdm(total=n_patches, desc=progress_desc, unit="patch", dynamic_ncols=True)

    try:
        for start in range(0, n_patches, batch_size):
            batch = torch.as_tensor(patches[start : start + batch_size], device=device, dtype=torch.float32)
            if normalize_visual:
                batch = F.normalize(batch, dim=-1)
            z, _ = head(batch)
            embeddings.append(z.detach().cpu().numpy().astype(np.float32, copy=False))
            if pbar is not None:
                pbar.update(int(batch.shape[0]))
    finally:
        if pbar is not None:
            pbar.close()

    out = np.concatenate(embeddings, axis=0)
    if out.shape != (patches.shape[0], teacher_dim):
        raise RuntimeError(f"expected {(patches.shape[0], teacher_dim)} patch embeddings, got {out.shape}")
    return out


def fit_patch_embeddings_to_atoms(
    patch_embeddings: np.ndarray,
    dictionary: np.ndarray,
    *,
    method: str = "lstsq",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-patch least-squares fit of dictionary atoms onto student embeddings.

    For each patch row ``z_p`` in ``Z`` ``(n_patches, feature_dim)``, solve
    ``z_p ≈ D @ w_p`` where ``D`` is ``(feature_dim, n_atoms)`` (atoms as columns).

    Returns ``W`` ``(n_patches, n_atoms)`` and per-patch reconstruction residuals.
    """
    z = np.asarray(patch_embeddings, dtype=np.float32)
    d = np.asarray(dictionary, dtype=np.float32)
    if z.ndim != 2 or d.ndim != 2:
        raise ValueError(f"expected 2-D arrays, got Z={z.shape} D={d.shape}")
    if z.shape[1] != d.shape[0]:
        raise ValueError(f"feature dim mismatch: Z={z.shape} D={d.shape}")

    method = str(method).lower().strip()
    if method == "lstsq":
        # Z.T = D @ W.T  =>  solve for W.T with shape (n_atoms, n_patches)
        w_t, _, _, _ = np.linalg.lstsq(d, z.T, rcond=None)
        weights = w_t.T.astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unknown fit method {method!r}; use lstsq.")

    recon = weights @ d.T
    residuals = np.linalg.norm(recon - z, axis=1).astype(np.float32, copy=False)
    return weights, residuals


def load_visual_from_checkpoint(
    checkpoint_path: Path,
    *,
    longclip_root: Optional[Path] = None,
    longclip_checkpoint: Optional[Path] = None,
    device: torch.device,
    strict: bool = True,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load only the visual tower (no student head / text encoder on GPU)."""
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    ckpt, state = _read_student_checkpoint(ckpt_path)
    ckpt["checkpoint_path"] = str(ckpt_path)
    visual, meta = _build_visual_tower(
        ckpt,
        state,
        longclip_root=longclip_root,
        longclip_checkpoint=longclip_checkpoint,
        device=device,
        strict=strict,
    )
    logger.info(
        "Loaded visual tower stage=%s epoch=%s global=%d",
        meta.get("stage"),
        meta.get("epoch"),
        meta.get("global_image_size"),
    )
    return visual, meta


def load_student_from_checkpoint(
    checkpoint_path: Path,
    *,
    longclip_root: Optional[Path] = None,
    longclip_checkpoint: Optional[Path] = None,
    device: torch.device,
    strict: bool = True,
) -> Tuple[LongCLIPStudent, Dict[str, Any]]:
    """Build the student model and load weights from a curriculum-trainer ``.pth`` file."""
    ckpt_path = Path(checkpoint_path).expanduser().resolve()
    ckpt, state = _read_student_checkpoint(ckpt_path)
    ckpt["checkpoint_path"] = str(ckpt_path)

    teacher_dim = int(ckpt.get("teacher_dim", EXPECTED_TEACHER_DIM))
    visual_dim = int(ckpt.get("visual_dim", EXPECTED_IMAGE_DIM))
    num_classes = _num_classes_from_state(state)

    visual, meta = _build_visual_tower(
        ckpt,
        state,
        longclip_root=longclip_root,
        longclip_checkpoint=longclip_checkpoint,
        device=torch.device("cpu"),
        strict=strict,
    )
    student = LongCLIPStudent(visual, visual_dim, teacher_dim, num_classes)
    head_state = {k[len("head.") :]: v for k, v in state.items() if k.startswith("head.")}
    if head_state:
        missing, unexpected = student.head.load_state_dict(head_state, strict=bool(strict))
        if missing or unexpected:
            logger.warning(
                "Head load strict=%s: missing=%s unexpected=%s",
                strict,
                missing,
                unexpected,
            )

    student = student.to(device).eval()
    logger.info(
        "Loaded student checkpoint stage=%s epoch=%s classes=%d global=%d",
        meta.get("stage"),
        meta.get("epoch"),
        num_classes,
        meta.get("global_image_size"),
    )
    return student, meta
