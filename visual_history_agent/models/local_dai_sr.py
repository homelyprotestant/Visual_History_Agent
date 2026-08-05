"""Local Dai-style RGB-guided super-resolution for spatial atom-map stacks."""

from __future__ import annotations

import contextlib
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from PIL import Image

from visual_history_agent.models.classical_ksvd import (
    ClassicalKsvdConfig,
    encode_samples,
    fit_approx_dictionary,
)


@dataclass(frozen=True)
class LocalDaiConfig:
    """Geometry and sparse-model settings matching the interpretability notebook."""

    hr_side: int = 224
    lr_window: int = 10
    lr_stride: int = 3
    hr_window: int | None = None
    hr_stride: int = 3
    rgb_atoms: int = 32
    rgb_ksvd_iters: int = 8
    rgb_sparse_method: str = "elastic_net"
    rgb_omp_nonzero: int = 3
    rgb_elastic_net_alpha: float = 1e-4
    rgb_elastic_net_l1_ratio: float = 0.5
    rgb_elastic_net_max_iter: int = 1000
    rgb_sparse_codes_nonnegative: bool = False
    encode_batch_size: int = 32768
    hann_blend: bool = False
    seed: int = 123


@dataclass(frozen=True)
class SpatialKsvdDenoiseConfig:
    """Spatial K-SVD denoising settings for low-resolution activation maps."""

    window: int = 10
    stride: int = 3
    n_atoms: int = 64
    ksvd_iters: int = 3
    sparse_method: str = "lasso"
    lasso_alpha: float = 1e-3
    lasso_max_iter: int = 1000
    elastic_net_alpha: float = 1e-4
    elastic_net_l1_ratio: float = 0.5
    elastic_net_max_iter: int = 1000
    sparse_codes_nonnegative: bool = False
    blend: float = 1.0
    seed: int = 42


def _extract_windows_2d(
    image: np.ndarray,
    *,
    window: int,
    stride: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    values = np.asarray(image, dtype=np.float32)
    h, w = values.shape
    patches: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []
    for y in range(0, h - int(window) + 1, int(stride)):
        for x in range(0, w - int(window) + 1, int(stride)):
            patches.append(values[y : y + int(window), x : x + int(window)].reshape(-1))
            positions.append((y, x))
    if not patches:
        raise ValueError(f"map shape {values.shape} is smaller than window={window}")
    return np.asarray(patches, dtype=np.float32), positions


def _overlap_average_windows(
    patches: np.ndarray,
    positions: list[tuple[int, int]],
    *,
    out_shape: tuple[int, int],
    window: int,
) -> np.ndarray:
    acc = np.zeros(out_shape, dtype=np.float64)
    count = np.zeros(out_shape, dtype=np.float64)
    for patch, (y, x) in zip(np.asarray(patches), positions):
        block = np.asarray(patch, dtype=np.float64).reshape(int(window), int(window))
        acc[y : y + int(window), x : x + int(window)] += block
        count[y : y + int(window), x : x + int(window)] += 1.0
    return (acc / np.maximum(count, 1.0)).astype(np.float32, copy=False)


def spatial_ksvd_denoise_map(
    activation_map: np.ndarray,
    *,
    config: SpatialKsvdDenoiseConfig | None = None,
    seed_offset: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Denoise one LR map using overlapping spatial patches and K-SVD."""
    cfg = config or SpatialKsvdDenoiseConfig()
    values = np.maximum(
        np.nan_to_num(
            np.asarray(activation_map, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
    )
    scale = float(np.percentile(values, 99.0)) if values.size else 0.0
    if not np.isfinite(scale) or scale <= 0:
        return values.copy(), {
            "method": f"spatial_ksvd_{str(cfg.sparse_method).strip().lower()}",
            "config": asdict(cfg),
            "skipped": "non-positive map scale",
        }
    window = max(2, min(int(cfg.window), *values.shape))
    patches, positions = _extract_windows_2d(
        values / scale,
        window=window,
        stride=max(1, int(cfg.stride)),
    )
    n_atoms = min(int(cfg.n_atoms), int(patches.shape[0]))
    sparse_method = str(cfg.sparse_method).strip().lower()
    if sparse_method not in {"lasso", "elastic_net"}:
        raise ValueError(
            "spatial sparse_method must be 'lasso' or 'elastic_net', "
            f"got {sparse_method!r}"
        )
    ksvd_config = ClassicalKsvdConfig(
        input_dim=window * window,
        source_input_dim=window * window,
        embedding_block="spatial_activation_window",
        block_start=0,
        block_end=window * window,
        n_atoms=n_atoms,
        max_iter=int(cfg.ksvd_iters),
        ksvd_iter=1,
        sparse_method=sparse_method,
        lasso_alpha=float(cfg.lasso_alpha),
        lasso_algorithm="lasso_cd",
        elastic_net_alpha=float(cfg.elastic_net_alpha),
        elastic_net_l1_ratio=float(cfg.elastic_net_l1_ratio),
        elastic_net_max_iter=int(cfg.elastic_net_max_iter),
        sparse_codes_nonnegative=bool(cfg.sparse_codes_nonnegative),
        preprocess_mode="raw",
        prune_enabled=False,
        reinit_dead_atoms=True,
        seed=int(cfg.seed) + int(seed_offset),
        max_abs_code=1e4,
    )
    with _suppress_runtime_warnings():
        dictionary, _history = fit_approx_dictionary(
            patches,
            ksvd_config,
            verbose=False,
        )
        codes = encode_samples(
            patches,
            dictionary,
            ksvd_config,
            refine=False,
        ).astype(np.float32, copy=False)
        reconstructed = (codes @ dictionary).astype(np.float32, copy=False)
    denoised = _overlap_average_windows(
        reconstructed,
        positions,
        out_shape=values.shape,
        window=window,
    )
    denoised = np.maximum(denoised * scale, 0.0)
    blend = float(np.clip(cfg.blend, 0.0, 1.0))
    result = ((1.0 - blend) * values + blend * denoised).astype(
        np.float32,
        copy=False,
    )
    return result, {
        "method": f"spatial_ksvd_{sparse_method}",
        "config": asdict(cfg),
        "window_count": len(positions),
        "dictionary_shape": list(dictionary.shape),
        "input_p99": scale,
    }


def _window_starts(side: int, window: int, stride: int) -> list[int]:
    side = int(side)
    window = int(window)
    stride = max(1, int(stride))
    if side <= window:
        return [0]
    starts = list(range(0, side - window + 1, stride))
    final = side - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def _lr_bounds_for_hr_window(
    y0: int,
    x0: int,
    *,
    hr_side: int,
    lr_h: int,
    lr_w: int,
    hr_window: int,
    lr_window: int,
) -> tuple[int, int, int, int]:
    scale_y = float(hr_side) / float(lr_h)
    scale_x = float(hr_side) / float(lr_w)
    cy = (float(y0) + 0.5 * float(hr_window)) / scale_y
    cx = (float(x0) + 0.5 * float(hr_window)) / scale_x
    ly0 = int(round(cy - 0.5 * float(lr_window)))
    lx0 = int(round(cx - 0.5 * float(lr_window)))
    ly0 = max(0, min(int(lr_h) - int(lr_window), ly0))
    lx0 = max(0, min(int(lr_w) - int(lr_window), lx0))
    return ly0, lx0, ly0 + int(lr_window), lx0 + int(lr_window)


@contextlib.contextmanager
def _suppress_runtime_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            yield


def _dictionary_from_shared_codes(alpha_lr: np.ndarray, targets: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha_lr, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        dictionary = np.linalg.lstsq(alpha, target, rcond=None)[0]
    return np.nan_to_num(
        dictionary.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _encode_rgb_batched(
    rgb: np.ndarray,
    dictionary: np.ndarray,
    config: ClassicalKsvdConfig,
    *,
    batch_size: int,
) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32)
    chunks: list[np.ndarray] = []
    for start in range(0, values.shape[0], max(1, int(batch_size))):
        stop = min(values.shape[0], start + max(1, int(batch_size)))
        with _suppress_runtime_warnings():
            chunks.append(
                encode_samples(
                    values[start:stop],
                    dictionary,
                    config,
                    refine=False,
                ).astype(np.float32, copy=False)
            )
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _local_dai_window(
    rgb_hr_window: np.ndarray,
    rgb_lr_window: np.ndarray,
    pos_lr_window: np.ndarray,
    neg_lr_window: np.ndarray,
    *,
    config: LocalDaiConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    x_rgb_lr = np.asarray(rgb_lr_window, dtype=np.float32).reshape(-1, 3)
    y_pos_lr = np.maximum(
        np.asarray(pos_lr_window, dtype=np.float32).reshape(
            -1, pos_lr_window.shape[-1]
        ),
        0.0,
    )
    y_neg_lr = np.maximum(
        np.asarray(neg_lr_window, dtype=np.float32).reshape(
            -1, neg_lr_window.shape[-1]
        ),
        0.0,
    )
    n_stack_atoms = int(y_pos_lr.shape[1])
    n_lr_samples = int(x_rgb_lr.shape[0])
    n_atoms = int(
        min(config.rgb_atoms, n_lr_samples, max(2, n_lr_samples // 2))
    )
    sparse_method = str(config.rgb_sparse_method).strip().lower()
    if sparse_method not in {"omp", "elastic_net", "nnls"}:
        raise ValueError(
            f"rgb_sparse_method must be 'omp', 'elastic_net', or 'nnls', "
            f"got {sparse_method!r}"
        )
    omp_n_nonzero = max(1, int(min(config.rgb_omp_nonzero, n_atoms, n_lr_samples)))
    rgb_config = ClassicalKsvdConfig(
        input_dim=3,
        source_input_dim=3,
        embedding_block="local_rgb_pixel",
        block_start=0,
        block_end=3,
        n_atoms=n_atoms,
        max_iter=int(config.rgb_ksvd_iters),
        ksvd_iter=1,
        sparse_method=sparse_method,
        omp_n_nonzero=omp_n_nonzero,
        elastic_net_alpha=float(config.rgb_elastic_net_alpha),
        elastic_net_l1_ratio=float(config.rgb_elastic_net_l1_ratio),
        elastic_net_max_iter=int(config.rgb_elastic_net_max_iter),
        sparse_codes_nonnegative=bool(config.rgb_sparse_codes_nonnegative),
        preprocess_mode="raw",
        prune_enabled=False,
        reinit_dead_atoms=True,
        seed=int(seed),
        max_abs_code=1e4,
    )

    with _suppress_runtime_warnings():
        d_rgb, _history = fit_approx_dictionary(
            x_rgb_lr,
            rgb_config,
            verbose=False,
        )
        alpha_lr = encode_samples(
            x_rgb_lr,
            d_rgb,
            rgb_config,
            refine=False,
        ).astype(np.float32, copy=False)
        d_pos = _dictionary_from_shared_codes(alpha_lr, y_pos_lr)
        d_neg = _dictionary_from_shared_codes(alpha_lr, y_neg_lr)
        x_rgb_hr = np.asarray(rgb_hr_window, dtype=np.float32).reshape(-1, 3)
        alpha_hr = _encode_rgb_batched(
            x_rgb_hr,
            d_rgb,
            rgb_config,
            batch_size=config.encode_batch_size,
        )
        hr_h, hr_w = rgb_hr_window.shape[:2]
        pos_hr = np.maximum(alpha_hr @ d_pos, 0.0).reshape(
            hr_h,
            hr_w,
            n_stack_atoms,
        )
        neg_hr = np.maximum(alpha_hr @ d_neg, 0.0).reshape(
            hr_h,
            hr_w,
            n_stack_atoms,
        )

    return (
        np.nan_to_num(pos_hr, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        ),
        np.nan_to_num(neg_hr, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32, copy=False
        ),
    )


def run_local_dai_sr(
    rgb_crop: np.ndarray,
    positive_lr_stack: np.ndarray,
    *,
    negative_lr_stack: np.ndarray | None = None,
    config: LocalDaiConfig | None = None,
    show_progress: bool = True,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Refine an ``H×W×atoms`` LR stack to local RGB-guided HR maps."""
    cfg = config or LocalDaiConfig()
    pos_lr = np.maximum(
        np.nan_to_num(
            np.asarray(positive_lr_stack, dtype=np.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        0.0,
    )
    if pos_lr.ndim != 3:
        raise ValueError(f"positive_lr_stack must be H×W×atoms, got {pos_lr.shape}")
    neg_lr = (
        np.zeros_like(pos_lr)
        if negative_lr_stack is None
        else np.maximum(
            np.nan_to_num(
                np.asarray(negative_lr_stack, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            0.0,
        )
    )
    if neg_lr.shape != pos_lr.shape:
        raise ValueError(f"negative stack {neg_lr.shape} != positive stack {pos_lr.shape}")

    lr_h, lr_w, n_stack_atoms = pos_lr.shape
    lr_window = max(2, min(int(cfg.lr_window), int(lr_h), int(lr_w)))
    hr_side = int(cfg.hr_side)
    inferred_hr_window = max(
        8,
        int(round(lr_window * float(hr_side) / float(min(lr_h, lr_w)))),
    )
    hr_window = int(cfg.hr_window or inferred_hr_window)
    hr_window = min(hr_window, hr_side)
    hr_stride = max(1, min(int(cfg.hr_stride), hr_window))

    crop = Image.fromarray(np.asarray(rgb_crop, dtype=np.uint8)).resize(
        (hr_side, hr_side),
        Image.Resampling.LANCZOS,
    )
    rgb_hr = np.asarray(crop, dtype=np.float32) / 255.0
    rgb_lr = np.asarray(
        crop.resize((lr_w, lr_h), Image.Resampling.BICUBIC),
        dtype=np.float32,
    ) / 255.0

    y_starts = _window_starts(hr_side, hr_window, hr_stride)
    x_starts = _window_starts(hr_side, hr_window, hr_stride)
    jobs = [
        (y_index, x_index, y0, x0)
        for y_index, y0 in enumerate(y_starts)
        for x_index, x0 in enumerate(x_starts)
    ]
    iterator: Any = jobs
    if show_progress:
        from tqdm.auto import tqdm

        iterator = tqdm(
            jobs,
            total=len(jobs),
            desc=f"local Dai {label}".strip(),
            dynamic_ncols=True,
            leave=True,
        )

    pos_acc = np.zeros((hr_side, hr_side, n_stack_atoms), dtype=np.float64)
    neg_acc = np.zeros_like(pos_acc)
    weight = np.zeros((hr_side, hr_side), dtype=np.float64)
    if cfg.hann_blend:
        window_1d = np.hanning(hr_window).astype(np.float64)
        blend = np.maximum(np.outer(window_1d, window_1d), 1e-3)
    else:
        blend = np.ones((hr_window, hr_window), dtype=np.float64)
    for y_index, x_index, y0, x0 in iterator:
        ly0, lx0, ly1, lx1 = _lr_bounds_for_hr_window(
            y0,
            x0,
            hr_side=hr_side,
            lr_h=lr_h,
            lr_w=lr_w,
            hr_window=hr_window,
            lr_window=lr_window,
        )
        pos_prediction, neg_prediction = _local_dai_window(
            rgb_hr[y0 : y0 + hr_window, x0 : x0 + hr_window, :],
            rgb_lr[ly0:ly1, lx0:lx1, :],
            pos_lr[ly0:ly1, lx0:lx1, :],
            neg_lr[ly0:ly1, lx0:lx1, :],
            config=cfg,
            seed=int(cfg.seed) + y_index * 100 + x_index,
        )
        y1 = y0 + hr_window
        x1 = x0 + hr_window
        pos_acc[y0:y1, x0:x1, :] += (
            pos_prediction.astype(np.float64) * blend[:, :, None]
        )
        neg_acc[y0:y1, x0:x1, :] += (
            neg_prediction.astype(np.float64) * blend[:, :, None]
        )
        weight[y0:y1, x0:x1] += blend

    pos_hr = (pos_acc / np.maximum(weight[:, :, None], 1e-12)).astype(np.float32)
    neg_hr = (neg_acc / np.maximum(weight[:, :, None], 1e-12)).astype(np.float32)
    metadata = {
        "method": "local_dai_rgb_shared_sparse_weights_sliding_window_overlap_average",
        "config": asdict(cfg),
        "lr_shape": [int(lr_h), int(lr_w), int(n_stack_atoms)],
        "hr_shape": [int(value) for value in pos_hr.shape],
        "lr_window": int(lr_window),
        "hr_window": int(hr_window),
        "hr_stride": int(hr_stride),
        "window_count": int(len(jobs)),
        "overlap_average": True,
        "hann_blend": bool(cfg.hann_blend),
    }
    return pos_hr, neg_hr, metadata
