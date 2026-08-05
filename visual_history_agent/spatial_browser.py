"""Interactive local-Dai spatial atom browser.

Pipeline per image:
1. Global nonnegative Elastic Net on pooled Z → active atoms + importance
2. Centered patch-Z → browsed global atoms via **fixed-dictionary NMF** codes
3. Local Dai SR (RGB K-SVD + **OMP**) upsamples those maps to 224×224
4. ipywidgets browser (pair-scaled magma maps + painterly description)
"""

from __future__ import annotations

import html
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from visual_history_agent.encode import encode_image
from visual_history_agent.models.classical_ksvd import (
    ClassicalKsvdConfig,
    encode_samples,
    load_classical_ksvd_config,
    nmf_codes,
)
from visual_history_agent.models.local_dai_sr import LocalDaiConfig, run_local_dai_sr
from visual_history_agent.models.longclip_student_model import (
    load_rgb_image_capped,
    top_crop_square_pil,
)
from visual_history_agent.paths import project_root, require_downstream_artifacts

# LR maps: batched NMF coding on fixed global dictionary atoms.
LR_MAP_SOURCE = "fixed-dictionary NMF centered patch-Z codes on browsed global atoms"
LR_NMF_ITERS = 200
# Local Dai RGB coding (must match LocalDaiConfig below; used for cache checks).
LOCAL_DAI_RGB_SPARSE_METHOD = "omp"
LOCAL_DAI_RGB_SPARSE_NONNEGATIVE = False
LOCAL_DAI_RGB_OMP_NONZERO = 3


@dataclass
class SpatialBrowserBundle:
    """Cached local-Dai maps and metadata for one or two images."""

    labels: tuple[str, ...]
    images: dict[str, str]
    inference_crops: dict[str, np.ndarray]
    browser_atom_ids: list[int]
    normalized_codes: dict[str, np.ndarray]
    local_dai_maps: dict[str, np.ndarray]
    description_by_id: dict[int, dict[str, Any]]
    atom_category: dict[int, str]
    output_dir: Path
    local_dai_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    spatial_response_maps: dict[str, np.ndarray] = field(default_factory=dict)


def _load_atom_descriptions(
    path: Path,
    atom_ids: Sequence[int],
) -> dict[int, dict[str, Any]]:
    wanted = {int(a) for a in atom_ids}
    found: dict[int, dict[str, Any]] = {}
    jsonl = path.with_suffix(".jsonl")
    source = jsonl if jsonl.is_file() else path
    if source.suffix == ".jsonl":
        with source.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                atom_id = int(record["atom_id"])
                if atom_id in wanted:
                    found[atom_id] = record
                    if len(found) == len(wanted):
                        break
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
        for record in payload:
            atom_id = int(record["atom_id"])
            if atom_id in wanted:
                found[atom_id] = record
    for atom_id in wanted:
        found.setdefault(
            atom_id,
            {
                "atom_id": atom_id,
                "label": f"Atom {atom_id}",
                "painterly_description": "No corpus description on disk for this atom.",
            },
        )
    return found


def _normalized_positive_code(code: np.ndarray) -> np.ndarray:
    clean = np.where(np.isfinite(code) & (code > 0), code, 0.0).astype(np.float64)
    total = float(clean.sum())
    if total <= 0:
        raise ValueError("Sparse code has no positive coefficient mass")
    return clean / total


def _nmf_spatial_coefficient_maps(
    patches: np.ndarray,
    browser_dictionary: np.ndarray,
    embedding_mean: np.ndarray,
    *,
    grid_h: int,
    grid_w: int,
    show_progress: bool = True,
    n_iter: int = LR_NMF_ITERS,
) -> np.ndarray:
    """Return (n_atoms, grid_h, grid_w) fixed-dictionary NMF codes for all patches."""
    centered = np.asarray(patches, dtype=np.float32) - embedding_mean[None, :]
    codes = nmf_codes(
        centered,
        browser_dictionary,
        n_iter=int(n_iter),
        show_progress=show_progress,
    )
    n_atoms = int(browser_dictionary.shape[0])
    if codes.shape != (grid_h * grid_w, n_atoms):
        raise ValueError(
            f"NMF codes {codes.shape} incompatible with grid "
            f"{(grid_h, grid_w)} and {n_atoms} atoms"
        )
    return codes.T.reshape(n_atoms, grid_h, grid_w)


def prepare_spatial_browser(
    image_a: str | Path,
    image_b: str | Path | None = None,
    *,
    output_dir: str | Path | None = None,
    atom_ids: Sequence[int] | None = None,
    top_k: int | None = None,
    force_recompute: bool = False,
    device: str = "mps",
    show_progress: bool = True,
) -> SpatialBrowserBundle:
    """Prepare Two_Image-identical local Dai maps (cached) for the browser."""
    root = project_root()
    paths = require_downstream_artifacts(root)
    image_paths = {"A": Path(image_a).expanduser().resolve()}
    if image_b is not None:
        image_paths["B"] = Path(image_b).expanduser().resolve()
    for label, path in image_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Image {label}: {path}")

    labels = tuple(image_paths.keys())
    stem = "__vs__".join(path.stem for path in image_paths.values())
    out = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else paths["outputs"] / "spatial_browser" / stem
    )
    out.mkdir(parents=True, exist_ok=True)
    # Same relative cache name as Two_Image OUTPUT_ROOT / spatial_local_dai_hr224
    cache_dir = out / "spatial_local_dai_hr224"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dictionary = np.load(paths["dictionary"]).astype(np.float32)
    embedding_mean = np.load(paths["embedding_mean"]).astype(np.float32)
    cfg_path = paths["ksvd_config"]
    cfg = (
        load_classical_ksvd_config(cfg_path)
        if cfg_path.is_file()
        else ClassicalKsvdConfig(
            n_atoms=dictionary.shape[0],
            sparse_method="elastic_net",
            elastic_net_alpha=3e-5,
            elastic_net_l1_ratio=0.5,
            sparse_codes_nonnegative=True,
            preprocess_mode="raw",
        )
    )
    cfg.sparse_method = "elastic_net"
    cfg.sparse_codes_nonnegative = True

    global_codes: dict[str, np.ndarray] = {}
    normalized_codes: dict[str, np.ndarray] = {}
    inference_crops: dict[str, np.ndarray] = {}
    patch_grids: dict[str, tuple[int, int]] = {}
    patch_arrays: dict[str, np.ndarray] = {}

    for label, image_path in image_paths.items():
        emb_dir = out / "embeddings" / label
        encoded = encode_image(
            image_path,
            output_dir=emb_dir,
            save_patch_embeddings=True,
            device=device,
            force=force_recompute,
        )
        if not encoded["patch_embeddings"]:
            raise RuntimeError(f"Patch embeddings missing for image {label}")
        if encoded.get("cached"):
            print(f"Reusing cached patch embeddings for image {label}: {emb_dir}", flush=True)
        patches = np.load(encoded["patch_embeddings"]).astype(np.float32)
        pooled = np.load(encoded["pooled_embedding"]).astype(np.float32).reshape(1, -1)
        manifest = encoded["manifest"]
        grid_h, grid_w = (int(v) for v in manifest["patch_grid"])
        if patches.shape[0] != grid_h * grid_w:
            raise ValueError(
                f"Image {label}: {patches.shape[0]} patches vs grid {(grid_h, grid_w)}"
            )
        patch_grids[label] = (grid_h, grid_w)
        patch_arrays[label] = patches
        code_path = out / f"global_codes_{label}.npy"
        if code_path.is_file() and not force_recompute:
            code = np.load(code_path).astype(np.float32)
        else:
            pooled_c = pooled - embedding_mean[None, :]
            code = encode_samples(pooled_c, dictionary, cfg, show_progress=False)[0]
            code = code.astype(np.float32)
            np.save(code_path, code)
        global_codes[label] = code
        normalized_codes[label] = _normalized_positive_code(code)
        infer_size = int(manifest["preprocessing"]["infer_size"])
        rgb = load_rgb_image_capped(image_path, max_decode_side=2048)
        crop, _ = top_crop_square_pil(rgb, infer_size)
        inference_crops[label] = np.asarray(crop)

    # Atom set: Two_Image uses union of positive-mass atoms, ordered by combined importance.
    if atom_ids is not None:
        browser_atom_ids = [int(a) for a in atom_ids]
    elif len(labels) == 1:
        code_a = normalized_codes["A"]
        active = [int(i) for i in np.flatnonzero(code_a > 0)]
        browser_atom_ids = sorted(active, key=lambda i: float(code_a[i]), reverse=True)
    else:
        active_a = set(int(i) for i in np.flatnonzero(normalized_codes["A"] > 0))
        active_b = set(int(i) for i in np.flatnonzero(normalized_codes["B"] > 0))
        union_ids = active_a | active_b
        code_a = normalized_codes["A"]
        code_b = normalized_codes["B"]
        browser_atom_ids = sorted(
            union_ids,
            key=lambda atom_id: float(code_a[atom_id] + code_b[atom_id]),
            reverse=True,
        )
    if top_k is not None and top_k > 0:
        browser_atom_ids = browser_atom_ids[: int(top_k)]
    if not browser_atom_ids:
        raise RuntimeError("No active atoms to browse")
    print(f"Browser atoms: {len(browser_atom_ids)}", flush=True)

    if len(labels) == 1:
        atom_category = {atom_id: "active" for atom_id in browser_atom_ids}
        shared: set[int] = set()
        unique_a: set[int] = set(browser_atom_ids)
        unique_b: set[int] = set()
    else:
        active_a = set(int(i) for i in np.flatnonzero(normalized_codes["A"] > 0))
        active_b = set(int(i) for i in np.flatnonzero(normalized_codes["B"] > 0))
        shared = active_a & active_b
        unique_a = active_a - active_b
        unique_b = active_b - active_a
        atom_category = {
            atom_id: (
                "common to both"
                if atom_id in shared
                else "A-Specific"
                if atom_id in unique_a
                else "B-Specific"
            )
            for atom_id in browser_atom_ids
        }

    # Browser dictionary rows as used for global coding (NNLS normalizes rows).
    browser_dictionary = np.asarray(
        dictionary[np.asarray(browser_atom_ids, dtype=np.int64)],
        dtype=np.float32,
    )

    spatial_response_maps: dict[str, np.ndarray] = {}
    for label in labels:
        grid_h, grid_w = patch_grids[label]
        lr_cache = out / f"nmf_coefficient_maps_{label}.npy"
        loaded_lr = None
        if lr_cache.is_file() and not force_recompute:
            loaded_lr = np.load(lr_cache).astype(np.float32)
            if loaded_lr.shape != (len(browser_atom_ids), grid_h, grid_w):
                loaded_lr = None
        if loaded_lr is not None:
            spatial_response_maps[label] = loaded_lr
            print(f"Loaded cached NMF coefficient maps for image {label}", flush=True)
        else:
            print(
                f"Computing fixed-dictionary NMF patch codes for image {label} "
                f"({grid_h * grid_w} patches × {len(browser_atom_ids)} atoms, "
                f"{LR_NMF_ITERS} iters)...",
                flush=True,
            )
            spatial_response_maps[label] = _nmf_spatial_coefficient_maps(
                patch_arrays[label],
                browser_dictionary,
                embedding_mean,
                grid_h=grid_h,
                grid_w=grid_w,
                show_progress=show_progress,
                n_iter=LR_NMF_ITERS,
            )
            np.save(lr_cache, spatial_response_maps[label])

    local_dai_maps: dict[str, np.ndarray] = {}
    local_dai_metadata: dict[str, dict[str, Any]] = {}

    for index, label in enumerate(labels):
        cache_path = cache_dir / f"image_{label.lower()}_local_dai_positive_maps.npz"
        metadata_path = cache_dir / f"image_{label.lower()}_local_dai_metadata.json"
        cached = None
        if cache_path.is_file() and not force_recompute:
            candidate = np.load(cache_path)
            cached_ids = np.asarray(candidate["atom_ids"], dtype=np.int64)
            meta = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.is_file()
                else {}
            )
            cfg_meta = meta.get("config") if isinstance(meta.get("config"), dict) else {}
            method_ok = (
                str(cfg_meta.get("rgb_sparse_method", "")).lower()
                == LOCAL_DAI_RGB_SPARSE_METHOD
                and bool(cfg_meta.get("rgb_sparse_codes_nonnegative", False))
                == LOCAL_DAI_RGB_SPARSE_NONNEGATIVE
                and int(cfg_meta.get("rgb_omp_nonzero", -1)) == LOCAL_DAI_RGB_OMP_NONZERO
                and str(meta.get("lr_map_source", "")) == LR_MAP_SOURCE
            )
            if (
                np.array_equal(cached_ids, np.asarray(browser_atom_ids, dtype=np.int64))
                and method_ok
            ):
                cached = np.asarray(candidate["pos_hr"], dtype=np.float32)
                local_dai_metadata[label] = meta
                print(
                    f"Loaded cached local Dai SR maps for image {label}: {cache_path}",
                    flush=True,
                )
            elif cache_path.is_file():
                print(
                    f"Ignoring cache for image {label} "
                    f"(need LR={LR_MAP_SOURCE!r}, "
                    f"RGB={LOCAL_DAI_RGB_SPARSE_METHOD}, "
                    f"omp_nonzero={LOCAL_DAI_RGB_OMP_NONZERO}).",
                    flush=True,
                )

        if cached is None:
            print(
                f"Running local Dai SR for image {label}; this is cached after completion.",
                flush=True,
            )
            # NMF LR codes are already nonnegative; OMP RGB coding for Dai upsample.
            positive_lr_stack = np.maximum(
                spatial_response_maps[label].transpose(1, 2, 0),
                0.0,
            ).astype(np.float32, copy=False)
            dai_config = LocalDaiConfig(
                hr_side=224,
                lr_window=10,
                lr_stride=3,
                hr_stride=3,
                rgb_atoms=32,
                rgb_ksvd_iters=8,
                rgb_sparse_method=LOCAL_DAI_RGB_SPARSE_METHOD,
                rgb_sparse_codes_nonnegative=LOCAL_DAI_RGB_SPARSE_NONNEGATIVE,
                rgb_omp_nonzero=LOCAL_DAI_RGB_OMP_NONZERO,
                encode_batch_size=32768,
                seed=123 + (0 if label == "A" else 1000),
            )
            pos_hr, neg_hr, metadata = run_local_dai_sr(
                inference_crops[label],
                positive_lr_stack,
                config=dai_config,
                show_progress=show_progress,
                label=f"image {label}",
            )
            metadata.update(
                {
                    "image_label": label,
                    "image_path": str(image_paths[label]),
                    "lr_map_source": LR_MAP_SOURCE,
                    "atom_ids": browser_atom_ids,
                }
            )
            np.savez_compressed(
                cache_path,
                pos_hr=pos_hr,
                neg_hr=neg_hr,
                atom_ids=np.asarray(browser_atom_ids, dtype=np.int64),
            )
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            cached = pos_hr
            local_dai_metadata[label] = metadata

        if cached.shape != (224, 224, len(browser_atom_ids)):
            raise ValueError(f"Unexpected local Dai map shape for image {label}: {cached.shape}")
        local_dai_maps[label] = cached

    description_by_id = _load_atom_descriptions(paths["atom_descriptions"], browser_atom_ids)
    bundle_meta = {
        "agent": "Visual History Agent",
        "labels": list(labels),
        "images": {k: str(v) for k, v in image_paths.items()},
        "browser_atom_ids": browser_atom_ids,
        "n_atoms": len(browser_atom_ids),
        "shared_atoms": sorted(shared),
        "a_specific_atoms": sorted(unique_a),
        "b_specific_atoms": sorted(unique_b),
        "output_dir": str(out),
        "local_dai_cache": str(cache_dir),
        "lr_map_source": LR_MAP_SOURCE,
        "lr_nmf_iters": LR_NMF_ITERS,
        "hr_side": 224,
        "rgb_sparse_method": LOCAL_DAI_RGB_SPARSE_METHOD,
        "rgb_sparse_codes_nonnegative": LOCAL_DAI_RGB_SPARSE_NONNEGATIVE,
        "rgb_omp_nonzero": LOCAL_DAI_RGB_OMP_NONZERO,
    }
    (out / "spatial_browser_meta.json").write_text(
        json.dumps(bundle_meta, indent=2),
        encoding="utf-8",
    )

    return SpatialBrowserBundle(
        labels=labels,
        images={k: str(v) for k, v in image_paths.items()},
        inference_crops=inference_crops,
        browser_atom_ids=browser_atom_ids,
        normalized_codes=normalized_codes,
        local_dai_maps=local_dai_maps,
        description_by_id=description_by_id,
        atom_category=atom_category,
        output_dir=out,
        local_dai_metadata=local_dai_metadata,
        spatial_response_maps=spatial_response_maps,
    )


def display_spatial_atom_browser(bundle: SpatialBrowserBundle):
    """Launch the Two_Image interactive local-Dai atom browser (ipywidgets)."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display

    labels = bundle.labels
    browser_atom_ids = list(bundle.browser_atom_ids)
    atom_position = {atom_id: index for index, atom_id in enumerate(browser_atom_ids)}
    category_colors = {
        "active": "#6a3d9a",
        "common to both": "#6a3d9a",
        "A-Specific": "#1f78b4",
        "B-Specific": "#e31a1c",
    }

    if len(labels) == 1:
        filter_options = ["All"]
    else:
        filter_options = ["All", "common to both", "A-Specific", "B-Specific"]

    category_filter = widgets.ToggleButtons(
        options=filter_options,
        value="All",
        description="Atoms",
    )
    atom_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=max(0, len(browser_atom_ids) - 1),
        step=1,
        description="Atom rank",
        continuous_update=False,
        layout=widgets.Layout(width="700px"),
    )
    previous_button = widgets.Button(description="◀", layout=widgets.Layout(width="45px"))
    next_button = widgets.Button(description="▶", layout=widgets.Layout(width="45px"))
    spatial_status = widgets.HTML()
    current_browser_ids = browser_atom_ids.copy()

    thumbnail_size = 224
    thumbnail_layout = widgets.Layout(
        width=f"{thumbnail_size}px",
        height=f"{thumbnail_size}px",
        object_fit="contain",
    )

    def _pil_png_bytes(image: Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def _rgb_thumbnail_bytes(label: str) -> bytes:
        image = Image.fromarray(bundle.inference_crops[label]).resize(
            (thumbnail_size, thumbnail_size),
            Image.Resampling.LANCZOS,
        )
        return _pil_png_bytes(image)

    rgb_titles = {
        label: widgets.HTML(
            (
                f"<b>{html.escape(Path(bundle.images[label]).stem)}</b>"
                if len(labels) == 1
                else f"<b>Image {label} — RGB</b>"
            ),
            layout=widgets.Layout(width="224px"),
        )
        for label in labels
    }
    rgb_widgets = {
        label: widgets.Image(
            value=_rgb_thumbnail_bytes(label),
            format="png",
            width=thumbnail_size,
            height=thumbnail_size,
            layout=thumbnail_layout,
        )
        for label in labels
    }
    map_widgets = {
        label: widgets.Image(
            format="png",
            width=thumbnail_size,
            height=thumbnail_size,
            layout=thumbnail_layout,
        )
        for label in labels
    }
    rgb_row = widgets.HBox(
        [widgets.VBox([rgb_titles[label], rgb_widgets[label]]) for label in labels],
        layout=widgets.Layout(gap="16px"),
    )
    alpha_row = widgets.HBox(
        [map_widgets[label] for label in labels],
        layout=widgets.Layout(gap="16px"),
    )
    image_rows = widgets.VBox(
        [rgb_row, alpha_row],
        layout=widgets.Layout(gap="14px", flex="0 0 auto"),
    )
    spatial_description = widgets.HTML(
        layout=widgets.Layout(
            width="auto",
            min_width="460px",
            height="530px",
            flex="1 1 auto",
            overflow_y="auto",
            padding="12px",
        )
    )
    spatial_content = widgets.HBox(
        [image_rows, spatial_description],
        layout=widgets.Layout(width="100%", gap="20px", align_items="stretch"),
    )

    def _filtered_atom_ids() -> list[int]:
        selected = category_filter.value
        if selected == "All":
            return browser_atom_ids.copy()
        return [
            atom_id
            for atom_id in browser_atom_ids
            if bundle.atom_category[atom_id] == selected
        ]

    def _render_spatial_atom(_change=None) -> None:
        nonlocal current_browser_ids
        if not current_browser_ids:
            spatial_status.value = "<b>No atoms in this category.</b>"
            spatial_description.value = ""
            for label in labels:
                map_widgets[label].value = b""
            return

        rank = int(np.clip(atom_slider.value, 0, len(current_browser_ids) - 1))
        atom_id = int(current_browser_ids[rank])
        category = bundle.atom_category[atom_id]
        color = category_colors.get(category, "#333333")
        atom_record = bundle.description_by_id[atom_id]
        atom_label = str(atom_record.get("label", ""))
        atom_description = str(atom_record.get("painterly_description", ""))

        importance_bits = []
        for label in labels:
            pct = 100.0 * float(bundle.normalized_codes[label][atom_id])
            importance_bits.append(f"{label} importance: <b>{pct:.3f}%</b>")
        spatial_status.value = (
            f"<div style='border-left:6px solid {color}; padding:8px 12px; margin:4px 0;'>"
            f"<b>Atom {atom_id}: {html.escape(atom_label)}</b> — "
            f"<b style='color:{color}'>{html.escape(category)}</b> "
            f"({rank + 1}/{len(current_browser_ids)})<br>"
            f"{' &nbsp;|&nbsp; '.join(importance_bits)}"
            "</div>"
        )
        spatial_description.value = (
            f"<div style='line-height:1.5;'>"
            f"<h3 style='margin:0 0 6px 0;'>Atom {atom_id}</h3>"
            f"<h4 style='margin:0 0 10px 0; color:{color};'>"
            f"{html.escape(atom_label)}</h4>"
            f"<div style='font-size:13px;'>{html.escape(atom_description)}</div>"
            "</div>"
        )

        map_index = atom_position[atom_id]
        positive_maps = {
            label: np.maximum(bundle.local_dai_maps[label][:, :, map_index], 0.0)
            for label in labels
        }
        importance_by_label = {
            label: float(bundle.normalized_codes[label][atom_id]) for label in labels
        }
        highest_importance = max(importance_by_label.values()) if importance_by_label else 0.0
        color_map = plt.get_cmap("magma")
        for label in labels:
            spatial_p99 = float(np.percentile(positive_maps[label], 99.0))
            if not np.isfinite(spatial_p99) or spatial_p99 <= 0:
                spatial_p99 = max(float(positive_maps[label].max()), 1e-12)
            relative_importance = (
                importance_by_label[label] / highest_importance
                if highest_importance > 0
                else 0.0
            )
            normalized_map = (
                np.clip(positive_maps[label] / spatial_p99, 0.0, 1.0) * relative_importance
            )
            map_rgb = np.asarray(color_map(normalized_map)[..., :3] * 255.0, dtype=np.uint8)
            map_rgb[normalized_map <= 0] = 0
            map_image = Image.fromarray(map_rgb).resize(
                (thumbnail_size, thumbnail_size),
                Image.Resampling.NEAREST,
            )
            map_widgets[label].value = _pil_png_bytes(map_image)

    def _change_category(_change=None) -> None:
        nonlocal current_browser_ids
        current_browser_ids = _filtered_atom_ids()
        atom_slider.unobserve(_render_spatial_atom, names="value")
        try:
            atom_slider.max = max(0, len(current_browser_ids) - 1)
            atom_slider.value = 0
        finally:
            atom_slider.observe(_render_spatial_atom, names="value")
        _render_spatial_atom()

    def _step_atom(delta: int) -> None:
        if current_browser_ids:
            atom_slider.value = int(
                np.clip(atom_slider.value + delta, 0, len(current_browser_ids) - 1)
            )

    category_filter.observe(_change_category, names="value")
    atom_slider.observe(_render_spatial_atom, names="value")
    previous_button.on_click(lambda _button: _step_atom(-1))
    next_button.on_click(lambda _button: _step_atom(1))

    _render_spatial_atom()
    display(
        widgets.VBox(
            [
                category_filter,
                widgets.HBox([previous_button, next_button, atom_slider]),
                spatial_status,
                spatial_content,
            ],
            layout=widgets.Layout(width="100%"),
        )
    )
    # Do not return the bundle — Jupyter would print its repr under the widget.
