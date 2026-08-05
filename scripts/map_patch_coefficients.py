#!/usr/bin/env python3
"""Fit non-negative Elastic Net coefficient maps on patch Z-embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataclasses import replace

from visual_history_agent.models.classical_ksvd import (
    ClassicalKsvdConfig,
    encode_samples,
    load_classical_ksvd_config,
)
from visual_history_agent.paths import artifact_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--patch-embeddings", type=Path, required=True)
    parser.add_argument("--pooled-embedding", type=Path, required=True)
    parser.add_argument("--dictionary", type=Path)
    parser.add_argument("--embedding-mean", type=Path)
    parser.add_argument("--ksvd-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atom-ids")
    parser.add_argument("--local-dai", action="store_true")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    paths = artifact_paths(PROJECT_ROOT)
    dictionary = np.load(args.dictionary or paths["dictionary"]).astype(np.float32)
    mean = np.load(args.embedding_mean or paths["embedding_mean"]).astype(np.float32)
    cfg_path = args.ksvd_config or paths["ksvd_config"]
    cfg = load_classical_ksvd_config(cfg_path) if cfg_path.is_file() else ClassicalKsvdConfig(
        n_atoms=dictionary.shape[0],
        sparse_method="elastic_net",
        elastic_net_alpha=3e-5,
        elastic_net_l1_ratio=0.5,
        sparse_codes_nonnegative=True,
        preprocess_mode="raw",
    )
    cfg.sparse_method = "elastic_net"
    cfg.sparse_codes_nonnegative = True

    patches = np.load(args.patch_embeddings).astype(np.float32)
    pooled = np.load(args.pooled_embedding).astype(np.float32).reshape(1, -1)
    if patches.ndim != 2 or patches.shape[1] != dictionary.shape[1]:
        raise ValueError(f"Bad patch shape {patches.shape} vs dict {dictionary.shape}")
    n_patches = patches.shape[0]
    grid = int(round(n_patches ** 0.5))
    if grid * grid != n_patches:
        raise ValueError(f"Patch count {n_patches} is not a square grid")

    patches_c = patches - mean[None, :]
    pooled_c = pooled - mean[None, :]
    global_codes = encode_samples(pooled_c, dictionary, cfg, show_progress=False)[0]
    if args.atom_ids:
        atom_ids = [int(x) for x in args.atom_ids.split(",") if x.strip()]
    else:
        order = np.argsort(-np.abs(global_codes))
        atom_ids = [int(i) for i in order[: max(1, args.top_k)] if global_codes[i] > cfg.active_eps]
        if not atom_ids:
            atom_ids = [int(i) for i in order[: max(1, args.top_k)]]

    d_sub = dictionary[np.asarray(atom_ids, dtype=np.int64)]
    cfg_sub = replace(cfg, n_atoms=len(atom_ids))
    codes = encode_samples(patches_c, d_sub, cfg_sub, show_progress=True)
    stack = codes.reshape(grid, grid, len(atom_ids)).astype(np.float32)

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "coefficient_stack.npy", stack)
    np.save(out / "atom_ids.npy", np.asarray(atom_ids, dtype=np.int64))
    np.save(out / "global_codes.npy", global_codes.astype(np.float32))

    dai_path = None
    if args.local_dai and args.image is not None:
        from visual_history_agent.models.local_dai_sr import LocalDaiConfig, run_local_dai_sr
        from visual_history_agent.models.longclip_student_model import (
            load_rgb_image_capped,
            top_crop_square_pil,
        )

        rgb = load_rgb_image_capped(args.image.expanduser().resolve(), max_decode_side=2048)
        crop, _ = top_crop_square_pil(rgb, 224)
        positive = np.maximum(stack, 0.0)
        try:
            pos_hr, _neg_hr, _meta = run_local_dai_sr(
                np.asarray(crop),
                positive,
                config=LocalDaiConfig(hr_side=224, lr_window=10, lr_stride=3, hr_stride=3),
                show_progress=True,
                label=args.image.stem,
            )
            dai_path = out / "coefficient_stack_local_dai.npy"
            np.save(dai_path, np.asarray(pos_hr, dtype=np.float32))
        except Exception as exc:
            print(f"Local Dai skipped: {exc}", flush=True)

    meta = {
        "agent": "Visual History Agent",
        "patch_grid": [grid, grid],
        "atom_ids": atom_ids,
        "coefficient_stack": str(out / "coefficient_stack.npy"),
        "local_dai": str(dai_path) if dai_path else None,
        "elastic_net_alpha": float(cfg.elastic_net_alpha),
        "elastic_net_l1_ratio": float(cfg.elastic_net_l1_ratio),
        "mean_active_patches": float(np.mean(np.sum(stack > cfg.active_eps, axis=-1))),
    }
    (out / "coefficient_maps_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"DONE maps={out / 'coefficient_stack.npy'} atoms={atom_ids}", flush=True)


if __name__ == "__main__":
    main()
