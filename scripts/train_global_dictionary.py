#!/usr/bin/env python3
"""Train the global mean-centered approximate K-SVD dictionary from the embedding cache."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual_history_agent.models.classical_ksvd import (
    ClassicalKsvdConfig,
    encode_samples,
    fit_approx_dictionary,
    history_to_dataframe,
    save_classical_ksvd_config,
)
from visual_history_agent.paths import require_prep_inputs


def train_global_dictionary(
    *,
    project_root: Path | None = None,
    output_dir: Path | None = None,
    n_atoms: int = 1000,
    elastic_net_alpha: float = 3e-5,
    l1_ratio: float = 0.5,
    max_iter: int = 10,
    subsample_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit the global dictionary; returns artifact paths and summary metrics."""
    root = (project_root or PROJECT_ROOT).expanduser().resolve()
    paths = require_prep_inputs(root)
    out = (output_dir or paths["ksvd_dir"]).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    stages = tqdm(total=5, desc="Dictionary learning", unit="stage")
    try:
        stages.set_postfix_str("load embeddings")
        embeddings = np.load(paths["embeddings"], mmap_mode="r")
        x = np.asarray(embeddings, dtype=np.float32)
        finite = np.isfinite(x).all(axis=1)
        x = x[finite]
        stages.update(1)

        stages.set_postfix_str("mean-center")
        mean = x.mean(axis=0).astype(np.float32)
        x_centered = x - mean[None, :]
        stages.update(1)

        cfg = ClassicalKsvdConfig(
            input_dim=2304,
            source_input_dim=2304,
            embedding_block="full",
            block_start=0,
            block_end=2304,
            n_atoms=int(n_atoms),
            max_iter=int(max_iter),
            ksvd_iter=1,
            sparse_method="elastic_net",
            elastic_net_alpha=float(elastic_net_alpha),
            elastic_net_l1_ratio=float(l1_ratio),
            elastic_net_max_iter=1000,
            sparse_codes_nonnegative=True,
            preprocess_mode="raw",
            subsample_fraction=float(subsample_fraction),
            active_eps=1e-4,
            reinit_dead_atoms=True,
            min_dict_change=1e-5,
            patience=6,
            seed=int(seed),
            prune_enabled=True,
            min_atom_support=20,
            max_top5_concentration=0.4,
            prune_every_iters=3,
            residual_pool_size=200,
            max_prune_per_cycle=50,
            max_abs_code=1e4,
        )

        stages.set_postfix_str(
            f"K-SVD n={x_centered.shape[0]:,} atoms={cfg.n_atoms}"
        )
        dictionary, history = fit_approx_dictionary(
            x_centered, cfg, verbose=True, checkpoint_dir=out
        )
        stages.update(1)

        stages.set_postfix_str("encode sparse codes")
        codes = encode_samples(x_centered, dictionary, cfg, show_progress=True)
        stages.update(1)

        stages.set_postfix_str("save artifacts")
        np.save(out / "global_patch_pooled_dictionary_atoms.npy", dictionary.astype(np.float32))
        np.save(out / "global_patch_pooled_embedding_mean.npy", mean)
        np.save(out / "global_patch_pooled_sparse_codes.npy", codes.astype(np.float32))
        save_classical_ksvd_config(cfg, out / "global_patch_pooled_ksvd_config.json")
        history_to_dataframe(history).to_csv(out / "ksvd_training_history.csv", index=False)
        meta = {
            "agent": "Visual History Agent",
            "n_atoms": int(n_atoms),
            "n_training_samples": int(x_centered.shape[0]),
            "elastic_net_alpha": float(elastic_net_alpha),
            "elastic_net_l1_ratio": float(l1_ratio),
            "mean_active_atoms": float(np.mean(np.sum(np.abs(codes) > cfg.active_eps, axis=1))),
            "dictionary": str(out / "global_patch_pooled_dictionary_atoms.npy"),
            "embedding_mean": str(out / "global_patch_pooled_embedding_mean.npy"),
            "sparse_codes": str(out / "global_patch_pooled_sparse_codes.npy"),
            "config": asdict(cfg),
        }
        (out / "global_dictionary_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        stages.update(1)
    finally:
        stages.close()

    print(f"DONE dictionary -> {out}", flush=True)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-atoms", type=int, default=1000)
    parser.add_argument("--elastic-net-alpha", type=float, default=3e-5)
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=10)
    parser.add_argument("--subsample-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    train_global_dictionary(
        project_root=PROJECT_ROOT,
        output_dir=args.output_dir,
        n_atoms=int(args.n_atoms),
        elastic_net_alpha=float(args.elastic_net_alpha),
        l1_ratio=float(args.l1_ratio),
        max_iter=int(args.max_iter),
        subsample_fraction=float(args.subsample_fraction),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
