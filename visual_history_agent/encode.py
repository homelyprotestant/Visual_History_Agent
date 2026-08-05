"""Encode query images with the Run4 student ViT + Z-embed head."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from visual_history_agent.paths import project_root, require_inference_weights
from visual_history_agent.stream_subprocess import run_streaming


def _python() -> str:
    return sys.executable


def encode_image(
    image_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    save_patch_embeddings: bool = False,
    device: str = "mps",
    infer_size: int = 1120,
    force: bool = False,
) -> dict[str, Any]:
    """Run student+Z-head inference. Returns paths to pooled (and optional patch) embeddings."""
    root = project_root()
    paths = require_inference_weights(root)
    image_path = Path(image_path).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    out = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else paths["outputs"] / "embeddings" / image_path.stem
    )
    out.mkdir(parents=True, exist_ok=True)
    pooled = out / "query_z_embed_pooled_2304.npy"
    manifest_path = out / "query_embedding_manifest.json"
    patch_path = out / "query_z_embed_patches_2304.npy"
    have_cached = (
        pooled.is_file()
        and manifest_path.is_file()
        and (patch_path.is_file() if save_patch_embeddings else True)
    )
    if have_cached and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {
            "image_path": str(image_path),
            "output_dir": str(out),
            "pooled_embedding": str(pooled),
            "patch_embeddings": str(patch_path) if patch_path.is_file() else None,
            "manifest": manifest,
            "cached": True,
        }
    script = root / "scripts" / "infer_single_image_z_embed.py"
    cmd = [
        _python(),
        str(script),
        str(image_path),
        "--output-dir",
        str(out),
        "--device",
        device,
        "--infer-size",
        str(infer_size),
    ]
    if save_patch_embeddings:
        cmd.append("--save-patch-embeddings")
    if not script.is_file():
        raise FileNotFoundError(
            f"Missing encode script (was it downloaded?): {script}"
        )
    print(f"[encode] infer {image_path.name} on {device}", flush=True)
    try:
        run_streaming(cmd, cwd=root)
    except RuntimeError as exc:
        raise RuntimeError(f"encode failed: {exc}") from exc
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "image_path": str(image_path),
        "output_dir": str(out),
        "pooled_embedding": str(pooled),
        "patch_embeddings": str(patch_path) if patch_path.is_file() else None,
        "manifest": manifest,
        "cached": False,
    }
