#!/usr/bin/env python3
"""Encode one external painting with the Run4 patch-pooled Z-embed pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual_history_agent.models.clip_device import configure_training_runtime, resolve_device
from visual_history_agent.models.longclip_student_model import (
    StudentHead,
    center_crop_square,
    encode_visual_patch_tokens,
    load_rgb_image_capped,
    load_visual_and_head_from_checkpoint,
    release_torch_memory,
)


def _load_z_embed_head(checkpoint: Path, device: torch.device) -> StudentHead:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    head_state = payload.get("head_state")
    if head_state is None:
        model_state = payload.get("model_state", {})
        head_state = {
            key.removeprefix("head."): value
            for key, value in model_state.items()
            if key.startswith("head.")
        }
    if not head_state:
        raise KeyError(f"No Z-embed head weights in {checkpoint}")
    teacher_dim = int(payload.get("teacher_dim", 2304))
    visual_dim = int(payload.get("visual_dim", 768))
    num_classes = int(
        payload.get("num_classes", head_state["classifier.weight"].shape[0])
    )
    head = StudentHead(visual_dim, teacher_dim, num_classes)
    head.load_state_dict(head_state, strict=True)
    return head.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--infer-size", type=int, default=1120)
    parser.add_argument("--max-decode-side", type=int, default=2048)
    parser.add_argument(
        "--save-patch-embeddings",
        action="store_true",
        help="Also save one 2304-D Z embedding per aligned ViT patch for spatial maps.",
    )
    parser.add_argument("--patch-batch-size", type=int, default=512)
    args = parser.parse_args()
    if args.patch_batch_size <= 0:
        parser.error("--patch-batch-size must be positive")

    from visual_history_agent.paths import artifact_paths, require_inference_weights
    vha = require_inference_weights(PROJECT_ROOT)
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else (vha["outputs"] / "embeddings" / image_path.stem)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    student_checkpoint = vha["student_checkpoint"]
    z_checkpoint = vha["z_embed_head"]

    device = resolve_device(args.device)
    configure_training_runtime(device)
    visual, _student_head, metadata = load_visual_and_head_from_checkpoint(
        student_checkpoint,
        device=device,
        strict=False,
    )
    z_head = _load_z_embed_head(z_checkpoint, device)
    use_dynamic = bool(metadata.get("use_dynamic_position_interpolation", True))
    use_rope = bool(metadata.get("use_experimental_2d_rope", False))

    rgb = load_rgb_image_capped(image_path, max_decode_side=args.max_decode_side)
    x_image = center_crop_square(rgb, args.infer_size)[None].to(device)
    with torch.inference_mode():
        patch_tokens, grid_h, grid_w = encode_visual_patch_tokens(
            visual,
            x_image,
            use_dynamic_position_interpolation=use_dynamic,
            use_experimental_2d_rope=use_rope,
            use_autocast=device.type != "mps",
        )
        normalized_patch_tokens = F.normalize(patch_tokens[0], dim=-1)
        pooled_768 = normalized_patch_tokens.mean(dim=0, keepdim=True)
        query_z, _ = z_head(pooled_768)
        patch_z_chunks: list[np.ndarray] = []
        if args.save_patch_embeddings:
            for start in range(0, normalized_patch_tokens.shape[0], args.patch_batch_size):
                patch_z, _ = z_head(
                    normalized_patch_tokens[start : start + args.patch_batch_size]
                )
                patch_z_chunks.append(
                    patch_z.detach().float().cpu().numpy().astype(np.float32, copy=False)
                )
    query_z = (
        query_z[0].detach().float().cpu().numpy().astype(np.float32, copy=False)
    )
    if query_z.shape != (2304,) or not np.isfinite(query_z).all():
        raise ValueError(f"Invalid query embedding shape/values: {query_z.shape}")

    embedding_path = output_dir / "query_z_embed_pooled_2304.npy"
    np.save(embedding_path, query_z)
    patch_embedding_path: Path | None = None
    patch_embedding_shape: list[int] | None = None
    if args.save_patch_embeddings:
        patch_embeddings = np.concatenate(patch_z_chunks, axis=0)
        expected_patches = int(grid_h) * int(grid_w)
        if (
            patch_embeddings.shape != (expected_patches, 2304)
            or not np.isfinite(patch_embeddings).all()
        ):
            raise ValueError(
                f"Invalid patch embedding shape/values: {patch_embeddings.shape}; "
                f"expected {(expected_patches, 2304)}"
            )
        patch_embedding_path = output_dir / "query_z_embed_patches_2304.npy"
        np.save(patch_embedding_path, patch_embeddings)
        patch_embedding_shape = list(patch_embeddings.shape)
    manifest = {
        "image_path": str(image_path),
        "embedding_path": str(embedding_path),
        "student_checkpoint": str(student_checkpoint),
        "z_embed_checkpoint": str(z_checkpoint),
        "preprocessing": {
            "infer_size": int(args.infer_size),
            "crop": "scale short side then center square crop",
            "patch_pooling": "L2-normalize patch tokens then mean",
        },
        "patch_grid": [int(grid_h), int(grid_w)],
        "patch_embedding_path": (
            str(patch_embedding_path) if patch_embedding_path is not None else None
        ),
        "patch_embedding_shape": patch_embedding_shape,
        "embedding_shape": list(query_z.shape),
        "embedding_norm": float(np.linalg.norm(query_z)),
    }
    (output_dir / "query_embedding_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        f"DONE embedding={embedding_path} norm={manifest['embedding_norm']:.6f}"
        + (
            f" patches={patch_embedding_path} shape={patch_embedding_shape}"
            if patch_embedding_path is not None
            else ""
        )
    )
    release_torch_memory(device, aggressive=True)


if __name__ == "__main__":
    main()
