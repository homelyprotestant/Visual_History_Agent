"""Resolve Visual History Agent project roots and shipped data artifacts."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def project_root() -> Path:
    env = os.environ.get("VISUAL_HISTORY_AGENT_ROOT") or os.environ.get("PAINTING_PIPELINE_ROOT")
    if env:
        root = Path(env).expanduser().resolve()
        if root.name == "data" and (root.parent / "visual_history_agent").is_dir():
            return root.parent
        return root
    return Path(__file__).resolve().parents[1]


def data_root(root: Path | None = None) -> Path:
    return (root or project_root()) / "data"


def visualhistory_root(root: Path | None = None) -> Path:
    """Parent of the ``VisualHistory`` package (shipped under ``vendor/``).

    Detached from any machine-local checkout. Override with ``VISUALHISTORY_ROOT``
    if needed; otherwise always ``<project>/vendor``.
    """
    env = (os.environ.get("VISUALHISTORY_ROOT") or "").strip()
    candidate = Path(env).expanduser().resolve() if env else (root or project_root()) / "vendor"
    if not (candidate / "VisualHistory").is_dir():
        raise FileNotFoundError(
            f"Vendored VisualHistory package not found under {candidate}. "
            "Expected vendor/VisualHistory/ inside the project zip "
            "(rebuild with scripts/pack_colab_zip.sh)."
        )
    return candidate


def ensure_visualhistory_on_path(root: Path | None = None) -> Path:
    """Insert vendored VisualHistory parent onto ``sys.path`` and load optional .env."""
    vh_root = visualhistory_root(root)
    if str(vh_root) not in sys.path:
        sys.path.insert(0, str(vh_root))
    try:
        from dotenv import load_dotenv

        load_dotenv(vh_root / ".env")
        load_dotenv((root or project_root()) / ".env")
    except Exception:
        pass
    return vh_root


def require_file(path: Path, *, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def artifact_paths(root: Path | None = None) -> dict[str, Path]:
    root = root or project_root()
    data = data_root(root)
    models = data / "models"
    embeddings = data / "embeddings"
    artifacts = data / "artifacts"
    ksvd = artifacts / "global_ksvd"
    atoms = artifacts / "atom_descriptions"
    return {
        "project_root": root,
        "data_root": data,
        "vendor": root / "vendor",
        "database": data / "Painting_Databaseb.xlsx",
        "embeddings": embeddings / "local_patch_pooled.npy",
        "embedding_manifest": embeddings / "local_patch_pooled_inference_manifest.json",
        "student_checkpoint": models / "VHM-B-16.pth",
        "z_embed_head": models / "student_head.pth",
        "ksvd_dir": ksvd,
        "dictionary": ksvd / "global_patch_pooled_dictionary_atoms.npy",
        "embedding_mean": ksvd / "global_patch_pooled_embedding_mean.npy",
        "ksvd_config": ksvd / "global_patch_pooled_ksvd_config.json",
        "sparse_codes": ksvd / "global_patch_pooled_sparse_codes.npy",
        "dictionary_meta": ksvd / "global_dictionary_meta.json",
        "atom_descriptions_dir": atoms,
        "atom_descriptions": atoms / "global_atom_painterly_descriptions.json",
        "qwen_cache": atoms / "qwen_rcs_chunk_cache.jsonl",
        "examples": data / "examples",
        "outputs": data / "outputs",
    }


def require_inference_weights(root: Path | None = None) -> dict[str, Path]:
    paths = artifact_paths(root)
    require_file(paths["student_checkpoint"], label="student ViT checkpoint")
    require_file(paths["z_embed_head"], label="Z-embed head checkpoint")
    return paths


def require_prep_inputs(root: Path | None = None) -> dict[str, Path]:
    paths = require_inference_weights(root)
    require_file(paths["embeddings"], label="corpus embedding cache")
    require_file(paths["database"], label="painting database")
    return paths


def require_downstream_artifacts(root: Path | None = None) -> dict[str, Path]:
    paths = require_inference_weights(root)
    require_file(paths["dictionary"], label="dictionary atoms")
    require_file(paths["embedding_mean"], label="embedding mean")
    require_file(paths["ksvd_config"], label="K-SVD config")
    require_file(paths["sparse_codes"], label="corpus sparse codes")
    require_file(paths["atom_descriptions"], label="atom painterly descriptions")
    require_file(paths["embeddings"], label="corpus embedding cache")
    require_file(paths["database"], label="painting database")
    visualhistory_root(root)  # ensure vendor present
    return paths
