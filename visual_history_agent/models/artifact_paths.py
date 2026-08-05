"""Resolve purged study artifacts from local or backup storage.

Large generated image and ``.npy`` artifacts may be removed from the local
study tree after they are backed up.  These helpers keep notebooks and scripts
pointing at the local codebase while falling back to the backup drive for
read-only assets that no longer exist locally.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

LOCAL_STABLE_ROOT = Path("/Users/marc/Desktop/Painting_Interpretability_Study/Stable")
DEFAULT_BACKUP_STABLE_ROOT = Path("/Volumes/Untitled/Painting_Interpretability_Study/Stable")


def _root_from_env() -> Path | None:
    raw = os.environ.get("PAINTING_STUDY_BACKUP_ROOT", "").strip()
    if not raw:
        return None
    root = Path(raw).expanduser()
    if root.name != "Stable" and (root / "Stable").is_dir():
        root = root / "Stable"
    return root


def candidate_stable_roots(project_root: Path | str | None = None) -> list[Path]:
    """Return local/backup Stable roots in preferred lookup order."""
    roots: list[Path] = []
    for raw in (
        project_root,
        Path.cwd(),
        Path.cwd().parent,
        LOCAL_STABLE_ROOT,
        _root_from_env(),
        DEFAULT_BACKUP_STABLE_ROOT,
    ):
        if raw is None:
            continue
        root = Path(raw).expanduser()
        try:
            root = root.resolve()
        except FileNotFoundError:
            root = root.absolute()
        if root not in roots:
            roots.append(root)
    return roots


def resolve_project_root(
    default: Path | str | None = None,
) -> Path:
    """Find the active Stable root, preferring the local working tree."""
    for root in candidate_stable_roots(default):
        if (root / "painting_data").is_dir() and (
            (root / "painting_glam").is_dir()
            or (root / "visual_history_agent").is_dir()
        ):
            return root
    fallback = Path(default).expanduser() if default is not None else LOCAL_STABLE_ROOT
    return fallback.resolve()


def _relative_to_known_root(path: Path, project_root: Path | str | None = None) -> Path | None:
    try:
        resolved = path.expanduser().resolve(strict=False)
    except RuntimeError:
        resolved = path.expanduser().absolute()
    for root in candidate_stable_roots(project_root):
        try:
            return resolved.relative_to(root)
        except ValueError:
            continue
    return None


def fallback_candidates(path: Path | str, project_root: Path | str | None = None) -> list[Path]:
    """Return equivalent local/backup paths for a study artifact path."""
    original = Path(path).expanduser()
    candidates: list[Path] = []
    if original.is_absolute():
        rel = _relative_to_known_root(original, project_root)
        if rel is None:
            candidates.append(original)
            return candidates
    else:
        rel = original

    for root in candidate_stable_roots(project_root):
        candidate = root / rel
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def resolve_existing_path(
    path: Path | str,
    *,
    project_root: Path | str | None = None,
    required: bool = False,
) -> Path:
    """Resolve a file or directory, falling back to the backup drive if needed."""
    candidates = fallback_candidates(path, project_root)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if required:
        searched = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(f"{path} not found. Searched:\n{searched}")
    return candidates[0]


def resolve_existing_file(
    path: Path | str,
    *,
    project_root: Path | str | None = None,
    required: bool = False,
) -> Path:
    original = Path(path).expanduser()
    candidates = fallback_candidates(path, project_root)
    # Keep the caller's absolute path even when root remapping is available.
    if original.is_absolute() and original not in candidates:
        candidates.append(original)
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    if required:
        searched = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(f"{path} not found as a file. Searched:\n{searched}")
    return candidates[0]


def resolve_existing_dir(
    path: Path | str,
    *,
    project_root: Path | str | None = None,
    required: bool = False,
    prefer_nonempty: bool = False,
    preferred_suffixes: set[str] | None = None,
) -> Path:
    candidates = fallback_candidates(path, project_root)
    if preferred_suffixes:
        suffixes = {suffix.lower() for suffix in preferred_suffixes}
        for candidate in candidates:
            if candidate.is_dir():
                try:
                    if any(child.is_file() and child.suffix.lower() in suffixes for child in candidate.iterdir()):
                        return candidate
                except OSError:
                    pass
    if prefer_nonempty:
        for candidate in candidates:
            if candidate.is_dir():
                try:
                    if any(candidate.iterdir()):
                        return candidate
                except OSError:
                    pass
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    if required:
        searched = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(f"{path} not found as a directory. Searched:\n{searched}")
    return candidates[0]


def np_load_existing(path: Any, *args: Any, project_root: Path | str | None = None, **kwargs: Any) -> Any:
    """``np.load`` wrapper that resolves missing study files from backup."""
    if isinstance(path, (str, os.PathLike)):
        path = resolve_existing_file(path, project_root=project_root)
    return np.load(path, *args, **kwargs)


def install_numpy_load_fallback(project_root: Path | str | None = None) -> None:
    """Patch ``numpy.load`` in the current process to use backup fallback paths."""
    current = np.load
    if getattr(current, "_vha_backup_fallback", False):
        return

    original_load = current

    def load_with_fallback(path: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(path, (str, os.PathLike)):
            path = resolve_existing_file(path, project_root=project_root)
        return original_load(path, *args, **kwargs)

    load_with_fallback._vha_backup_fallback = True  # type: ignore[attr-defined]
    load_with_fallback._vha_original_load = original_load  # type: ignore[attr-defined]
    np.load = load_with_fallback  # type: ignore[assignment]


def install_pil_image_open_fallback(project_root: Path | str | None = None) -> None:
    """Patch ``PIL.Image.open`` in the current process to use backup fallback paths."""
    from PIL import Image

    current = Image.open
    if getattr(current, "_vha_backup_fallback", False):
        return

    original_open = current

    def open_with_fallback(fp: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(fp, (str, os.PathLike)):
            fp = resolve_existing_file(fp, project_root=project_root)
        return original_open(fp, *args, **kwargs)

    open_with_fallback._vha_backup_fallback = True  # type: ignore[attr-defined]
    open_with_fallback._vha_original_open = original_open  # type: ignore[attr-defined]
    Image.open = open_with_fallback  # type: ignore[assignment]


def install_study_artifact_fallbacks(project_root: Path | str | None = None) -> None:
    """Install notebook-friendly fallback hooks for purged study artifacts."""
    install_numpy_load_fallback(project_root)
    install_pil_image_open_fallback(project_root)
