"""Spatial coefficient maps and local-Dai atom browser entry points.

Pipeline: global EN atom selection → fixed-dict NMF LR maps → local Dai (OMP).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from visual_history_agent.spatial_browser import (
    SpatialBrowserBundle,
    display_spatial_atom_browser,
    prepare_spatial_browser,
)


def map_coefficients(
    image_path: str | Path,
    *,
    image_b: str | Path | None = None,
    output_dir: str | Path | None = None,
    atom_ids: Sequence[int] | None = None,
    top_k: int | None = None,
    run_local_dai: bool = True,
    force_recompute: bool = False,
    device: str = "mps",
    show_browser: bool = False,
) -> dict[str, Any]:
    """Run fixed-dictionary NMF LR maps + local-Dai (OMP) upsample to HR224."""
    if not run_local_dai:
        raise ValueError(
            "This entry point runs the local-Dai browser pipeline. "
            "Use prepare_spatial_browser(...) with run_local_dai semantics only."
        )
    bundle = prepare_spatial_browser(
        image_path,
        image_b,
        output_dir=output_dir,
        atom_ids=atom_ids,
        top_k=top_k,
        force_recompute=force_recompute,
        device=device,
    )
    if show_browser:
        display_spatial_atom_browser(bundle)
    return {
        "image_path": bundle.images["A"],
        "image_b": bundle.images.get("B"),
        "output_dir": str(bundle.output_dir),
        "browser_atom_ids": bundle.browser_atom_ids,
        "local_dai_maps": {
            label: str(
                bundle.output_dir
                / "spatial_local_dai_hr224"
                / f"image_{label.lower()}_local_dai_positive_maps.npz"
            )
            for label in bundle.labels
        },
        "bundle": bundle,
    }


def launch_atom_browser(
    image_a: str | Path,
    image_b: str | Path | None = None,
    **kwargs: Any,
) -> SpatialBrowserBundle:
    """Prepare Two_Image local Dai maps (cached) and open the interactive browser."""
    bundle = prepare_spatial_browser(image_a, image_b, **kwargs)
    display_spatial_atom_browser(bundle)
    return bundle
