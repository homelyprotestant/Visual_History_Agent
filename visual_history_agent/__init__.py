"""Visual History Agent — embeddings to formal painting description, maps, and comparison."""

from visual_history_agent.compare import (
    compare_frozen_assessments,
    compare_images,
    print_comparison_summary,
)
from visual_history_agent.describe import (
    describe_image,
    formal_description_text,
    print_formal_description,
)
from visual_history_agent.maps import launch_atom_browser, map_coefficients
from visual_history_agent.spatial_browser import (
    display_spatial_atom_browser,
    prepare_spatial_browser,
)

__all__ = [
    "describe_image",
    "print_formal_description",
    "formal_description_text",
    "map_coefficients",
    "prepare_spatial_browser",
    "display_spatial_atom_browser",
    "launch_atom_browser",
    "compare_images",
    "compare_frozen_assessments",
    "print_comparison_summary",
]
__version__ = "0.1.0"
