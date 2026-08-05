"""Hand-first then copy comparison for the Visual History Agent."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from visual_history_agent.describe import (
    describe_image,
    print_formal_description,
    wrap_for_display,
)
from visual_history_agent.paths import project_root


PROMPT_VERSION = "vha-hand-conditioned-copy-comparison-v1"
SAME_HAND_PROMPT_VERSION = "vha-same-hand-first-v1"

SAME_HAND_PROMPT = """Act as a cautious senior paintings connoisseur. Decide whether images A and B are consistent with the same painter's hand using only the supplied frozen individual-assessment JSON evidence. You cannot inspect either image, infer new pixel-level observations, or use any copy-direction assessment.

Return ONLY valid JSON:
{
  "same_hand_assessment": "probable same hand|possibly same hand|probably different hands|indeterminate",
  "confidence": "high|medium|low",
  "formal_closeness": "high|medium|low",
  "summary": "2-4 sentences",
  "evidence_for_same_hand": ["observation explicitly supported by both frozen records"],
  "evidence_for_different_hands": ["observation explicitly supported by both frozen records"],
  "uncertainties": ["limitation"]
}

Frozen evidence:
{evidence}
"""

COPY_PROMPT = """Act as a cautious senior paintings connoisseur. Using only the supplied frozen individual-assessment JSON evidence and the prior same-hand assessment, decide the copy relationship between A and B. Do not inspect images.

Critical order:
1. Respect the same-hand assessment already made.
2. Only then judge copying/closeness conditioned on that hand result.
3. If same-hand is indeterminate or probably different hands, do not claim a confident workshop copy chain unless the frozen evidence forces it.

Return ONLY valid JSON:
{
  "copy_relationship": "A_copies_B|B_copies_A|common_model_or_type|independent|indeterminate",
  "confidence": "high|medium|low",
  "summary": "2-4 sentences conditioned on the same-hand result",
  "same_hand_conditioning": "how the prior same-hand assessment constrained this decision",
  "evidence_for_copy_or_type": ["observation supported by frozen records"],
  "evidence_against_direct_copy": ["observation supported by frozen records"],
  "uncertainties": ["limitation"]
}

Same-hand assessment:
{same_hand}

Frozen evidence:
{evidence}
"""


def _visualhistory_root() -> Path:
    from visual_history_agent.paths import visualhistory_root

    return visualhistory_root()


def _ensure_vh() -> None:
    from visual_history_agent.paths import ensure_visualhistory_on_path

    ensure_visualhistory_on_path()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _role_chat(
    prompt: str,
    *,
    coordinator_backend: str = "openai",
    coordinator_model: str = "gpt-5.1",
) -> str:
    _ensure_vh()
    from VisualHistory.literature.gateway_llm import apply_gateway_env
    from VisualHistory.literature.literature_llm_roles import role_chat

    apply_gateway_env(backend=coordinator_backend, deployment=coordinator_model)
    return role_chat("answer", prompt, max_tokens=8192, temperature=0.1)


def _extract_json(raw: str) -> dict[str, Any]:
    _ensure_vh()
    from VisualHistory.literature.json_utils import extract_json_object

    payload = extract_json_object(raw)
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object from coordinator")
    return payload


def print_comparison_summary(
    comparison: dict[str, Any],
    *,
    width: int = 88,
) -> None:
    """Print only the same-hand and copy executive summary prose."""
    same_hand = comparison.get("same_hand_assessment") or {}
    copy_assessment = comparison.get("assessment") or {}
    rule = "=" * min(width, 72)
    print(f"\n{rule}\nExecutive summary\n{rule}\n", flush=True)
    same_summary = str(same_hand.get("summary") or "").strip()
    copy_summary = str(copy_assessment.get("summary") or "").strip()
    if same_summary:
        print(wrap_for_display(same_summary, width=width), flush=True)
    if same_summary and copy_summary:
        print(flush=True)
    if copy_summary:
        print(wrap_for_display(copy_summary, width=width), flush=True)
    if not same_summary and not copy_summary:
        print("(no summary text in comparison payload)", flush=True)
    print(f"\n{rule}\n", flush=True)


def compare_frozen_assessments(
    json_a: str | Path,
    json_b: str | Path,
    *,
    output_dir: str | Path | None = None,
    coordinator_backend: str = "openai",
    coordinator_model: str = "gpt-5.1",
    print_summary: bool = True,
) -> dict[str, Any]:
    """Compare two frozen individual-assessment JSON files (hand-first, then copy)."""
    root = project_root()
    json_a = Path(json_a).expanduser().resolve()
    json_b = Path(json_b).expanduser().resolve()
    out = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else root / "data" / "outputs" / "comparisons" / f"{json_a.stem}__vs__{json_b.stem}"
    )
    out.mkdir(parents=True, exist_ok=True)

    assessment_a = _load_json(json_a)
    assessment_b = _load_json(json_b)
    frozen_manifest = {
        "A": {"path": str(json_a), "sha256": _sha256_file(json_a)},
        "B": {"path": str(json_b), "sha256": _sha256_file(json_b)},
    }
    evidence = {"A": assessment_a, "B": assessment_b}

    same_hand_path = out / "same_hand_assessment.json"
    same_prompt = SAME_HAND_PROMPT.replace(
        "{evidence}", json.dumps(evidence, ensure_ascii=False, indent=2)
    )
    raw_hand = _role_chat(
        same_prompt,
        coordinator_backend=coordinator_backend,
        coordinator_model=coordinator_model,
    )
    same_hand = _extract_json(raw_hand)
    same_artifact = {
        "prompt_version": SAME_HAND_PROMPT_VERSION,
        "frozen_inputs": frozen_manifest,
        "coordinator_backend": coordinator_backend,
        "coordinator_model": coordinator_model,
        "assessment": same_hand,
    }
    _save_json(same_hand_path, same_artifact)

    copy_path = out / "copy_comparison.json"
    copy_prompt = COPY_PROMPT.replace(
        "{same_hand}", json.dumps(same_hand, ensure_ascii=False, indent=2)
    ).replace("{evidence}", json.dumps(evidence, ensure_ascii=False, indent=2))
    raw_copy = _role_chat(
        copy_prompt,
        coordinator_backend=coordinator_backend,
        coordinator_model=coordinator_model,
    )
    copy_assessment = _extract_json(raw_copy)
    comparison = {
        "prompt_version": PROMPT_VERSION,
        "frozen_inputs": frozen_manifest,
        "coordinator_backend": coordinator_backend,
        "coordinator_model": coordinator_model,
        "same_hand_assessment": same_hand,
        "assessment": copy_assessment,
        "method": {
            "hand_first": True,
            "copy_conditioned_on_same_hand": True,
            "agent": "Visual History Agent",
        },
    }
    _save_json(copy_path, comparison)
    result = {
        "output_dir": str(out),
        "same_hand_json": str(same_hand_path),
        "comparison_json": str(copy_path),
        "comparison": comparison,
    }
    if print_summary:
        print_comparison_summary(comparison)
    return result


def _compile_comparison_pdf(
    *,
    root: Path,
    package_json: Path,
    output_dir: Path,
    stem: str,
) -> str | None:
    cmd = [
        sys.executable,
        str(root / "scripts" / "render_hand_first_comparison_supplement.py"),
        str(package_json),
        "--output-dir",
        str(output_dir),
        "--stem",
        stem,
        "--compile",
    ]
    subprocess.run(cmd, check=True, cwd=str(root))
    pdfs = list(output_dir.glob(f"{stem}.pdf"))
    return str(pdfs[0]) if pdfs else None


def compare_images(
    image_a: str | Path,
    image_b: str | Path,
    *,
    output_dir: str | Path | None = None,
    device: str = "mps",
    coordinator_backend: str = "openai",
    coordinator_model: str = "gpt-5.1",
    print_descriptions: bool = False,
    print_summary: bool = True,
    compile_pdf: bool = True,
) -> dict[str, Any]:
    """Describe A and B if needed, then run frozen hand-first comparison.

    By default only the executive summary is printed (not the full A/B formal
    descriptions). Pass ``print_descriptions=True`` to also print those.
    """
    root = project_root()
    image_a = Path(image_a).expanduser().resolve()
    image_b = Path(image_b).expanduser().resolve()
    out = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else root / "data" / "outputs" / "comparisons" / f"{image_a.stem}__vs__{image_b.stem}"
    )
    out.mkdir(parents=True, exist_ok=True)
    # Individual description PDFs are not emitted; the comparison PDF embeds both full reports.
    desc_a = describe_image(
        image_a,
        output_dir=out / "individual" / "A",
        device=device,
        coordinator_backend=coordinator_backend,
        coordinator_model=coordinator_model,
        print_description=False,
        compile_pdf=False,
    )
    if print_descriptions:
        print_formal_description(
            desc_a["payload"],
            title=f"Final formal description — A ({image_a.name})",
        )
    desc_b = describe_image(
        image_b,
        output_dir=out / "individual" / "B",
        device=device,
        coordinator_backend=coordinator_backend,
        coordinator_model=coordinator_model,
        print_description=False,
        compile_pdf=False,
    )
    if print_descriptions:
        print_formal_description(
            desc_b["payload"],
            title=f"Final formal description — B ({image_b.name})",
        )
    result = compare_frozen_assessments(
        desc_a["description_json"],
        desc_b["description_json"],
        output_dir=out,
        coordinator_backend=coordinator_backend,
        coordinator_model=coordinator_model,
        print_summary=print_summary,
    )
    package = {
        "images": {"A": str(image_a), "B": str(image_b)},
        "coordinator_backend": coordinator_backend,
        "coordinator_model": coordinator_model,
        "descriptions": {"A": desc_a["payload"], "B": desc_b["payload"]},
        "comparison": result["comparison"],
        "individual_json": {
            "A": desc_a["description_json"],
            "B": desc_b["description_json"],
        },
    }
    package_json = out / "comparison_package.json"
    _save_json(package_json, package)
    comparison_pdf = None
    if compile_pdf:
        comparison_pdf = _compile_comparison_pdf(
            root=root,
            package_json=package_json,
            output_dir=out / "supplement",
            stem=f"{image_a.stem}__vs__{image_b.stem}",
        )
        if comparison_pdf:
            print(f"Comparison PDF (single combined report): {comparison_pdf}", flush=True)
    result["individual"] = {"A": desc_a, "B": desc_b}
    result["package_json"] = str(package_json)
    result["pdf"] = comparison_pdf
    return result
