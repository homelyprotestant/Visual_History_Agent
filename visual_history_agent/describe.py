"""Description Agent — atom-primary formal painting descriptions (ReAct)."""

from __future__ import annotations

import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

from visual_history_agent.encode import encode_image
from visual_history_agent.paths import project_root, require_downstream_artifacts
from visual_history_agent.stream_subprocess import run_streaming


def formal_description_text(payload: dict[str, Any] | str | Path) -> str:
    """Return the final formal description from a payload dict or description JSON path."""
    if not isinstance(payload, dict):
        path = Path(payload).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
    text = str(
        payload.get("final_formal_description")
        or payload.get("formal_visual_description")
        or ""
    ).strip()
    if not text:
        raise KeyError(
            "No final_formal_description or formal_visual_description in payload"
        )
    return text


def wrap_for_display(text: str, *, width: int = 88) -> str:
    """Word-wrap paragraphs for terminal / notebook display."""
    blocks: list[str] = []
    for para in text.split("\n\n"):
        compact = " ".join(para.split())
        blocks.append(textwrap.fill(compact, width=width) if compact else "")
    return "\n\n".join(blocks)


def print_formal_description(
    payload: dict[str, Any] | str | Path,
    *,
    title: str | None = "Final formal description",
    width: int = 88,
) -> str:
    """Print (word-wrapped) and return the final formal description (unwrapped)."""
    text = formal_description_text(payload)
    if title:
        rule = "=" * min(width, 72)
        print(f"\n{rule}\n{title}\n{rule}\n", flush=True)
    print(wrap_for_display(text, width=width), flush=True)
    if title:
        print(f"\n{'=' * min(width, 72)}\n", flush=True)
    return text


def describe_image(
    image_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    coordinator_backend: str = "openai",
    coordinator_model: str = "gpt-5.1",
    react_rounds: int = 3,
    metadata_retrieval: str = "embedding-cosine",
    device: str = "mps",
    compile_pdf: bool = False,
    print_description: bool = True,
) -> dict[str, Any]:
    """Run the Description Agent on one painting; return description JSON (optional PDF)."""
    root = project_root()
    paths = require_downstream_artifacts(root)
    image_path = Path(image_path).expanduser().resolve()
    out = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else paths["outputs"] / "descriptions" / image_path.stem
    )
    out.mkdir(parents=True, exist_ok=True)

    print(f"[describe] encoding {image_path.name} (device={device})", flush=True)
    encoded = encode_image(image_path, output_dir=out / "embedding", device=device)
    if encoded.get("cached"):
        print("[describe] using cached embedding", flush=True)
    description_json = out / "painting_description_from_global_atoms.json"
    cmd = [
        sys.executable,
        str(root / "scripts" / "run_description_agent.py"),
        "--query-embedding",
        encoded["pooled_embedding"],
        "--query-image",
        str(image_path),
        "--atom-descriptions",
        str(paths["atom_descriptions"]),
        "--output",
        str(description_json),
        "--metadata-retrieval",
        metadata_retrieval,
        "--coordinator-backend",
        coordinator_backend,
        "--coordinator-model",
        coordinator_model,
        "--react-rounds",
        str(react_rounds),
    ]
    env = os.environ.copy()
    vendor = root / "vendor"
    if (vendor / "VisualHistory").is_dir():
        env.setdefault("VISUALHISTORY_ROOT", str(vendor))
    print(
        f"[describe] Description Agent starting → {description_json.name}",
        flush=True,
    )
    try:
        run_streaming(cmd, cwd=root, env=env)
    except RuntimeError as exc:
        raise RuntimeError(f"description agent failed: {exc}") from exc
    payload = json.loads(description_json.read_text(encoding="utf-8"))
    pdf_path = None
    if compile_pdf:
        pdf_cmd = [
            sys.executable,
            str(root / "scripts" / "render_painting_description_supplement.py"),
            str(description_json),
            "--output-dir",
            str(out / "supplement"),
            "--stem",
            image_path.stem,
            "--compile",
        ]
        try:
            run_streaming(pdf_cmd, cwd=root)
        except RuntimeError as exc:
            raise RuntimeError(f"description PDF render failed: {exc}") from exc
        pdfs = list((out / "supplement").glob("*.pdf"))
        pdf_path = str(pdfs[0]) if pdfs else None
    result = {
        "image_path": str(image_path),
        "output_dir": str(out),
        "description_json": str(description_json),
        "pdf": pdf_path,
        "payload": payload,
        "encoding": encoded,
        "final_formal_description": formal_description_text(payload),
    }
    if print_description:
        print_formal_description(payload)
    return result
