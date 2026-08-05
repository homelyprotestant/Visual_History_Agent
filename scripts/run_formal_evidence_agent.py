#!/usr/bin/env python3
"""Formal Evidence Agent — painterly labels for global dictionary atoms.

Retrieval uses the complete mean-centered 2304-D Run4 vectors. The top paintings'
catalogue descriptions are chunked with VisualHistory, scored by the Formal Evidence
Agent (Qwen RCS), then reranked and synthesized by the Visual History Agent cloud
coordinator.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual_history_agent.paths import ensure_visualhistory_on_path  # noqa: E402

VISUALHISTORY_ROOT = ensure_visualhistory_on_path(PROJECT_ROOT)

from VisualHistory.literature.json_utils import extract_json_object  # noqa: E402
from VisualHistory.literature.gateway_llm import apply_gateway_env  # noqa: E402
from VisualHistory.literature.literature_llm_roles import role_chat  # noqa: E402
from VisualHistory.literature.paragraph_split import SplitParagraph  # noqa: E402
from VisualHistory.literature.atomic_chunker import (  # noqa: E402
    atomize_with_overlap,
    default_atomic_chunk_tokens,
    default_atomic_overlap_tokens,
)


from visual_history_agent.paths import artifact_paths as _vha_paths

_VHA = _vha_paths(PROJECT_ROOT)
DATA_ROOT = _VHA["data_root"]
DICT_ROOT = _VHA["ksvd_dir"]
INFER_ROOT = DATA_ROOT / "embeddings"
DEFAULT_OUTPUT = _VHA["atom_descriptions_dir"]

DICTIONARY_PATH = _VHA["dictionary"]
MEAN_PATH = _VHA["embedding_mean"]
EMBEDDINGS_PATH = _VHA["embeddings"]
DATABASE_PATH = _VHA["database"]

TOP_K = 40
RCS_MODEL = "qwen2.5:7b"
COORDINATOR_BACKEND = "azure-openai-fallback"
COORDINATOR_MODEL = "gpt-5.1"
RCS_CACHE_VERSION = "all-paintings-visualhistory-atomic-v5"

PAINTERLY_QUESTION = """Which passages provide detailed, concrete visual evidence about
how the paintings look and how their pictorial effects are constructed? Prioritize
form, shape, line, contour, color relationships, hue, saturation, light, shadow,
modeling, spatial depth, texture, surface, edges, drapery, composition, scale,
pattern, brushwork, paint handling, layering, underdrawing, and finish. Art-historical
details such as workshop practice, period conventions, influence, iconography, or
function are useful when they clarify those visible qualities. Subject matter alone
is weak evidence unless tied to visible form such as a virgin and child forming a pyramidal shape which is a halmark of Renaissance compositions."""

RCS_PROMPT = """You are an expert art historian. Assess one painting
catalogue-description chunk for its usefulness in interpreting a learned visual atom.

Return ONLY valid JSON:
{{
  "relevance_score": <integer 1-10>,
  "contextual_sentences": [
    "<one complete, detailed sentence preserving concrete painterly evidence>",
    "<one complete, detailed sentence preserving concrete painterly evidence>",
    "<include 1-8 entries, matching the amount of evidence in the passage>"
  ],
  "material_terms": []
}}

Scoring:
- 8-10: specific, detailed evidence about form, line, contour, color, light, shadow,
  modeling, spatial depth, texture, surface, edge, drapery, composition, pattern,
  brushwork, paint handling, layering, underdrawing, or finish.
- 4-7: useful visible composition or spatial organization but limited painterly detail.
- 1-3: primarily attribution, provenance, date, iconography, or narrative.
- Use only the passage. Do not infer pigments, techniques, colors, or handling.
- Preserve precise visual particulars rather than replacing them with generic phrases:
  name the described colors, tonal contrasts, kinds of edges, textures, poses,
  spatial devices, surface effects, and evidence of execution when present.
- Explain how those particulars produce volume, depth, emphasis, rhythm, luminosity,
  material differentiation, or another visible effect when the passage supports it.
- Include artist, workshop, influence, chronology, iconography, or devotional
  function only where it helps interpret the visual or technical evidence.
- If the passage has little painterly evidence, say exactly what visible information
  it does provide instead of padding the summary with generalities.
- Return 1-8 `contextual_sentences` array entries. Each entry must be exactly one
  complete sentence; do not combine sentences within an entry or use abbreviations
  containing periods.

Research question:
{question}

Passage metadata:
Title: {title}
Authors: {authors}
Year: {year}
DOI: {doi}
Section: {section}
Concept: {concept}

Passage text:
{text}
"""

COORDINATOR_RERANK_PROMPT = """You are the reranker for one learned visual dictionary
atom. Its complete mean-centered 2304-dimensional vector retrieved 40 painting
descriptions from the full corpus. Qwen extracted and scored painterly evidence from
their chunks. The candidates remain in cosine-retrieval order.

Task:
Rerank all 40 candidates by usefulness for interpreting the atom, jointly considering
atom association (cosine), concrete painterly evidence, and cross-candidate recurrence.
Do not copy either cosine order or Qwen-score order mechanically.

Return ONLY valid JSON containing every CANDIDATE number exactly once:
{{
  "reranked_description_ranks": [<40 integer CANDIDATE numbers, most to least useful>]
}}

Cosine-ordered candidates with Qwen evidence:
{evidence}
"""

COORDINATOR_SYNTHESIS_PROMPT = """You are the cloud synthesizer labeling one learned visual
dictionary atom. The 40 painting-description candidates below have already been
reranked by the cloud coordinator from most to least useful.

Task:
Synthesize a precise painterly label and a detailed 4-8 sentence description of the
recurring visual qualities supported across multiple descriptions. Make painterly
analysis the main thrust. You may add art-historical context where it explains the
handling, pictorial construction, workshop practice, period convention, influence,
iconographic function, or viewing effect.

Painterly qualities include form, shape, line, contour, color, hue, saturation,
shade, light, shadow, modeling, spatial depth, texture, surface, edge, drapery,
composition, scale, pattern, and brushwork.

Critical rules:
- Use ONLY the supplied Qwen summaries, which derive only from description text.
- Treat cosine retrieval as atom association and Qwen RCS as painterly usefulness;
  neither score is itself visual evidence.
- Aggregate across multiple descriptions; never center the answer on one painting.
- Move from concrete particulars to interpretation: specify the recurring colors,
  tonal structures, contours, edges, modeling, textures, surfaces, drapery systems,
  spatial devices, compositional rhythms, brushwork, layering, or finish before
  explaining their visual effect.
- Prefer exact evidence over umbrella phrases such as "detailed brushwork,"
  "spatial depth," "dramatic lighting," or "rich texture." If using such a phrase,
  immediately state what concrete features justify it.
- Distinguish observed appearance from documented process. For example, do not call
  handling smooth, blended, linear, wet-into-wet, glazed, or impastoed unless the
  supplied evidence supports that wording.
- Do not turn repeated iconography or narrative into a painterly claim unless the
  descriptions explicitly connect it to visible form, color, light, space, or surface.
- Do not invent pigments, materials, techniques, colors, or brushwork.
- Use a painterly label about visible form or handling; do not use generic labels
  such as "religious devotion", "religious scene", or "portraiture" alone.
- Claim a recurring painterly quality only when at least two descriptions support it.
- If the descriptions contain little painterly language, state a cautious,
  subject/compositional reading and set confidence to "low".
- Return 4-8 `painterly_sentences` array entries. Each entry must be exactly one
  complete sentence; do not combine sentences within an entry.

Return ONLY valid JSON:
{{
  "label": "<3-8 word painterly label>",
  "painterly_sentences": [
    "<one complete, information-dense painterly sentence>",
    "<one complete, information-dense painterly sentence>",
    "<continue until there are 4-8 array entries total>"
  ],
  "confidence": "high|medium|low",
  "evidence_summary": [
    "<feature; supported by CANDIDATE numbers X and Y>",
    "<another feature and supporting CANDIDATE numbers, if supported>"
  ]
}}

Coordinator-reranked candidates with Qwen evidence:
{evidence}
"""


def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str:
    normalized = {" ".join(str(c).strip().lower().split()): str(c) for c in df.columns}
    for candidate in candidates:
        key = " ".join(candidate.strip().lower().split())
        if key in normalized:
            return normalized[key]
    raise KeyError(f"None of {candidates!r} found in database")


def _safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\x00", " ").strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _description_chunks(text: str) -> list[str]:
    """Apply VisualHistory's atomic overlapping RAG windows to one painting."""
    source = SplitParagraph(page=1, text=text, local_index=1, section="description")
    chunks = atomize_with_overlap([source])
    return [chunk.text.strip() for chunk in chunks if chunk.text.strip()] or [text]


def _load_database_and_columns() -> tuple[pd.DataFrame, dict[str, str]]:
    database = pd.read_excel(DATABASE_PATH)
    columns = {
        "period": _resolve_column(
            database,
            [
                "art_historical_time_period",
                "art historical time period",
                "time_period",
            ],
        ),
        "description": _resolve_column(
            database,
            ["description", "Description", "description_raw", "Painting Name/ Description"],
        ),
        "painting_name": _resolve_column(
            database,
            ["painting_name", "painting name", "title", "Painting Name"],
        ),
        "artist": _resolve_column(database, ["artist", "Artist"]),
    }
    return database, columns


def _load_retrieval_data() -> tuple[np.ndarray, pd.DataFrame, dict[str, str]]:
    dictionary = np.load(DICTIONARY_PATH).astype(np.float32)
    center = np.load(MEAN_PATH).astype(np.float32)
    pooled = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    valid = np.ones(int(pooled.shape[0]), dtype=bool)
    database, columns = _load_database_and_columns()
    if len(database) != int(pooled.shape[0]):
        raise ValueError(
            f"Database rows {len(database)} != pooled embedding rows {pooled.shape[0]}"
        )
    if dictionary.shape != (1000, 2304):
        raise ValueError(f"Expected dictionary (1000, 2304), got {dictionary.shape}")
    if center.shape != (2304,):
        raise ValueError(f"Expected embedding mean (2304,), got {center.shape}")

    has_description = (
        database[columns["description"]].map(_safe_text).str.len().gt(0).to_numpy(dtype=bool)
    )
    candidate_rows = np.flatnonzero(valid & has_description)
    matrix = np.asarray(pooled[candidate_rows], dtype=np.float64)
    finite = np.isfinite(matrix).all(axis=1)
    candidate_rows = candidate_rows[finite]
    matrix = matrix[finite]
    matrix = matrix - center.astype(np.float64, copy=False)[None, :]
    norms = np.linalg.norm(matrix, axis=1)
    nonzero = np.isfinite(norms) & (norms > 1e-8)
    candidate_rows = candidate_rows[nonzero]
    matrix = matrix[nonzero]
    norms = norms[nonzero]
    matrix_unit = matrix / norms[:, None]

    dictionary_64 = dictionary.astype(np.float64, copy=False)
    dictionary_norms = np.linalg.norm(dictionary_64, axis=1)
    dictionary_unit = dictionary_64 / np.maximum(dictionary_norms[:, None], 1e-12)
    # Accelerate/vecLib emits spurious floating-point warnings for this valid
    # matrix multiplication on some Apple builds. The explicit contraction is
    # stable and produces finite cosine scores.
    similarities = np.einsum(
        "ij,kj->ik",
        dictionary_unit,
        matrix_unit,
        optimize=False,
    )
    top_positions = np.argsort(similarities, axis=1)[:, ::-1][:, :TOP_K]

    retrieval_rows: list[dict[str, Any]] = []
    for atom_id in tqdm(
        range(dictionary.shape[0]),
        desc="Formal Evidence retrieval",
        unit="atom",
        dynamic_ncols=True,
    ):
        hits: list[dict[str, Any]] = []
        for retrieval_rank, position in enumerate(top_positions[atom_id], start=1):
            database_row = int(candidate_rows[int(position)])
            description = _safe_text(database.iloc[database_row][columns["description"]])
            hits.append(
                {
                    "retrieval_rank": retrieval_rank,
                    "database_row": database_row,
                    "cosine": float(similarities[atom_id, int(position)]),
                    "painting_name": _safe_text(
                        database.iloc[database_row][columns["painting_name"]]
                    ),
                    "artist": _safe_text(database.iloc[database_row][columns["artist"]]),
                    "description": description,
                    "chunks": _description_chunks(description) if description else [],
                }
            )
        retrieval_rows.append({"atom_id": atom_id, "hits": hits})

    return np.asarray(retrieval_rows, dtype=object), database, columns


def _write_frozen_retrieval_manifest(path: Path, retrieval: np.ndarray) -> None:
    samples: dict[int, dict[str, Any]] = {}
    atom_records: list[dict[str, Any]] = []
    for atom_value in retrieval:
        atom = atom_value.item() if hasattr(atom_value, "item") else atom_value
        frozen_hits: list[dict[str, Any]] = []
        for hit in atom["hits"]:
            database_row = int(hit["database_row"])
            chunks = [str(chunk) for chunk in hit.get("chunks", [])]
            samples.setdefault(
                database_row,
                {
                    "record_type": "sample",
                    "database_row": database_row,
                    "painting_name": str(hit.get("painting_name", "")),
                    "artist": str(hit.get("artist", "")),
                    "description": str(hit.get("description", "")),
                    "chunks": chunks,
                    "chunk_keys": [_chunk_key(chunk) for chunk in chunks],
                },
            )
            frozen_hits.append(
                {
                    "retrieval_rank": int(hit["retrieval_rank"]),
                    "database_row": database_row,
                    "cosine": float(hit["cosine"]),
                }
            )
        atom_records.append(
            {
                "record_type": "atom",
                "atom_id": int(atom["atom_id"]),
                "hits": frozen_hits,
            }
        )

    header = {
        "record_type": "metadata",
        "format_version": 1,
        "top_k": TOP_K,
        "rcs_cache_version": RCS_CACHE_VERSION,
        "atom_count": len(atom_records),
        "sample_count": len(samples),
        "dictionary_path": str(DICTIONARY_PATH),
        "embedding_path": str(EMBEDDINGS_PATH),
        "embedding_mean_path": str(MEAN_PATH),
        "database_path": str(DATABASE_PATH),
    }
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header, ensure_ascii=False) + "\n")
        for database_row in sorted(samples):
            handle.write(json.dumps(samples[database_row], ensure_ascii=False) + "\n")
        for record in atom_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def _load_frozen_retrieval_manifest(path: Path) -> np.ndarray:
    metadata: dict[str, Any] | None = None
    samples: dict[int, dict[str, Any]] = {}
    atoms: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            record_type = record.get("record_type")
            if record_type == "metadata":
                metadata = record
            elif record_type == "sample":
                samples[int(record["database_row"])] = record
            elif record_type == "atom":
                atoms[int(record["atom_id"])] = record
    if metadata is None:
        raise ValueError(f"Frozen retrieval manifest lacks metadata: {path}")
    if int(metadata.get("format_version", -1)) != 1:
        raise ValueError(f"Unsupported frozen retrieval format: {metadata}")
    if int(metadata.get("top_k", -1)) != TOP_K:
        raise ValueError(
            f"Frozen retrieval TOP_K={metadata.get('top_k')} does not match TOP_K={TOP_K}"
        )
    if str(metadata.get("rcs_cache_version")) != RCS_CACHE_VERSION:
        raise ValueError(
            "Frozen retrieval RCS cache version does not match current chunk-key version"
        )
    if sorted(atoms) != list(range(1000)):
        raise ValueError(f"Frozen retrieval manifest must contain atoms 0-999: {path}")

    retrieval_rows: list[dict[str, Any]] = []
    for atom_id in range(1000):
        hits: list[dict[str, Any]] = []
        for frozen_hit in atoms[atom_id]["hits"]:
            database_row = int(frozen_hit["database_row"])
            sample = samples.get(database_row)
            if sample is None:
                raise ValueError(
                    f"Atom {atom_id} references missing sample row {database_row}"
                )
            chunks = [str(chunk) for chunk in sample.get("chunks", [])]
            expected_keys = [str(key) for key in sample.get("chunk_keys", [])]
            actual_keys = [_chunk_key(chunk) for chunk in chunks]
            if actual_keys != expected_keys:
                raise ValueError(
                    f"Frozen chunk keys do not match stored text for database row {database_row}"
                )
            hits.append(
                {
                    "retrieval_rank": int(frozen_hit["retrieval_rank"]),
                    "database_row": database_row,
                    "cosine": float(frozen_hit["cosine"]),
                    "painting_name": str(sample.get("painting_name", "")),
                    "artist": str(sample.get("artist", "")),
                    "description": str(sample.get("description", "")),
                    "chunks": chunks,
                }
            )
        if len(hits) != TOP_K:
            raise ValueError(
                f"Frozen atom {atom_id} has {len(hits)} hits; expected {TOP_K}"
            )
        retrieval_rows.append({"atom_id": atom_id, "hits": hits})
    return np.asarray(retrieval_rows, dtype=object)


def _load_or_create_frozen_retrieval(
    path: Path,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, str]]:
    if path.is_file():
        retrieval = _load_frozen_retrieval_manifest(path)
        database, columns = _load_database_and_columns()
        print(f"Loaded frozen atom→sample→chunk manifest: {path}", flush=True)
        return retrieval, database, columns
    retrieval, database, columns = _load_retrieval_data()
    _write_frozen_retrieval_manifest(path, retrieval)
    print(f"Saved frozen atom→sample→chunk manifest: {path}", flush=True)
    return retrieval, database, columns


def _chunk_key(text: str) -> str:
    cache_input = f"{RCS_CACHE_VERSION}\n{text.strip()}"
    return hashlib.sha256(cache_input.encode("utf-8")).hexdigest()


def _split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]


def _count_sentences(text: str) -> int:
    return len(_split_sentences(text))


def _load_rcs_cache(path: Path) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return cached
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                cached[str(record["chunk_key"])] = record
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    return cached


def _score_chunk(text: str, model: str) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            prompt = RCS_PROMPT.format(
                question=PAINTERLY_QUESTION,
                title="",
                authors="",
                year="",
                doi="",
                section="description",
                concept="painting catalogue description",
                text=text,
            )
            raw = role_chat("rcs", prompt, max_tokens=1024, temperature=0.1)
            payload = extract_json_object(raw)
            raw_sentences = payload.get("contextual_sentences")
            if isinstance(raw_sentences, str):
                raw_sentences = [raw_sentences]
            if not isinstance(raw_sentences, list):
                alternate = payload.get("contextual_summary") or payload.get("summary")
                if isinstance(alternate, str) and alternate.strip():
                    raw_sentences = [alternate]
            if not isinstance(raw_sentences, list):
                raise ValueError("Qwen contextual_sentences must be a JSON array")
            sentences = [
                sentence
                for entry in raw_sentences
                for sentence in _split_sentences(str(entry))
            ]
            sentences = [
                sentence if sentence[-1:] in {".", "!", "?"} else f"{sentence}."
                for sentence in sentences
            ]
            if not 1 <= len(sentences) <= 8:
                raise ValueError(
                    "Qwen contextual_sentences must contain 1-8 sentences after normalization, "
                    f"got {len(sentences)}"
                )
            score = float(payload.get("relevance_score"))
            if not np.isfinite(score) or not 1 <= score <= 10:
                raise ValueError(f"Invalid Qwen relevance_score: {score!r}")
            return {
                "chunk_key": _chunk_key(text),
                "text": text,
                "relevance_score": score,
                "contextual_summary": " ".join(sentences),
                "sentence_count": len(sentences),
                "rcs_model": model,
            }
        except Exception as exc:
            last_error = exc
            print(
                f"Qwen RCS attempt {attempt + 1}/5 failed: {exc}",
                flush=True,
            )
            if attempt < 4:
                time.sleep(min(30.0, 2.0**attempt))
    print(
        "Qwen RCS exhausted retries; saving a low-relevance fallback so the bulk "
        f"cache can continue: {last_error}",
        flush=True,
    )
    return {
        "chunk_key": _chunk_key(text),
        "text": text,
        "relevance_score": 1.0,
        "contextual_summary": (
            "No concrete painterly evidence was reliably extracted from this chunk."
        ),
        "sentence_count": 1,
        "rcs_model": model,
        "rcs_fallback": True,
        "rcs_error": str(last_error),
    }


def _ensure_rcs_cache(
    retrieval: np.ndarray,
    atom_ids: list[int],
    cache_path: Path,
    *,
    model: str,
    workers: int,
) -> dict[str, dict[str, Any]]:
    cached = _load_rcs_cache(cache_path)
    unique: dict[str, str] = {}
    for atom_id in atom_ids:
        row = retrieval[atom_id].item() if hasattr(retrieval[atom_id], "item") else retrieval[atom_id]
        for hit in row["hits"]:
            for chunk in hit["chunks"]:
                unique.setdefault(_chunk_key(chunk), chunk)
    pending = [(key, text) for key, text in unique.items() if key not in cached]
    print(
        f"Qwen RCS chunks | unique={len(unique)} cached={len(unique) - len(pending)} "
        f"pending={len(pending)} model={model}",
        flush=True,
    )
    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_score_chunk, text, model): key for key, text in pending}
        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Qwen RCS ({len(unique) - len(pending)} cached)",
            unit="chunk",
            dynamic_ncols=True,
            mininterval=1.0,
        )
        for future in progress:
            record = future.result()
            with write_lock:
                with cache_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                cached[record["chunk_key"]] = record
    return cached


def _enrich_hits_with_qwen(
    hits: list[dict[str, Any]],
    rcs_cache: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for hit in hits:
        chunk_records = [rcs_cache[_chunk_key(chunk)] for chunk in hit["chunks"]]
        best_score = max((float(record["relevance_score"]) for record in chunk_records), default=0.0)
        summaries = [
            record["contextual_summary"]
            for record in sorted(
                chunk_records,
                key=lambda record: float(record["relevance_score"]),
                reverse=True,
            )
            if record["contextual_summary"]
        ]
        enriched.append(
            {
                **hit,
                "qwen_rcs_score": best_score,
                "qwen_rcs_summaries": summaries,
            }
        )
    return enriched


def _evidence_text(hits: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for hit in hits:
        summaries = hit["qwen_rcs_summaries"] or ["No concrete painterly evidence extracted."]
        summary_text = "\n".join(f"  - {text}" for text in summaries)
        blocks.append(
            f"CANDIDATE {hit['retrieval_rank']} | embedding rank {hit['retrieval_rank']} | "
            f"Qwen RCS {hit['qwen_rcs_score']:.1f}/10 | cosine {hit['cosine']:.6f}\n"
            f"{summary_text}"
        )
    return "\n\n".join(blocks)


def _validate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    label = str(payload.get("label") or "").strip()
    raw_sentences = payload.get("painterly_sentences")
    confidence = str(payload.get("confidence") or "").strip().lower()
    if not label:
        raise ValueError("Missing label")
    if not isinstance(raw_sentences, list):
        raise ValueError("painterly_sentences must be a JSON array")
    sentences = [
        sentence
        for entry in raw_sentences
        for sentence in _split_sentences(str(entry))
    ]
    sentences = [
        sentence if sentence[-1:] in {".", "!", "?"} else f"{sentence}."
        for sentence in sentences
    ]
    if not 4 <= len(sentences) <= 8:
        raise ValueError(
            f"painterly_sentences must contain 4-8 sentences after normalization, got {len(sentences)}"
        )
    if confidence not in {"high", "medium", "low"}:
        raise ValueError(f"Invalid confidence: {confidence!r}")
    notes = payload.get("evidence_summary") or []
    if isinstance(notes, str):
        notes = [notes]
    return {
        "label": label,
        "painterly_description": " ".join(sentences),
        "sentence_count": len(sentences),
        "confidence": confidence,
        "evidence_summary": [str(note).strip() for note in notes if str(note).strip()],
    }


def _validate_rerank_payload(
    payload: dict[str, Any],
    expected_ids: list[int],
) -> list[int]:
    raw_rerank = payload.get("reranked_description_ranks")
    if not isinstance(raw_rerank, list):
        raise ValueError("reranked_description_ranks must be a JSON array")
    reranked_ids = [int(rank) for rank in raw_rerank]
    if len(reranked_ids) != len(expected_ids) or set(reranked_ids) != set(expected_ids):
        raise ValueError(
            "Coordinator rerank must contain every candidate exactly once; "
            f"expected {expected_ids}, got {reranked_ids}"
        )
    return reranked_ids


def _synthesize_atom(
    atom_id: int,
    hits: list[dict[str, Any]],
    *,
    rcs_cache: dict[str, dict[str, Any]],
    rcs_model: str,
    coordinator_backend: str,
    coordinator_model: str,
) -> dict[str, Any]:
    enriched_hits = _enrich_hits_with_qwen(hits, rcs_cache)
    expected_ids = [int(hit["retrieval_rank"]) for hit in enriched_hits]
    rerank_prompt = COORDINATOR_RERANK_PROMPT.format(evidence=_evidence_text(enriched_hits))
    last_error: Exception | None = None
    for attempt in range(5):
        raw = ""
        try:
            raw = role_chat("answer", rerank_prompt, max_tokens=16384, temperature=0.1)
            reranked_ids = _validate_rerank_payload(
                extract_json_object(raw),
                expected_ids,
            )
            break
        except Exception as exc:
            last_error = exc
            preview = raw.replace("\n", " ")[:500]
            print(
                f"Coordinator rerank atom {atom_id} attempt {attempt + 1}/5 failed: "
                f"{exc} | response={preview!r}",
                flush=True,
            )
            if attempt < 4:
                time.sleep(min(60.0, 2.0 ** attempt * 3.0))
    else:
        raise RuntimeError(f"Atom {atom_id} rerank failed after retries: {last_error}")

    hit_by_rank = {
        int(hit["retrieval_rank"]): hit
        for hit in enriched_hits
    }
    gemini_reranked_hits = [
        {
            **hit_by_rank[retrieval_rank],
            "gemini_rerank": gemini_rank,
        }
        for gemini_rank, retrieval_rank in enumerate(reranked_ids, start=1)
    ]
    synthesis_prompt = COORDINATOR_SYNTHESIS_PROMPT.format(
        evidence=_evidence_text(gemini_reranked_hits)
    )
    last_error = None
    for attempt in range(5):
        raw = ""
        try:
            raw = role_chat("answer", synthesis_prompt, max_tokens=16384, temperature=0.1)
            payload = _validate_payload(extract_json_object(raw))
            return {
                "atom_id": atom_id,
                "rcs_model": rcs_model,
                "coordinator_backend": coordinator_backend,
                "coordinator_model": coordinator_model,
                "reranked_description_ranks": reranked_ids,
                **payload,
                "top_40_retrieval": gemini_reranked_hits,
            }
        except Exception as exc:
            last_error = exc
            preview = raw.replace("\n", " ")[:500]
            print(
                f"Coordinator synthesis atom {atom_id} attempt {attempt + 1}/5 failed: "
                f"{exc} | response={preview!r}",
                flush=True,
            )
            if attempt < 4:
                time.sleep(min(60.0, 2.0 ** attempt * 3.0))
    raise RuntimeError(f"Atom {atom_id} synthesis failed after retries: {last_error}")


def _load_completed(path: Path) -> dict[int, dict[str, Any]]:
    completed: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("coordinator_model") and record.get("rcs_model"):
                    completed[int(record["atom_id"])] = record
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return completed


@contextmanager
def _exclusive_checkpoint_lock(lock_path: Path):
    """Serialize JSONL appends and consolidated exports across processes."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _write_exports(output_dir: Path, completed: dict[int, dict[str, Any]]) -> None:
    ordered = [completed[key] for key in sorted(completed)]
    json_path = output_dir / "global_atom_painterly_descriptions.json"
    temp_json = json_path.with_suffix(".json.tmp")
    temp_json.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")
    temp_json.replace(json_path)

    csv_path = output_dir / "global_atom_painterly_descriptions.csv"
    temp_csv = csv_path.with_suffix(".csv.tmp")
    with temp_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "atom_id",
                "label",
                "painterly_description",
                "sentence_count",
                "confidence",
                "rcs_model",
                "coordinator_backend",
                "coordinator_model",
                "reranked_description_ranks",
                "evidence_summary",
            ],
        )
        writer.writeheader()
        for record in ordered:
            writer.writerow(
                {
                    "atom_id": record["atom_id"],
                    "label": record["label"],
                    "painterly_description": record["painterly_description"],
                    "sentence_count": record["sentence_count"],
                    "confidence": record["confidence"],
                    "rcs_model": record["rcs_model"],
                    "coordinator_backend": record["coordinator_backend"],
                    "coordinator_model": record["coordinator_model"],
                    "reranked_description_ranks": json.dumps(
                        record["reranked_description_ranks"]
                    ),
                    "evidence_summary": " | ".join(record.get("evidence_summary") or []),
                }
            )
    temp_csv.replace(csv_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rcs-model", default=RCS_MODEL)
    parser.add_argument(
        "--coordinator-backend",
        choices=["azure", "gemini", "openai", "azure-openai-fallback"],
        default=COORDINATOR_BACKEND,
    )
    parser.add_argument("--coordinator-model", default=COORDINATOR_MODEL)
    parser.add_argument("--rcs-workers", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--start-atom", type=int, default=0)
    parser.add_argument("--end-atom", type=int, default=1000)
    stage = parser.add_mutually_exclusive_group()
    stage.add_argument(
        "--rcs-only",
        action="store_true",
        help="Formal Evidence Agent: populate the Qwen RCS cache, then stop before cloud synthesis.",
    )
    stage.add_argument(
        "--synthesis-only",
        action="store_true",
        help="Formal Evidence Agent: require complete RCS cache, then cloud rerank/synthesize atom labels.",
    )
    parser.add_argument(
        "--atom-ids",
        help="Comma-separated global atom IDs; overrides --start-atom/--end-atom.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "global_atom_painterly_descriptions.jsonl"
    checkpoint_lock_path = output_dir / ".global_atom_painterly_descriptions.lock"
    error_path = output_dir / "errors.jsonl"
    retrieval_path = output_dir / "retrieval_manifest.json"
    frozen_retrieval_path = output_dir / "atom_sample_chunk_manifest.jsonl"
    rcs_cache_path = output_dir / "qwen_rcs_chunk_cache.jsonl"

    if args.force:
        with _exclusive_checkpoint_lock(checkpoint_lock_path):
            for stale_path in (
                checkpoint_path,
                error_path,
                output_dir / "global_atom_painterly_descriptions.json",
                output_dir / "global_atom_painterly_descriptions.csv",
            ):
                stale_path.unlink(missing_ok=True)

    retrieval, database, columns = _load_or_create_frozen_retrieval(
        frozen_retrieval_path
    )
    if args.atom_ids:
        try:
            requested = list(
                dict.fromkeys(
                    int(value.strip())
                    for value in args.atom_ids.split(",")
                    if value.strip()
                )
            )
        except ValueError as exc:
            parser.error(f"--atom-ids must be comma-separated integers: {exc}")
        invalid_ids = [atom_id for atom_id in requested if not 0 <= atom_id < 1000]
        if invalid_ids:
            parser.error(f"--atom-ids values must be in [0, 999], got {invalid_ids}")
        if not requested:
            parser.error("--atom-ids did not contain any atom IDs")
    else:
        requested = list(range(max(0, args.start_atom), min(1000, args.end_atom)))
    os.environ["LITERATURE_RCS_BACKEND"] = "ollama"
    os.environ["LITERATURE_RCS_MODEL"] = args.rcs_model
    apply_gateway_env(
        backend=args.coordinator_backend,
        deployment=args.coordinator_model,
    )
    if args.synthesis_only:
        rcs_cache = _load_rcs_cache(rcs_cache_path)
        required_chunk_keys = {
            _chunk_key(chunk)
            for atom_id in requested
            for hit in (
                retrieval[atom_id].item()
                if hasattr(retrieval[atom_id], "item")
                else retrieval[atom_id]
            )["hits"]
            for chunk in hit["chunks"]
        }
        missing_chunk_keys = required_chunk_keys.difference(rcs_cache)
        if missing_chunk_keys:
            raise RuntimeError(
                "Synthesis-only mode requires complete RCS coverage; "
                f"{len(missing_chunk_keys)} chunk keys are missing"
            )
        print(
            f"Synthesis-only RCS validation complete | required={len(required_chunk_keys)} "
            f"cached={len(rcs_cache)} missing=0",
            flush=True,
        )
    else:
        rcs_cache = _ensure_rcs_cache(
            retrieval,
            requested,
            rcs_cache_path,
            model=args.rcs_model,
            workers=args.rcs_workers,
        )
    metadata = {
        "scope": "all valid paintings with non-empty descriptions",
        "retrieval": "whole mean-centered 2304-D atom cosine against whole mean-centered Run4 pooled vectors",
        "top_k": TOP_K,
        "requested_atom_ids": requested,
        "chunker": "VisualHistory.literature.atomic_chunker.atomize_with_overlap",
        "chunk_tokens": {
            "target": default_atomic_chunk_tokens(),
            "overlap": default_atomic_overlap_tokens(),
        },
        "rcs_cache_version": RCS_CACHE_VERSION,
        "pipeline_roles": {
            "chunking": "VisualHistory atomic sentence windows with token overlap",
            "formal_evidence_agent": args.rcs_model,
            "coordinator_backend": args.coordinator_backend,
            "coordinator_reranker_and_synthesizer": args.coordinator_model,
        },
        "dictionary_path": str(DICTIONARY_PATH),
        "embedding_path": str(EMBEDDINGS_PATH),
        "embedding_mean_path": str(MEAN_PATH),
        "database_path": str(DATABASE_PATH),
        "database_rows": len(database),
        "columns": columns,
        "frozen_atom_sample_chunk_manifest": str(frozen_retrieval_path),
        "stage_mode": (
            "rcs_only"
            if args.rcs_only
            else "synthesis_only"
            if args.synthesis_only
            else "full"
        ),
    }
    retrieval_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if args.rcs_only:
        print(
            f"DONE RCS-only atoms={len(requested)} cached_chunks={len(rcs_cache)} "
            f"cache={rcs_cache_path}",
            flush=True,
        )
        return

    with _exclusive_checkpoint_lock(checkpoint_lock_path):
        completed = {} if args.force else _load_completed(checkpoint_path)
    pending = [atom_id for atom_id in requested if atom_id not in completed]
    print(
        f"All-paintings whole-2304 retrieval ready | atoms={len(requested)} "
        f"completed={len(completed)} pending={len(pending)} | "
        f"RCS={args.rcs_model} coordinator="
        f"{args.coordinator_backend}/{args.coordinator_model}",
        flush=True,
    )

    write_lock = threading.Lock()

    def run_one(atom_id: int) -> dict[str, Any]:
        row = retrieval[atom_id].item() if hasattr(retrieval[atom_id], "item") else retrieval[atom_id]
        return _synthesize_atom(
            atom_id,
            row["hits"],
            rcs_cache=rcs_cache,
            rcs_model=args.rcs_model,
            coordinator_backend=args.coordinator_backend,
            coordinator_model=args.coordinator_model,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_one, atom_id): atom_id for atom_id in pending}
        progress = tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Cloud atom synthesis",
            unit="atom",
            dynamic_ncols=True,
            mininterval=1.0,
        )
        for future in progress:
            atom_id = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                failure = {"atom_id": atom_id, "error": str(exc)}
                with write_lock:
                    with _exclusive_checkpoint_lock(checkpoint_lock_path):
                        with error_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                print(f"ERROR atom {atom_id}: {exc}", flush=True)
                continue
            with write_lock:
                with _exclusive_checkpoint_lock(checkpoint_lock_path):
                    latest = _load_completed(checkpoint_path)
                    if atom_id in latest:
                        record = latest[atom_id]
                        wrote_record = False
                    else:
                        with checkpoint_path.open("a", encoding="utf-8") as handle:
                            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        latest[atom_id] = record
                        wrote_record = True
                    completed = latest
                    if len(completed) % 10 == 0 or len(completed) == len(requested):
                        _write_exports(output_dir, completed)
            print(
                f"atom {atom_id:04d} | {record['confidence']:6s} | {record['label']}"
                f"{'' if wrote_record else ' | already cached by another process'}",
                flush=True,
            )

    with _exclusive_checkpoint_lock(checkpoint_lock_path):
        completed = _load_completed(checkpoint_path)
        _write_exports(output_dir, completed)
    missing = [atom_id for atom_id in requested if atom_id not in completed]
    print(
        f"DONE completed={len(completed)} requested={len(requested)} missing={len(missing)} "
        f"output={output_dir}",
        flush=True,
    )
    if missing:
        print("Missing atoms:", missing, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
