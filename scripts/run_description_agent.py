#!/usr/bin/env python3
"""Description Agent — atom-primary formal painting description (ReAct).

Encodes a query against the global dictionary, retrieves Formal Evidence Agent
atom labels, and produces a cited formal description through iterative
Description Agent ReAct revision/observation rounds.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable

import faiss
import numpy as np
import pandas as pd
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual_history_agent.paths import ensure_visualhistory_on_path  # noqa: E402

VISUALHISTORY_ROOT = ensure_visualhistory_on_path(PROJECT_ROOT)

from visual_history_agent.models.classical_ksvd import encode_samples, load_classical_ksvd_config  # noqa: E402
from VisualHistory.llm import get_client  # noqa: E402
from VisualHistory.literature.gateway_llm import (  # noqa: E402
    activate_openai_fallback,
    active_gateway_backend,
    apply_gateway_env,
)
from VisualHistory.literature.json_utils import extract_json_object  # noqa: E402
from VisualHistory.literature.literature_llm_roles import role_chat  # noqa: E402


from visual_history_agent.paths import artifact_paths as _vha_paths

_VHA = _vha_paths(PROJECT_ROOT)
DATA_ROOT = _VHA["data_root"]
DICT_ROOT = _VHA["ksvd_dir"]
INFER_ROOT = DATA_ROOT / "embeddings"

DEFAULT_ATOMS = _VHA["atom_descriptions"]
DEFAULT_OUTPUT = _VHA["outputs"] / "descriptions" / "painting_description_from_global_atoms.json"
DATABASE_PATH = _VHA["database"]
EMBEDDINGS_PATH = _VHA["embeddings"]
SPARSE_CODES_PATH = _VHA["sparse_codes"]
DICTIONARY_META_PATH = DICT_ROOT / "global_dictionary_meta.json"
DICTIONARY_PATH = _VHA["dictionary"]
DICTIONARY_CONFIG_PATH = _VHA["ksvd_config"]
EMBEDDING_MEAN_PATH = _VHA["embedding_mean"]


def _progress(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


RERANK_PROMPT = """You coordinate formal visual evidence for one painting.
The atoms below are exactly the active atoms in the painting's original non-negative
Elastic Net sparse code against the global dictionary. No artist, cohort, period, or
reference-group information was used to choose them.

Rerank every atom for synthesis. Treat its normalized coefficient as the primary measure
of importance. For close weights, prioritize concrete evidence about line, contour, form,
shape, composition, color, light, space, brushwork, texture, and surface. Do not remove
atoms or treat artists and subjects mentioned in the global atom prose as attribution.

Return JSON only:
{{
  "reranked_atom_ids": [every supplied integer atom ID exactly once],
  "rationale": ["brief explanation of the weighted ordering"]
}}

TARGET IDENTIFIER
{target}

ORIGINAL WEIGHTED GLOBAL ATOMS
{atoms}
"""


SYNTHESIS_PROMPT = """Write a grounded visual description using only the target painting's
original weighted global-dictionary atom descriptions.

Rules:
- The weighted atoms are the only evidence for visible contents and appearance.
- Preserve coefficient importance. Stronger atoms should govern the account; weaker atoms
  may only refine it.
- Every atom listed under HIGH-WEIGHT COVERAGE REQUIREMENT must be substantively represented
  and cited at least once. Do not satisfy this by attaching an unsupported citation: state the
  distinct or complementary formal tendency contributed by that atom.
- Distribute attention across the strongest supported tendencies instead of repeatedly
  paraphrasing the dominant atom. Preserve real conflicts as qualified contrasts.
- Focus on line and contour; two- and three-dimensional form; geometric and organic shape;
  composition; color; illumination and value; spatial depth; brushwork; texture; and surface.
- Reconcile globally shared atom tendencies rather than importing every object, artist,
  genre, or technique mentioned in their descriptions.
- You have not been given a metadata neighbor, artist, date, culture, period, catalogue description,
  or cohort identity. Do not infer them.
- Qualify conflicts or uncertain claims.
- End every sentence in formal_visual_description with the IDs of the atoms that support it,
  immediately before the final punctuation: for example, "Soft contours organize the group
  around a stable diagonal [Atoms: 11, 16]."
- Cite only supplied atom IDs. Choose the smallest set that directly supports each sentence;
  do not attach every atom mechanically to every sentence.

Return JSON only:
{{
  "formal_visual_description": "6-10 coherent sentences led by formal painterly qualities",
  "formal_elements": {{
    "line_and_contour": "concise account",
    "form_and_shape": "concise account",
    "composition_and_space": "concise account",
    "color_and_light": "concise account",
    "brushwork_texture_and_surface": "concise account"
  }},
  "uncertainties": ["specific limitations"],
  "atom_contributions": [
    {{"atom_id": 0, "importance_percent": 0.0, "contribution": "concise contribution"}}
  ]
}}

TARGET IDENTIFIER
{target}

AZURE-RERANKED ORIGINAL GLOBAL ATOMS
{atoms}

HIGH-WEIGHT COVERAGE REQUIREMENT
{required_high_weight_atoms}
"""


VISUAL_OBSERVATION_PROMPT = """Inspect only the supplied painting image. Do not use or infer from
its filename, directory, title, attribution, metadata, atoms, or nearest neighbors. Report direct,
conservative visual observations that can constrain later subject hypotheses.

Separate living or narratively active figures in the principal scene from painted sculptures,
reliefs, pictures-within-the-picture, and ambiguous human-like forms. Count only clearly visible
figures; record uncertainty rather than inventing occluded figures. Describe age class, position,
pose, and visible relationships without naming identities unless an iconographic attribute is
unambiguous. A metadata hypothesis that requires an additional figure is incompatible when that
figure is not visibly present.

Inspect the visible paint surface and condition as carefully as the image permits. Record visible
craquelure or cracking by pattern, scale, and distribution; abrasion, paint loss, cupping, raised
edges, retouching, discolored varnish, uneven gloss, support texture, panel joins, canvas weave,
or other material clues. Distinguish genuine-looking paint-film phenomena from compression,
sharpening, glare, reflections, printed texture, and photographic artifacts. Craquelure can support
an inference of an aged paint film, but it is not by itself proof of date, authenticity, originality,
or copy direction; artificial aging, restoration, drying cracks, and support movement remain
possible. If the reproduction is not detailed enough, state that surface condition is unassessable.

Return JSON only:
{{
  "principal_scene_figures": {{
    "count": 0,
    "figures": [
      {{
        "figure_id": "figure_1",
        "age_class": "infant|child|adult|elderly|uncertain",
        "position": "concise position",
        "pose_or_action": "only what is visible"
      }}
    ],
    "relationships": ["visible spatial or gestural relationships"]
  }},
  "secondary_human_images": [
    {{
      "kind": "sculpture|relief|picture|ambiguous",
      "count": 0,
      "location": "concise location"
    }}
  ],
  "setting_and_objects": ["conservative visible observations"],
  "formal_observations": ["line, shape, composition, color, light, space, and surface"],
  "surface_condition": {{
    "craquelure": {{
      "visibility": "clearly visible|possibly visible|not visible|unassessable",
      "pattern_scale_distribution": "direct description only",
      "confidence": "low|medium|high"
    }},
    "paint_and_support_observations": ["abrasion, loss, varnish, retouching, texture, joins, weave"],
    "possible_age_indicators": ["visible clues consistent with material aging"],
    "possible_restoration_or_imaging_artifacts": ["alternative explanations"],
    "condition_caution": "what cannot be concluded from this reproduction"
  }},
  "subject_constraints": {{
    "compatible_broad_subjects": ["broad classes supported by visible evidence"],
    "incompatible_subject_features": ["features requiring visibly absent figures or objects"],
    "constraint_summary": "concise binding constraint"
  }},
  "uncertainties": ["genuine visual ambiguities"]
}}
"""


ATOM_NEIGHBOR_VISION_PROMPT = """Inspect the supplied atom-cosine neighbor images as a blind visual
set. They were retrieved because their full uncapped weighted atom vectors resemble the query
painting's atom vector. You are not given their titles, artists, dates, cultures, or catalogue text.

Use the images only as supplementary evidence for what the atom-space retrieval may be responding
to. Identify recurring scene types, figure arrangements, settings, compositional structures,
lighting, color organization, and broad handling shared across multiple neighbors. Separate stable
recurrences from features found in only one image. Do not infer attribution, date, priority,
authenticity, or a copy relationship, and do not transfer a depicted object or narrative to the
query merely because it appears in a neighbor.

Return JSON only:
{
  "scene_type_consensus": "recurring broad scene type or indeterminate",
  "recurring_visual_structure": ["feature visible across multiple neighbors"],
  "neighbor_specific_features": [
    {"rank": 1, "scene_type": "...", "figures_and_setting": "...", "distinctive_structure": "..."}
  ],
  "conflicts_across_neighbors": ["material disagreement within the retrieved set"],
  "relevance_to_atom_retrieval": "what common visual organization likely drove atom similarity",
  "transfer_guardrail": "what must not be transferred from these neighbors to the query",
  "confidence": "low|medium|high"
}
"""


CONTEXT_PROMPT = """Form provisional subject, period, date, and attribution-context hypotheses by
testing a five-neighbor metadata neighborhood against direct visual observations and the
atom-generated formal description.

Evidence hierarchy:
1. Weighted atom evidence remains primary for formal interpretation. Direct vision is a secondary,
   conservative check, not a source that may replace the atom-generated account.
2. Only high-confidence vision observations may constrain gross figure count, age class,
   relationships, major objects, and setting. Treat ambiguous visibility or apparent absence as
   uncertainty rather than a veto. Sculptures and pictures-within-the-picture are not principal actors.
3. Consider all five neighbors jointly. Average their evidence conceptually by identifying recurring
   subject, date, culture, region, and period patterns, while giving greater weight to higher-ranked,
   more similar neighbors. Do not simply adopt the first neighbor's title or assessment.
4. Neighbor titles propose broad subject classes only. Preserve recurring broad classes unless their
   required features are clearly and confidently contradicted by direct observation.
5. If a specific neighbor detail is contradicted, reject only that detail and retain the closest broader
   class supported jointly by atoms, the five-neighbor consensus, and the image.
6. Derive period and date hypotheses from the neighborhood's aggregate metadata pattern. Summarize
   recurring artist or workshop labels only as provisional attribution context. Neither an individual
   match nor the neighborhood proves attribution, authorship, or chronology.
7. Neighbor catalogue descriptions were never loaded.
8. The blind vision summary of atom-cosine neighbors is supplementary scene-type evidence. Use
   recurring structures across that set to interpret what atom retrieval responds to, but do not
   transfer isolated objects or narratives to the query and do not use it for attribution.

Return JSON only:
{{
  "subject_hypothesis": "concise visually compatible likely subject class",
  "subject_confidence": "low|medium|high",
  "l2_subject_as_proposed": "consensus subject class proposed across the five neighbor titles",
  "visual_compatibility": "compatible|partially_compatible|incompatible",
  "rejected_l2_details": ["metadata-derived details contradicted by direct observation"],
  "period_hypothesis": "concise likely culture and period",
  "period_confidence": "low|medium|high",
  "date_hypothesis": "provisional date or date range supported across the neighborhood",
  "date_confidence": "low|medium|high",
  "attribution_context": "recurring artist, workshop, school, or manner context without asserting authorship",
  "attribution_confidence": "low|medium|high",
  "compatibility_observation": "how visual facts, atom evidence, and the five-neighbor metadata pattern agree or conflict",
  "neighbor_consensus_summary": "recurring and conflicting patterns across all five neighbors",
  "limitations": ["binding limitations on subject, period, and attribution claims"]
}}

DIRECT VISUAL OBSERVATIONS
{visual_observations}

INITIAL ATOM-GENERATED FORMAL DESCRIPTION
{formal_description}

FIVE-NEIGHBOR METADATA CONTEXT
{metadata}

BLIND VISION OF ATOM-COSINE NEIGHBORS
{atom_neighbor_vision}
"""


REACT_ACTION_PROMPT = """You are the Description Agent. Perform one revision action in an iterative ReAct-style formal-analysis
loop. Revise the current draft using direct visual observations, weighted atom evidence, and
provisional subject and period hypotheses. Follow the previous observation's instructions.

Evidence hierarchy:
1. Weighted atoms are the primary evidence and must govern the description's structure, emphasis,
   formal vocabulary, and at least three quarters of its sentences.
   Every atom in HIGH-WEIGHT COVERAGE REQUIREMENT must make a substantive, accurately cited
   contribution somewhere in the revised description. Preserve diversity among their formal
   tendencies rather than allowing the dominant atom to absorb the whole account.
2. Direct vision is secondary and constraint-only. Use it to correct only clear, high-confidence
   contradictions in gross visible content; do not let it replace, reinterpret wholesale, or
   dominate the atom-derived formal account.
3. Use the visually tested subject hypothesis, not the raw neighbor title, to name a broad subject
   provisionally.
4. The period hypothesis may calibrate vocabulary for composition, modeling, color, and space.
5. Limitations are binding: do not claim attribution or invent fine iconography.

The revised text must be natural and self-contained. Do not discuss atoms, embeddings, neighbors,
metadata, prompts, or the analytic process in the prose, but preserve compact inline provenance
tags immediately before each sentence's final punctuation:
- use [Atoms: 11, 16] for formal claims supported by those supplied atoms;
- use [Vision] for content or condition directly verified from the image;
- use [L2] for a contextual subject or period hypothesis;
- use multiple tags when a sentence combines evidence, e.g. [Atoms: 11] [Vision] [L2].
Every sentence must have at least one source tag, and every atom ID must be supplied in WEIGHTED
ATOM EVIDENCE. Use the smallest directly relevant atom set rather than citing every atom.
At least 75% of sentences must cite one or more atoms. No more than one third of sentences may use
[Vision], normally one sentence for gross content and at most one sentence for condition or a clear
constraint. Do not add [Vision] to atom-supported formal sentences merely because the image was seen.
The text must include one qualified sentence naming the subject hypothesis and one sentence using
the period hypothesis to frame formal qualities.

Return JSON only:
{{
  "action_summary": "brief public summary of the revision action, not hidden reasoning",
  "revised_formal_description": "6-10 coherent formal sentences",
  "hypothesis_usage": "how subject and period hypotheses were cautiously incorporated"
}}

ITERATION
{iteration}

CURRENT DRAFT
{current_draft}

WEIGHTED ATOM EVIDENCE
{atoms}

HIGH-WEIGHT COVERAGE REQUIREMENT
{required_high_weight_atoms}

DIRECT VISUAL OBSERVATIONS
{visual_observations}

VISUALLY TESTED SUBJECT AND PERIOD HYPOTHESES
{hypotheses}

PREVIOUS OBSERVATION
{observation}
"""


REACT_OBSERVATION_PROMPT = """You are the Description Agent. Evaluate one candidate in an iterative ReAct-style formal-analysis
loop. Return a concise evidence audit, not private chain-of-thought.

Check that:
- the weighted atoms, not direct vision, control the organization and majority of the description;
- only clear high-confidence visual contradictions constrain stated figures, relationships, or objects;
- no principal actor is inferred from a sculpture, relief, or picture-within-the-picture;
- rejected neighbor details do not reappear in the candidate;
- the visually tested subject hypothesis is expressed as a qualified identification;
- the period hypothesis grounds formal vocabulary without replacing weighted formal evidence;
- line, form, shape, composition, color, light, space, brushwork, texture, and surface remain
  central;
- no attribution claim or unsupported fine iconography is introduced;
- stronger atoms have visibly greater influence than weaker atoms.
- every atom in HIGH-WEIGHT COVERAGE REQUIREMENT is substantively discussed and accurately cited;
- the candidate reflects distinct formal tendencies across the high-weight atoms rather than
  repeatedly restating the dominant atom;
- every sentence ends with at least one inline [Atoms: ...], [Vision], or [L2] source tag;
- every cited atom ID exists in the supplied weighted evidence and genuinely supports that claim;
- formal painterly claims retain atom citations rather than being supported only by [Vision] or [L2].
- at least 75% of sentences cite atoms and no more than one third cite [Vision].

Return JSON only:
{{
  "ready": true,
  "supported_claims": ["concise supported claims"],
  "unsupported_or_overstated_claims": ["claims to remove or qualify"],
  "revision_instructions": ["specific instructions for the next action"]
}}

Set "ready" to true only when unsupported_or_overstated_claims is empty, citation coverage and
atom IDs are valid, and no substantive revision remains. Otherwise set it to false.

ITERATION
{iteration}

CANDIDATE DESCRIPTION
{candidate}

WEIGHTED ATOM EVIDENCE
{atoms}

HIGH-WEIGHT COVERAGE REQUIREMENT
{required_high_weight_atoms}

DIRECT VISUAL OBSERVATIONS
{visual_observations}

VISUALLY TESTED SUBJECT AND PERIOD HYPOTHESES
{hypotheses}
"""


def _safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).replace("\x00", " ").strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def _resolve_target_image(
    explicit_path: Path | None,
    target_info: dict[str, Any],
) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path.expanduser())
    image_value = _safe_text(target_info.get("image"))
    if image_value:
        image_path = Path(image_value).expanduser()
        configured_images_root = (os.getenv("PAINTING_IMAGES_DIR") or "").strip()
        candidates.extend(
            [
                image_path,
                *(
                    [Path(configured_images_root).expanduser() / image_path.name]
                    if configured_images_root
                    else []
                ),
                DATA_ROOT / "images" / image_path.name,
                Path(
                    "/Volumes/Untitled/Painting_Interpretability_Study/Stable/"
                    "painting_data/images"
                )
                / image_path.name,
            ]
        )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "A readable target image is required for direct visual verification; "
        "provide --query-image. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _column(frame: pd.DataFrame, names: list[str]) -> str:
    normalized = {" ".join(str(c).strip().lower().split()): str(c) for c in frame.columns}
    for name in names:
        key = " ".join(name.strip().lower().split())
        if key in normalized:
            return normalized[key]
    raise KeyError(f"None of {names!r} found")


def _resolve_target(database: pd.DataFrame, row: int | None, title: str | None) -> int:
    if row is not None:
        if not 0 <= row < len(database):
            raise ValueError(f"database row must be in [0, {len(database) - 1}]")
        return int(row)
    title_col = _column(database, ["painting_name", "painting name", "title"])
    matches = np.flatnonzero(
        database[title_col].map(_safe_text).str.casefold().eq(str(title).strip().casefold())
    )
    if matches.size != 1:
        raise ValueError(f"Expected one exact title match for {title!r}, found {matches.size}")
    return int(matches[0])


def _load_atom_descriptions(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Expected a non-empty JSON array in {path}")
    return {int(item["atom_id"]): item for item in payload}


def _load_canonical_task2_weights(
    database_row: int,
    valid: np.ndarray,
    descriptions: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not bool(valid[database_row]):
        raise ValueError(f"Target row {database_row} has no valid Task 2 embedding")
    codes = np.load(SPARSE_CODES_PATH, mmap_mode="r")
    if codes.shape[0] == valid.shape[0]:
        code_row = database_row
    elif codes.shape[0] == int(valid.sum()):
        code_row = int(np.count_nonzero(valid[:database_row]))
    else:
        raise ValueError(
            f"Task 2 sparse-code rows {codes.shape[0]} do not match "
            f"all rows {valid.shape[0]} or valid rows {int(valid.sum())}"
        )
    coefficients = np.asarray(codes[code_row], dtype=np.float64)
    active_ids = np.flatnonzero(np.isfinite(coefficients) & (coefficients > 0))
    if active_ids.size == 0:
        raise ValueError("Canonical Task 2 sparse code has no positive atoms")
    order = active_ids[np.argsort(-coefficients[active_ids], kind="stable")]
    missing = [int(atom_id) for atom_id in order if int(atom_id) not in descriptions]
    if missing:
        raise ValueError(
            f"First-pass JSON lacks canonical Task 2 atom descriptions {missing}"
        )
    total = float(coefficients[order].sum())
    return [
        {
            "atom_id": int(atom_id),
            "coefficient": float(coefficients[atom_id]),
            "importance_percent": 100.0 * float(coefficients[atom_id]) / total,
            "label": str(descriptions[int(atom_id)].get("label", "")).strip(),
            "painterly_description": str(
                descriptions[int(atom_id)].get("painterly_description", "")
            ).strip(),
            "confidence": str(
                descriptions[int(atom_id)].get("confidence", "")
            ).strip(),
        }
        for atom_id in order
    ]


def _encode_external_task2_weights(
    query_embedding: np.ndarray,
    descriptions: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if query.shape != (2304,) or not np.isfinite(query).all():
        raise ValueError(f"Expected one finite 2304-D query embedding, got {query.shape}")
    dictionary = np.asarray(np.load(DICTIONARY_PATH), dtype=np.float32)
    center = np.asarray(np.load(EMBEDDING_MEAN_PATH), dtype=np.float32)
    config = load_classical_ksvd_config(DICTIONARY_CONFIG_PATH)
    coefficients = encode_samples(
        (query[None, :] - center[None, :]).astype(np.float32, copy=False),
        dictionary,
        config,
    )[0].astype(np.float64, copy=False)
    active_ids = np.flatnonzero(np.isfinite(coefficients) & (coefficients > 0))
    if active_ids.size == 0:
        raise ValueError("External query received no positive Task 2 coefficients")
    order = active_ids[np.argsort(-coefficients[active_ids], kind="stable")]
    missing = [int(atom_id) for atom_id in order if int(atom_id) not in descriptions]
    if missing:
        raise ValueError(
            "First-pass JSON lacks external query Task 2 atom descriptions "
            f"{missing}. Generate those descriptions, then rerun."
        )
    total = float(coefficients[order].sum())
    return [
        {
            "atom_id": int(atom_id),
            "coefficient": float(coefficients[atom_id]),
            "importance_percent": 100.0 * float(coefficients[atom_id]) / total,
            "label": str(descriptions[int(atom_id)].get("label", "")).strip(),
            "painterly_description": str(
                descriptions[int(atom_id)].get("painterly_description", "")
            ).strip(),
            "confidence": str(descriptions[int(atom_id)].get("confidence", "")).strip(),
        }
        for atom_id in order
    ]


def _nearest_l2_metadata(
    database: pd.DataFrame,
    database_row: int | None,
    valid: np.ndarray,
    *,
    query_embedding: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    valid_rows = np.flatnonzero(valid)
    matrix = np.ascontiguousarray(embeddings[valid_rows], dtype=np.float32)
    finite = np.isfinite(matrix).all(axis=1)
    valid_rows = valid_rows[finite]
    matrix = np.ascontiguousarray(matrix[finite], dtype=np.float32)
    query = (
        np.ascontiguousarray(embeddings[[database_row]], dtype=np.float32)
        if query_embedding is None and database_row is not None
        else np.ascontiguousarray(
            np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        )
    )
    if not np.isfinite(query).all():
        raise ValueError("Target has a non-finite embedding")

    index = faiss.IndexFlatL2(int(matrix.shape[1]))
    index.add(matrix)
    distances, positions = index.search(query, min(6, len(valid_rows)))
    neighbor_rows: list[tuple[int, float]] = []
    for distance, position in zip(distances[0], positions[0]):
        candidate = int(valid_rows[int(position)])
        if database_row is None or candidate != database_row:
            neighbor_rows.append((candidate, float(distance)))
            if len(neighbor_rows) == 5:
                break
    if len(neighbor_rows) < min(5, len(valid_rows) - int(database_row is not None)):
        raise ValueError("FAISS returned too few non-self L2 neighbors")

    columns = {
        "painting_name": _column(database, ["painting_name", "painting name", "title"]),
        "artist": _column(database, ["artist"]),
        "date": _column(database, ["dates", "date"]),
        "culture": _column(database, ["culture"]),
        "region": _column(database, ["region"]),
        "art_historical_time_period": _column(
            database,
            ["art_historical_time_period", "art historical time period", "time_period"],
        ),
    }
    # The description column is intentionally neither resolved nor read.
    return [
        {
            "rank": rank,
            "database_row": neighbor_row,
            "squared_l2_distance": neighbor_distance,
            "retrieval_metric": "squared_l2_distance",
            "retrieval_features": "raw_whole_2304d_pooled_embedding",
            **{
                key: _safe_text(database.iloc[neighbor_row][column])
                for key, column in columns.items()
            },
        }
        for rank, (neighbor_row, neighbor_distance) in enumerate(neighbor_rows, start=1)
    ]


def _matching_database_image_rows(
    database: pd.DataFrame,
    query_image: Path | None,
) -> set[int]:
    """Find database rows whose stored image filename matches the query image."""
    if query_image is None:
        return set()
    image_column = next(
        (
            column
            for column in database.columns
            if str(column).strip().casefold() in {"image", "image filename", "filename"}
        ),
        None,
    )
    if image_column is None:
        return set()
    query_name = query_image.name.casefold()
    return {
        position
        for position, value in enumerate(database[image_column])
        if Path(_safe_text(value)).name.casefold() == query_name
    }


def _nearest_embedding_cosine_metadata(
    database: pd.DataFrame,
    database_row: int | None,
    valid: np.ndarray,
    *,
    query_embedding: np.ndarray | None = None,
    excluded_database_rows: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve five metadata neighbors by cosine similarity of pooled embeddings."""
    embeddings = np.load(EMBEDDINGS_PATH, mmap_mode="r")
    valid_rows = np.flatnonzero(valid)
    matrix = np.ascontiguousarray(embeddings[valid_rows], dtype=np.float32)
    finite = np.isfinite(matrix).all(axis=1)
    norms = np.linalg.norm(np.where(np.isfinite(matrix), matrix, 0.0), axis=1)
    usable = finite & (norms > 0)
    valid_rows = valid_rows[usable]
    matrix = np.ascontiguousarray(matrix[usable], dtype=np.float32)
    query = (
        np.ascontiguousarray(embeddings[[database_row]], dtype=np.float32)
        if query_embedding is None and database_row is not None
        else np.ascontiguousarray(
            np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        )
    )
    if (
        query.shape != (1, matrix.shape[1])
        or not np.isfinite(query).all()
        or float(np.linalg.norm(query)) <= 0
    ):
        raise ValueError(f"Target has an invalid embedding for cosine search: {query.shape}")

    faiss.normalize_L2(matrix)
    faiss.normalize_L2(query)
    index = faiss.IndexFlatIP(int(matrix.shape[1]))
    index.add(matrix)
    # Retrieve extra candidates because a database image may be supplied as an external
    # query embedding, in which case database_row is unavailable but the identical
    # normalized embedding must still be excluded as a self-match.
    similarities, positions = index.search(query, min(10, len(valid_rows)))
    neighbor_rows: list[tuple[int, float]] = []
    excluded_rows = set(excluded_database_rows or ())
    if database_row is not None:
        excluded_rows.add(database_row)
    for similarity, position in zip(similarities[0], positions[0]):
        matrix_position = int(position)
        candidate = int(valid_rows[matrix_position])
        is_known_self = candidate in excluded_rows
        is_embedding_self = np.allclose(
            matrix[matrix_position],
            query[0],
            rtol=1e-6,
            atol=1e-7,
        )
        if is_known_self or is_embedding_self:
            continue
        neighbor_rows.append((candidate, float(similarity)))
        if len(neighbor_rows) == 5:
            break
    if len(neighbor_rows) < 5:
        raise ValueError("FAISS returned too few non-self embedding-cosine neighbors")

    columns = {
        "painting_name": _column(database, ["painting_name", "painting name", "title"]),
        "artist": _column(database, ["artist"]),
        "date": _column(database, ["dates", "date"]),
        "culture": _column(database, ["culture"]),
        "region": _column(database, ["region"]),
        "art_historical_time_period": _column(
            database,
            ["art_historical_time_period", "art historical time period", "time_period"],
        ),
    }
    # The description column is intentionally neither resolved nor read.
    return [
        {
            "rank": rank,
            "database_row": neighbor_row,
            "cosine_similarity": neighbor_similarity,
            "retrieval_metric": "cosine_similarity",
            "retrieval_features": "raw_whole_2304d_pooled_embedding",
            **{
                key: _safe_text(database.iloc[neighbor_row][column])
                for key, column in columns.items()
            },
        }
        for rank, (neighbor_row, neighbor_similarity) in enumerate(
            neighbor_rows, start=1
        )
    ]


def _nearest_atom_cosine_metadata(
    database: pd.DataFrame,
    database_row: int | None,
    valid: np.ndarray,
    weighted_atoms: list[dict[str, Any]],
    *,
    excluded_database_rows: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve five metadata neighbors by cosine similarity of full atom codes."""
    codes = np.load(SPARSE_CODES_PATH, mmap_mode="r")
    valid_rows = np.flatnonzero(valid)
    if codes.shape[0] == valid.shape[0]:
        matrix = np.asarray(codes[valid_rows], dtype=np.float32)
    elif codes.shape[0] == int(valid.sum()):
        matrix = np.asarray(codes, dtype=np.float32)
    else:
        raise ValueError(
            f"Task 2 sparse-code rows {codes.shape[0]} do not match "
            f"all rows {valid.shape[0]} or valid rows {int(valid.sum())}"
        )

    finite = np.isfinite(matrix).all(axis=1)
    norms = np.linalg.norm(np.where(np.isfinite(matrix), matrix, 0.0), axis=1)
    usable = finite & (norms > 0)
    valid_rows = valid_rows[usable]
    matrix = np.ascontiguousarray(matrix[usable], dtype=np.float32)

    query = np.zeros((1, int(codes.shape[1])), dtype=np.float32)
    for atom in weighted_atoms:
        atom_id = int(atom["atom_id"])
        if not 0 <= atom_id < query.shape[1]:
            raise ValueError(f"Atom ID {atom_id} is outside sparse-code width {query.shape[1]}")
        query[0, atom_id] = float(atom["coefficient"])
    if not np.isfinite(query).all() or float(np.linalg.norm(query)) <= 0:
        raise ValueError("Target has no finite, positive-norm weighted atom code")

    faiss.normalize_L2(matrix)
    faiss.normalize_L2(query)
    index = faiss.IndexFlatIP(int(matrix.shape[1]))
    index.add(matrix)
    similarities, positions = index.search(query, min(10, len(valid_rows)))
    neighbor_rows: list[tuple[int, float]] = []
    excluded_rows = set(excluded_database_rows or ())
    if database_row is not None:
        excluded_rows.add(database_row)
    for similarity, position in zip(similarities[0], positions[0]):
        candidate = int(valid_rows[int(position)])
        if candidate in excluded_rows:
            continue
        neighbor_rows.append((candidate, float(similarity)))
        if len(neighbor_rows) == 5:
            break
    if len(neighbor_rows) < 5:
        raise ValueError("FAISS returned too few non-self atom-cosine neighbors")

    columns = {
        "image": _column(database, ["Image", "image"]),
        "painting_name": _column(database, ["painting_name", "painting name", "title"]),
        "artist": _column(database, ["artist"]),
        "date": _column(database, ["dates", "date"]),
        "culture": _column(database, ["culture"]),
        "region": _column(database, ["region"]),
        "art_historical_time_period": _column(
            database,
            ["art_historical_time_period", "art historical time period", "time_period"],
        ),
    }
    # The description column is intentionally neither resolved nor read.
    return [
        {
            "rank": rank,
            "database_row": neighbor_row,
            "cosine_similarity": neighbor_similarity,
            "retrieval_metric": "cosine_similarity",
            "retrieval_features": "full_uncapped_task2_atom_coefficient_vector",
            "query_active_atom_count": len(weighted_atoms),
            **{
                key: _safe_text(database.iloc[neighbor_row][column])
                for key, column in columns.items()
            },
        }
        for rank, (neighbor_row, neighbor_similarity) in enumerate(
            neighbor_rows, start=1
        )
    ]


def _metadata_neighborhood(neighbors: list[dict[str, Any]]) -> dict[str, Any]:
    if not neighbors:
        raise ValueError("Metadata neighborhood is empty")
    metric = str(neighbors[0]["retrieval_metric"])
    value_key = (
        "cosine_similarity" if metric == "cosine_similarity" else "squared_l2_distance"
    )
    values = [float(neighbor[value_key]) for neighbor in neighbors]
    return {
        "neighbor_count": len(neighbors),
        "retrieval_metric": metric,
        "retrieval_features": neighbors[0]["retrieval_features"],
        f"mean_{value_key}": float(np.mean(values)),
        "aggregation_policy": (
            "Treat all five neighbors jointly. Infer subject and period from recurring "
            "metadata patterns, weighted by rank and similarity; do not copy the top "
            "neighbor's assessment."
        ),
        "neighbors": neighbors,
    }


def _format_atoms(atoms: list[dict[str, Any]]) -> str:
    return "\n\n".join(
        (
            f"ATOM {atom['atom_id']} | coefficient={atom['coefficient']:.8g} | "
            f"importance={atom['importance_percent']:.3f}% | "
            f"confidence={atom['confidence']}\n"
            f"LABEL: {atom['label']}\n"
            f"PAINTERLY EVIDENCE: {atom['painterly_description']}"
        )
        for atom in atoms
    )


def _required_high_weight_atoms(
    atoms: list[dict[str, Any]],
    *,
    cumulative_mass_percent: float = 85.0,
    minimum_atoms: int = 5,
    maximum_atoms: int = 10,
) -> list[dict[str, Any]]:
    """Select the strongest atoms requiring substantive coverage in generated prose."""
    ordered = sorted(
        atoms,
        key=lambda atom: (
            -float(atom["importance_percent"]),
            int(atom["atom_id"]),
        ),
    )
    required: list[dict[str, Any]] = []
    cumulative = 0.0
    for atom in ordered:
        if len(required) >= maximum_atoms:
            break
        required.append(atom)
        cumulative += float(atom["importance_percent"])
        if len(required) >= minimum_atoms and cumulative >= cumulative_mass_percent:
            break
    return required


def _format_required_high_weight_atoms(atoms: list[dict[str, Any]]) -> str:
    return "\n".join(
        (
            f"- Atom {int(atom['atom_id'])}: "
            f"{float(atom['importance_percent']):.3f}% — {atom['label']}"
        )
        for atom in atoms
    )


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'“‘]?[A-Z])")
_ATOM_CITATION_RE = re.compile(r"\[Atoms?:\s*(\d+(?:\s*,\s*\d+)*)\]")
_SOURCE_CITATION_RE = re.compile(r"\[(?:Atoms?:\s*\d+(?:\s*,\s*\d+)*|Vision|L2)\]")
_ENDING_CITATIONS_RE = re.compile(
    r"(?:\[(?:Atoms?:\s*\d+(?:\s*,\s*\d+)*|Vision|L2)\]\s*)+[.!?][\"'”’]?$"
)
_POST_PUNCTUATION_CITATIONS_RE = re.compile(
    r"([.!?])\s*((?:\[(?:Atoms?:\s*\d+(?:\s*,\s*\d+)*|Vision|L2)\]\s*)+)"
)


def _normalize_citation_placement(text: str) -> str:
    """Move post-punctuation source tags before punctuation and normalize spacing."""

    def replace(match: re.Match[str]) -> str:
        punctuation = match.group(1)
        tags = _SOURCE_CITATION_RE.findall(match.group(2))
        return f" {' '.join(tags)}{punctuation} "

    normalized = _POST_PUNCTUATION_CITATIONS_RE.sub(replace, text.strip())
    return re.sub(r"[ \t]{2,}", " ", normalized).strip()


def _validate_description_citations(
    text: str,
    atom_ids: set[int],
    *,
    require_atom_each_sentence: bool,
    min_atom_sentence_fraction: float = 0.0,
    max_vision_sentence_fraction: float = 1.0,
    required_atom_ids: set[int] | None = None,
) -> str:
    text = _normalize_citation_placement(text)
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY_RE.split(text.strip())
        if sentence.strip()
    ]
    if not 6 <= len(sentences) <= 10:
        raise ValueError(f"Description must contain 6-10 sentences; found {len(sentences)}")
    errors: list[str] = []
    atom_sentence_count = 0
    vision_sentence_count = 0
    all_cited_atom_ids: set[int] = set()
    for index, sentence in enumerate(sentences, start=1):
        atom_matches = _ATOM_CITATION_RE.findall(sentence)
        cited_ids = {
            int(value.strip())
            for match in atom_matches
            for value in match.split(",")
            if value.strip()
        }
        unknown = sorted(cited_ids - atom_ids)
        if unknown:
            errors.append(f"sentence {index} cites unknown atoms {unknown}")
        if require_atom_each_sentence and not cited_ids:
            errors.append(f"sentence {index} lacks an atom citation")
        if cited_ids:
            atom_sentence_count += 1
            all_cited_atom_ids.update(cited_ids)
        if "[Vision]" in sentence:
            vision_sentence_count += 1
        if not _SOURCE_CITATION_RE.search(sentence):
            errors.append(f"sentence {index} lacks a source citation")
        if not _ENDING_CITATIONS_RE.search(sentence):
            errors.append(f"sentence {index} does not end with citation tag(s)")
    atom_fraction = atom_sentence_count / len(sentences)
    vision_fraction = vision_sentence_count / len(sentences)
    if atom_fraction < min_atom_sentence_fraction:
        errors.append(
            f"atom citations cover {atom_sentence_count}/{len(sentences)} sentences; "
            f"minimum fraction is {min_atom_sentence_fraction:.0%}"
        )
    if vision_fraction > max_vision_sentence_fraction:
        errors.append(
            f"vision citations cover {vision_sentence_count}/{len(sentences)} sentences; "
            f"maximum fraction is {max_vision_sentence_fraction:.0%}"
        )
    missing_required = sorted((required_atom_ids or set()) - all_cited_atom_ids)
    if missing_required:
        errors.append(
            "description omits required high-weight atoms "
            f"{missing_required}; each must be substantively represented and cited"
        )
    if errors:
        raise ValueError("; ".join(errors))
    return text


def _build_citation_map(
    text: str,
    atoms: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    atom_by_id = {int(atom["atom_id"]): atom for atom in atoms}
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY_RE.split(text.strip())
        if sentence.strip()
    ]
    result: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences, start=1):
        cited_ids = sorted(
            {
                int(value.strip())
                for match in _ATOM_CITATION_RE.findall(sentence)
                for value in match.split(",")
                if value.strip()
            }
        )
        result.append(
            {
                "sentence_number": index,
                "sentence": sentence,
                "atom_ids": cited_ids,
                "atom_labels": [
                    atom_by_id[atom_id]["label"]
                    for atom_id in cited_ids
                    if atom_id in atom_by_id
                ],
                "uses_direct_vision": "[Vision]" in sentence,
                "uses_l2_context": "[L2]" in sentence,
            }
        )
    return result


def _coordinator_json(
    prompt: str,
    stage: str,
    *,
    validator: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(5):
        raw = ""
        started = time.perf_counter()
        backend = active_gateway_backend()
        _progress(
            f"START {stage} | attempt={attempt + 1}/5 backend={backend} "
            f"prompt_chars={len(prompt):,}"
        )
        try:
            raw = role_chat(
                "answer",
                prompt,
                max_tokens=16384,
                temperature=0,
                log=lambda message: _progress(f"{stage} API | {message.strip()}"),
            )
            payload = extract_json_object(raw)
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            if validator is not None:
                validator(payload)
            _progress(
                f"DONE  {stage} | backend={active_gateway_backend()} "
                f"elapsed={time.perf_counter() - started:.1f}s "
                f"response_chars={len(raw):,} keys={sorted(payload)}"
            )
            return payload
        except Exception as exc:
            last_error = exc
            delay = min(60.0, 3.0 * (2.0**attempt)) if attempt < 4 else 0.0
            _progress(
                f"FAIL  {stage} | attempt={attempt + 1}/5 backend={backend} "
                f"elapsed={time.perf_counter() - started:.1f}s error={exc} | "
                f"response={raw.replace(chr(10), ' ')[:500]!r}"
            )
            if attempt < 4:
                _progress(f"RETRY {stage} in {delay:.1f}s")
                time.sleep(delay)
    raise RuntimeError(f"{stage} failed after retries: {last_error}")


def _coordinator_vision_json(
    prompt: str,
    image_path: Path,
    stage: str,
    *,
    backend: str,
    model: str,
) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    image_url = f"data:{mime_type};base64,{encoded}"
    last_error: Exception | None = None
    for attempt in range(5):
        raw = ""
        started = time.perf_counter()
        active_backend = (
            active_gateway_backend()
            if backend == "azure-openai-fallback"
            else backend
        )
        _progress(
            f"START {stage} | attempt={attempt + 1}/5 backend={active_backend} "
            f"image={image_path.name} encoded_chars={len(encoded):,}"
        )
        if active_backend == "azure":
            client = get_client()
        elif active_backend == "openai":
            api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
            if not api_key:
                raise RuntimeError(
                    "Direct OpenAI visual verification requires OPENAI_API_KEY in .env"
                )
            client = OpenAI(api_key=api_key)
        else:
            raise ValueError(
                "Direct visual verification supports Azure, OpenAI, or Azure→OpenAI fallback"
            )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url, "detail": "high"},
                            },
                        ],
                    }
                ],
                max_completion_tokens=8192,
                reasoning_effort="low",
            )
            raw = response.choices[0].message.content or ""
            payload = extract_json_object(raw)
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            principal = payload.get("principal_scene_figures")
            if not isinstance(principal, dict) or not isinstance(
                principal.get("count"), int
            ):
                raise ValueError(
                    "Visual observation must include integer principal_scene_figures.count"
                )
            _progress(
                f"DONE  {stage} | backend={active_backend} "
                f"elapsed={time.perf_counter() - started:.1f}s "
                f"response_chars={len(raw):,} "
                f"principal_figures={principal['count']}"
            )
            return payload
        except Exception as exc:
            last_error = exc
            if backend == "azure-openai-fallback" and active_backend == "azure":
                activate_openai_fallback()
                _progress(
                    f"FAILOVER {stage} | Azure failed after "
                    f"{time.perf_counter() - started:.1f}s: {exc}; "
                    "switching permanently to direct OpenAI"
                )
                continue
            delay = min(60.0, 3.0 * (2.0**attempt)) if attempt < 4 else 0.0
            _progress(
                f"FAIL  {stage} | attempt={attempt + 1}/5 backend={active_backend} "
                f"elapsed={time.perf_counter() - started:.1f}s error={exc} | "
                f"response={raw.replace(chr(10), ' ')[:500]!r}"
            )
            if attempt < 4:
                _progress(f"RETRY {stage} in {delay:.1f}s")
                time.sleep(delay)
    raise RuntimeError(f"{stage} failed after retries: {last_error}")


def _resolve_atom_neighbor_images(
    neighbors: list[dict[str, Any]],
) -> tuple[list[tuple[int, Path]], list[dict[str, Any]]]:
    resolved: list[tuple[int, Path]] = []
    unavailable: list[dict[str, Any]] = []
    for neighbor in neighbors:
        rank = int(neighbor["rank"])
        try:
            image_path = _resolve_target_image(
                None,
                {"image": neighbor.get("image", "")},
            )
        except FileNotFoundError:
            unavailable.append(
                {
                    "rank": rank,
                    "database_row": int(neighbor["database_row"]),
                    "image": neighbor.get("image", ""),
                }
            )
            continue
        resolved.append((rank, image_path))
    return resolved, unavailable


def _coordinator_multi_vision_json(
    prompt: str,
    ranked_images: list[tuple[int, Path]],
    stage: str,
    *,
    backend: str,
    model: str,
) -> dict[str, Any]:
    if not ranked_images:
        raise ValueError("At least one atom-neighbor image is required")
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    encoded_chars = 0
    for rank, image_path in ranked_images:
        mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        encoded_chars += len(encoded)
        content.extend(
            [
                {"type": "text", "text": f"ATOM-COSINE NEIGHBOR RANK {rank}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{encoded}",
                        "detail": "high",
                    },
                },
            ]
        )

    last_error: Exception | None = None
    for attempt in range(5):
        raw = ""
        started = time.perf_counter()
        active_backend = (
            active_gateway_backend()
            if backend == "azure-openai-fallback"
            else backend
        )
        _progress(
            f"START {stage} | attempt={attempt + 1}/5 backend={active_backend} "
            f"images={len(ranked_images)} encoded_chars={encoded_chars:,}"
        )
        if active_backend == "azure":
            client = get_client()
        elif active_backend == "openai":
            api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
            if not api_key:
                raise RuntimeError(
                    "Direct OpenAI atom-neighbor vision requires OPENAI_API_KEY in .env"
                )
            client = OpenAI(api_key=api_key)
        else:
            raise ValueError(
                "Atom-neighbor vision supports Azure, OpenAI, or Azure→OpenAI fallback"
            )
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_completion_tokens=8192,
                reasoning_effort="low",
            )
            raw = response.choices[0].message.content or ""
            payload = extract_json_object(raw)
            if not isinstance(payload, dict):
                raise ValueError("Expected a JSON object")
            if not str(payload.get("scene_type_consensus", "")).strip():
                raise ValueError("Atom-neighbor vision omitted scene_type_consensus")
            _progress(
                f"DONE  {stage} | backend={active_backend} "
                f"elapsed={time.perf_counter() - started:.1f}s "
                f"response_chars={len(raw):,}"
            )
            return payload
        except Exception as exc:
            last_error = exc
            if backend == "azure-openai-fallback" and active_backend == "azure":
                activate_openai_fallback()
                _progress(
                    f"FAILOVER {stage} | Azure failed after "
                    f"{time.perf_counter() - started:.1f}s: {exc}; "
                    "switching permanently to direct OpenAI"
                )
                continue
            delay = min(60.0, 3.0 * (2.0**attempt)) if attempt < 4 else 0.0
            _progress(
                f"FAIL  {stage} | attempt={attempt + 1}/5 backend={active_backend} "
                f"elapsed={time.perf_counter() - started:.1f}s error={exc} | "
                f"response={raw.replace(chr(10), ' ')[:500]!r}"
            )
            if attempt < 4:
                _progress(f"RETRY {stage} in {delay:.1f}s")
                time.sleep(delay)
    raise RuntimeError(f"{stage} failed after retries: {last_error}")


def _validate_rerank(payload: dict[str, Any], atoms: list[dict[str, Any]]) -> list[int]:
    """Accept a partial LLM ranking; keep unique valid IDs, append any missing in original order."""
    raw = payload.get("reranked_atom_ids")
    expected = [int(atom["atom_id"]) for atom in atoms]
    expected_set = set(expected)
    if not isinstance(raw, list):
        raise ValueError("reranked_atom_ids must be a JSON array")
    ranked: list[int] = []
    seen: set[int] = set()
    for value in raw:
        atom_id = int(value)
        if atom_id not in expected_set or atom_id in seen:
            continue
        ranked.append(atom_id)
        seen.add(atom_id)
    missing = [atom_id for atom_id in expected if atom_id not in seen]
    if missing:
        _progress(
            f"Rerank repaired | kept={len(ranked)} appended_missing={len(missing)}"
        )
        ranked.extend(missing)
    return ranked


def _iterative_react_rewrite(
    initial_description: str,
    atoms: list[dict[str, Any]],
    hypotheses: dict[str, Any],
    visual_observations: dict[str, Any],
    *,
    max_rounds: int,
) -> tuple[str, list[dict[str, Any]]]:
    draft = initial_description.strip()
    observation: dict[str, Any] = {
        "ready": False,
        "revision_instructions": [
            "Test the broad subject and period hypotheses against the weighted formal evidence."
        ],
    }
    trace: list[dict[str, Any]] = []
    atom_text = _format_atoms(atoms)
    valid_atom_ids = {int(atom["atom_id"]) for atom in atoms}
    required_high_weight_atoms = _required_high_weight_atoms(atoms)
    required_high_weight_atom_ids = {
        int(atom["atom_id"]) for atom in required_high_weight_atoms
    }
    required_high_weight_text = _format_required_high_weight_atoms(
        required_high_weight_atoms
    )
    hypothesis_text = json.dumps(hypotheses, ensure_ascii=False, indent=2)
    visual_text = json.dumps(visual_observations, ensure_ascii=False, indent=2)

    def validate_action(payload: dict[str, Any]) -> None:
        candidate = str(payload.get("revised_formal_description", "")).strip()
        if not candidate:
            raise ValueError("ReAct action omitted revised_formal_description")
        payload["revised_formal_description"] = _validate_description_citations(
            candidate,
            valid_atom_ids,
            require_atom_each_sentence=False,
            min_atom_sentence_fraction=0.75,
            max_vision_sentence_fraction=0.34,
            required_atom_ids=required_high_weight_atom_ids,
        )

    react_started = time.perf_counter()
    _progress(
        f"START Description Agent (ReAct) refinement | max_rounds={max_rounds} "
        f"atoms={len(atoms)} draft_chars={len(draft):,}"
    )
    for iteration in range(1, max_rounds + 1):
        round_started = time.perf_counter()
        _progress(f"Description Agent round {iteration}/{max_rounds} | requesting revision action")
        action = _coordinator_json(
            REACT_ACTION_PROMPT.format(
                iteration=iteration,
                current_draft=draft,
                atoms=atom_text,
                required_high_weight_atoms=required_high_weight_text,
                visual_observations=visual_text,
                hypotheses=hypothesis_text,
                observation=json.dumps(observation, ensure_ascii=False, indent=2),
            ),
            f"Description Agent action {iteration}",
            validator=validate_action,
        )
        candidate = str(action.get("revised_formal_description", "")).strip()
        if not candidate:
            raise ValueError(
                f"Description Agent action {iteration} omitted revised_formal_description"
            )
        _progress(
            f"Description Agent round {iteration}/{max_rounds} | action complete "
            f"candidate_chars={len(candidate):,}; requesting evidence check"
        )
        observation = _coordinator_json(
            REACT_OBSERVATION_PROMPT.format(
                iteration=iteration,
                candidate=candidate,
                atoms=atom_text,
                required_high_weight_atoms=required_high_weight_text,
                visual_observations=visual_text,
                hypotheses=hypothesis_text,
            ),
            f"Description Agent observation {iteration}",
        )
        trace.append(
            {
                "iteration": iteration,
                "action_summary": action.get("action_summary", ""),
                "hypothesis_usage": action.get("hypothesis_usage", ""),
                "candidate_description": candidate,
                "observation": observation,
            }
        )
        draft = candidate
        unsupported = observation.get("unsupported_or_overstated_claims", [])
        unsupported_count = len(unsupported) if isinstance(unsupported, list) else -1
        _progress(
            f"Description Agent round {iteration}/{max_rounds} complete | "
            f"elapsed={time.perf_counter() - round_started:.1f}s "
            f"ready={bool(observation.get('ready', False))} "
            f"unsupported_claims={unsupported_count}"
        )
        if (
            iteration >= 2
            and bool(observation.get("ready", False))
            and isinstance(unsupported, list)
            and not unsupported
        ):
            _progress(f"ReAct stopping early after round {iteration}: evidence check passed")
            break
    _progress(
        f"DONE  Description Agent (ReAct) refinement | rounds={len(trace)} "
        f"elapsed={time.perf_counter() - react_started:.1f}s"
    )
    return draft, trace


def main() -> None:
    pipeline_started = time.perf_counter()
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--database-row", type=int)
    target.add_argument("--painting-name")
    target.add_argument("--query-embedding", type=Path)
    parser.add_argument("--query-image", type=Path)
    parser.add_argument("--atom-descriptions", type=Path, default=DEFAULT_ATOMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--coordinator-backend",
        choices=["azure", "gemini", "openai", "azure-openai-fallback"],
        default="openai",
    )
    parser.add_argument("--coordinator-model", default="gpt-5.1")
    parser.add_argument("--react-rounds", type=int, default=3)
    parser.add_argument(
        "--inspect-atom-neighbor-images",
        action="store_true",
        help=(
            "Retrieve the five atom-cosine neighbors and use coordinator vision to "
            "summarize their recurring scene type as supplementary evidence."
        ),
    )
    parser.add_argument(
        "--metadata-retrieval",
        choices=["embedding-l2", "embedding-cosine", "atom-cosine"],
        default="embedding-cosine",
        help=(
            "Retrieve metadata using pooled-embedding L2 distance, pooled-embedding "
            "cosine similarity (default), or weighted-atom cosine similarity."
        ),
    )
    args = parser.parse_args()
    if args.react_rounds < 2:
        parser.error("--react-rounds must be at least 2")

    database = pd.read_excel(DATABASE_PATH)
    query_embedding: np.ndarray | None = None
    if args.query_embedding is not None:
        query_path = args.query_embedding.expanduser().resolve()
        query_embedding = np.asarray(np.load(query_path), dtype=np.float32).reshape(-1)
        query_image = args.query_image.expanduser().resolve() if args.query_image else None
        database_row = None
        target_info = {
            "database_row": None,
            "painting_name": query_image.stem if query_image else query_path.stem,
            "image": str(query_image) if query_image else "",
            "query_embedding": str(query_path),
        }
    else:
        database_row = _resolve_target(database, args.database_row, args.painting_name)
        title_col = _column(database, ["painting_name", "painting name", "title"])
        image_col = _column(database, ["Image", "image"])
        target_info = {
            "database_row": database_row,
            "painting_name": _safe_text(database.iloc[database_row][title_col]),
            "image": _safe_text(database.iloc[database_row][image_col]),
        }
    blind_target_info = {
        "database_row": database_row,
        "source": "external query" if query_embedding is not None else "database row",
    }
    target_image_path = _resolve_target_image(args.query_image, target_info)
    matching_image_rows = _matching_database_image_rows(database, target_image_path)

    pooled_n = int(np.load(EMBEDDINGS_PATH, mmap_mode="r").shape[0])
    valid = np.ones(pooled_n, dtype=bool)
    atom_path = args.atom_descriptions.expanduser().resolve()
    descriptions = _load_atom_descriptions(atom_path)
    weighted_atoms = (
        _encode_external_task2_weights(query_embedding, descriptions)
        if query_embedding is not None
        else _load_canonical_task2_weights(
            int(database_row),
            valid,
            descriptions,
        )
    )
    atom_cosine_neighbors = (
        _nearest_atom_cosine_metadata(
            database,
            database_row,
            valid,
            weighted_atoms,
            excluded_database_rows=matching_image_rows,
        )
        if args.inspect_atom_neighbor_images or args.metadata_retrieval == "atom-cosine"
        else []
    )
    if args.metadata_retrieval == "atom-cosine":
        metadata_neighbors = atom_cosine_neighbors
    elif args.metadata_retrieval == "embedding-cosine":
        metadata_neighbors = _nearest_embedding_cosine_metadata(
            database,
            database_row,
            valid,
            query_embedding=query_embedding,
            excluded_database_rows=matching_image_rows,
        )
    else:
        metadata_neighbors = _nearest_l2_metadata(
            database,
            database_row,
            valid,
            query_embedding=query_embedding,
        )
    metadata_neighborhood = _metadata_neighborhood(metadata_neighbors)
    metadata = metadata_neighbors[0]

    apply_gateway_env(
        backend=args.coordinator_backend,
        deployment=args.coordinator_model,
    )
    _progress(
        f"Description Agent configured | requested={args.coordinator_backend} "
        f"active={active_gateway_backend()} model={args.coordinator_model} "
        f"react_rounds={args.react_rounds}"
    )
    if args.coordinator_backend not in {
        "azure",
        "openai",
        "azure-openai-fallback",
    }:
        raise ValueError(
            "Direct visual verification requires Azure, OpenAI, or Azure→OpenAI fallback"
        )
    atom_neighbor_vision: dict[str, Any] = {
        "status": "disabled",
        "scene_type_consensus": "not assessed",
    }
    atom_neighbor_unavailable_images: list[dict[str, Any]] = []
    if args.inspect_atom_neighbor_images:
        ranked_neighbor_images, atom_neighbor_unavailable_images = (
            _resolve_atom_neighbor_images(atom_cosine_neighbors)
        )
        if ranked_neighbor_images:
            atom_neighbor_vision = _coordinator_multi_vision_json(
                ATOM_NEIGHBOR_VISION_PROMPT,
                ranked_neighbor_images,
                "Coordinator blind atom-neighbor scene inspection",
                backend=args.coordinator_backend,
                model=args.coordinator_model,
            )
            atom_neighbor_vision["status"] = "completed"
            atom_neighbor_vision["inspected_ranks"] = [
                rank for rank, _ in ranked_neighbor_images
            ]
        else:
            atom_neighbor_vision = {
                "status": "unavailable",
                "scene_type_consensus": "not assessed because no atom-neighbor image files resolved",
                "inspected_ranks": [],
            }
    visual_observations = _coordinator_vision_json(
        VISUAL_OBSERVATION_PROMPT,
        target_image_path,
        "Coordinator direct visual observation",
        backend=args.coordinator_backend,
        model=args.coordinator_model,
    )
    rerank = _coordinator_json(
        RERANK_PROMPT.format(
            target=json.dumps(blind_target_info, ensure_ascii=False, indent=2),
            atoms=_format_atoms(weighted_atoms),
        ),
        "Coordinator weighted rerank",
    )
    ranked_ids = _validate_rerank(rerank, weighted_atoms)
    atom_by_id = {int(atom["atom_id"]): atom for atom in weighted_atoms}
    ranked_atoms = [atom_by_id[atom_id] for atom_id in ranked_ids]
    required_high_weight_atoms = _required_high_weight_atoms(ranked_atoms)
    required_high_weight_atom_ids = {
        int(atom["atom_id"]) for atom in required_high_weight_atoms
    }
    required_high_weight_text = _format_required_high_weight_atoms(
        required_high_weight_atoms
    )

    def validate_synthesis(payload: dict[str, Any]) -> None:
        description = str(payload.get("formal_visual_description", "")).strip()
        if not description:
            raise ValueError("Synthesis omitted formal_visual_description")
        payload["formal_visual_description"] = _validate_description_citations(
            description,
            set(ranked_ids),
            require_atom_each_sentence=True,
            required_atom_ids=required_high_weight_atom_ids,
        )

    synthesis = _coordinator_json(
        SYNTHESIS_PROMPT.format(
            target=json.dumps(blind_target_info, ensure_ascii=False, indent=2),
            atoms=_format_atoms(ranked_atoms),
            required_high_weight_atoms=required_high_weight_text,
        ),
        "Coordinator formal synthesis",
        validator=validate_synthesis,
    )
    if not str(synthesis.get("formal_visual_description", "")).strip():
        raise ValueError("Coordinator synthesis omitted formal_visual_description")
    l2_hypotheses = _coordinator_json(
        CONTEXT_PROMPT.format(
            visual_observations=json.dumps(
                visual_observations, ensure_ascii=False, indent=2
            ),
            formal_description=str(synthesis["formal_visual_description"]).strip(),
            metadata=json.dumps(metadata_neighborhood, ensure_ascii=False, indent=2),
            atom_neighbor_vision=json.dumps(
                atom_neighbor_vision,
                ensure_ascii=False,
                indent=2,
            ),
        ),
        "Coordinator five-neighbor subject-period-date-attribution hypothesis",
    )
    required_context_fields = (
        "neighbor_consensus_summary",
        "period_hypothesis",
        "period_confidence",
        "date_hypothesis",
        "date_confidence",
        "attribution_context",
        "attribution_confidence",
    )
    missing_context_fields = [
        field for field in required_context_fields if not l2_hypotheses.get(field)
    ]
    if missing_context_fields:
        raise ValueError(
            "Coordinator omitted five-neighbor context fields: "
            f"{missing_context_fields}"
        )
    final_description, react_trace = _iterative_react_rewrite(
        str(synthesis["formal_visual_description"]),
        ranked_atoms,
        l2_hypotheses,
        visual_observations,
        max_rounds=args.react_rounds,
    )

    output = {
        "target": target_info,
        "pipeline": {
            "atom_selection": (
                "canonical Task 2 Elastic Net live encoding"
                if query_embedding is not None
                else "canonical Task 2 global sparse-code row"
            ),
            "artist_or_cohort_conditioning": False,
            "task2_sparse_codes": str(SPARSE_CODES_PATH),
            "task2_dictionary_meta": str(DICTIONARY_META_PATH),
            "post_encoding_atom_cap": None,
            "selected_atom_count": len(weighted_atoms),
            "required_high_weight_atom_ids": sorted(required_high_weight_atom_ids),
            "high_weight_coverage_cumulative_target_percent": 85.0,
            "high_weight_coverage_minimum_atoms": 5,
            "high_weight_coverage_maximum_atoms": 10,
            "external_query_encoded_live": query_embedding is not None,
            "first_pass_atom_descriptions": str(atom_path),
            "metadata_retrieval": args.metadata_retrieval,
            "metadata_neighbor_count": len(metadata_neighbors),
            "metadata_neighbor_aggregation": "rank-and-similarity-weighted five-neighbor consensus",
            "metadata_self_excluded_database_rows": sorted(matching_image_rows),
            "supplementary_atom_cosine_neighbor_vision": args.inspect_atom_neighbor_images,
            "atom_cosine_neighbor_count": len(atom_cosine_neighbors),
            "faiss": (
                "IndexFlatIP over L2-normalized full uncapped Task 2 atom coefficient vectors"
                if args.metadata_retrieval == "atom-cosine"
                else "IndexFlatIP over L2-normalized raw whole 2304-D pooled embeddings"
                if args.metadata_retrieval == "embedding-cosine"
                else "IndexFlatL2 over raw whole 2304-D pooled embeddings"
            ),
            "faiss_self_match_excluded": bool(
                database_row is not None or matching_image_rows
            ),
            "neighbor_description_loaded": False,
            "l2_used_in_formal_synthesis": False,
            "direct_visual_verification": True,
            "direct_visual_verification_image": str(target_image_path),
            "direct_vision_role": "secondary high-confidence contradiction check",
            "minimum_atom_cited_sentence_fraction": 0.75,
            "maximum_vision_cited_sentence_fraction": 0.34,
            "l2_subject_visually_tested_after_formal_synthesis": True,
            "iterative_react_rewrite": True,
            "description_agent": True,
            "react_max_rounds": args.react_rounds,
            "react_rounds_completed": len(react_trace),
            "coordinator_backend": args.coordinator_backend,
            "coordinator_model": args.coordinator_model,
            "inline_evidence_citations": True,
        },
        "direct_visual_observations": visual_observations,
        "nearest_metadata_match": metadata,
        "nearest_metadata_matches": metadata_neighbors,
        "metadata_neighborhood": metadata_neighborhood,
        "atom_cosine_metadata_matches": atom_cosine_neighbors,
        "atom_cosine_neighbor_vision": atom_neighbor_vision,
        "atom_cosine_neighbor_unavailable_images": atom_neighbor_unavailable_images,
        "nearest_l2_metadata_match": (
            metadata if args.metadata_retrieval == "embedding-l2" else None
        ),
        "nearest_l2_metadata_matches": (
            metadata_neighbors if args.metadata_retrieval == "embedding-l2" else None
        ),
        "weighted_original_global_atoms": ranked_atoms,
        "coordinator_rerank": {
            "reranked_atom_ids": ranked_ids,
            "rationale": rerank.get("rationale", []),
        },
        **synthesis,
        "citation_legend": {
            "Atoms": "Weighted global-dictionary evidence; IDs resolve below.",
            "Vision": "Direct observation from the supplied painting image.",
            "L2": "Visually tested five-neighbor consensus for subject or period context.",
            "atoms": [
                {
                    "atom_id": int(atom["atom_id"]),
                    "label": atom["label"],
                    "coefficient": float(atom["coefficient"]),
                    "importance_percent": float(atom["importance_percent"]),
                }
                for atom in ranked_atoms
            ],
        },
        "formal_visual_description_citations": _build_citation_map(
            str(synthesis["formal_visual_description"]),
            ranked_atoms,
        ),
        "l2_hypotheses": l2_hypotheses,
        "react_trace": react_trace,
        "final_formal_description": final_description,
        "final_formal_description_citations": _build_citation_map(
            final_description,
            ranked_atoms,
        ),
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"DONE target={target_info['painting_name']!r} "
        f"task2_atoms={len(ranked_atoms)} "
        f"{args.metadata_retrieval}_neighbors="
        f"{[neighbor['painting_name'] for neighbor in metadata_neighbors]!r} "
        f"output={output_path} "
        f"elapsed={time.perf_counter() - pipeline_started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
