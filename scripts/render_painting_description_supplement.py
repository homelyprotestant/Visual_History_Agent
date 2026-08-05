#!/usr/bin/env python3
"""Render a painting-description JSON artifact as a readable LaTeX supplement."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from datetime import date
from pathlib import Path
from typing import Any


TAG_RE = re.compile(r"(\[Atoms:\s*[0-9,\s]+\]|\[Vision\]|\[L2\])")


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "–": "--",
        "‑": "-",
        "—": "---",
        "−": "-",
        "…": r"\ldots{}",
        "’": "'",
        "‘": "`",
        "“": "``",
        "”": "''",
        "×": r"$\times$",
        "≥": r"$\geq$",
        "≤": r"$\leq$",
        "→": r"$\rightarrow$",
        "ō": r"\={o}",
        "ū": r"\={u}",
    }
    return "".join(replacements.get(char, char) for char in text)


def evidence_text(value: Any) -> str:
    parts = TAG_RE.split(str(value))
    rendered: list[str] = []
    for part in parts:
        if not part:
            continue
        if part.startswith("[Atoms:"):
            rendered.append(r"\atomtag{" + latex_escape(part[1:-1]) + "}")
        elif part == "[Vision]":
            rendered.append(r"\visiontag{Vision}")
        elif part == "[L2]":
            rendered.append(r"\contexttag{Context}")
        else:
            rendered.append(latex_escape(part))
    return "".join(rendered)


def humanize(key: str) -> str:
    special = {
        "faiss": "FAISS index",
        "l2": "L2",
        "rcs": "RCS",
        "vit": "ViT",
    }
    words = key.replace("_", " ").split()
    return " ".join(special.get(word.lower(), word.capitalize()) for word in words)


def display_value(value: Any) -> str:
    if value is None:
        return "Not applicable"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def section(title: str, level: int = 1) -> str:
    command = {1: "section", 2: "subsection", 3: "subsubsection"}[level]
    return rf"\{command}{{{latex_escape(title)}}}" + "\n"


def paragraph(text: Any) -> str:
    return evidence_text(text) + "\n\n"


def itemize(items: list[Any], *, empty: str | None = None) -> str:
    if not items:
        return paragraph(empty) if empty else ""
    body = "\n".join(r"\item " + evidence_text(item) for item in items)
    return "\\begin{itemize}\n" + body + "\n\\end{itemize}\n"


def description_list(items: list[tuple[str, Any]]) -> str:
    body = "\n".join(
        rf"\item[\textbf{{{latex_escape(label)}}}] {evidence_text(display_value(value))}"
        for label, value in items
    )
    return "\\begin{description}\n" + body + "\n\\end{description}\n"


def callout(title: str, body: str, style: str = "evidencebox") -> str:
    return (
        rf"\begin{{{style}}}{{{latex_escape(title)}}}" + "\n"
        + body.strip()
        + "\n"
        + rf"\end{{{style}}}"
        + "\n\n"
    )


def render_pipeline(data: dict[str, Any]) -> str:
    pipeline = data["pipeline"]
    rows = [
        ("Atom selection", pipeline.get("atom_selection")),
        ("Active atoms", pipeline.get("selected_atom_count")),
        ("Post-encoding atom cap", pipeline.get("post_encoding_atom_cap")),
        ("Required high-weight atoms", pipeline.get("required_high_weight_atom_ids")),
        (
            "High-weight coverage",
            f"{pipeline.get('high_weight_coverage_cumulative_target_percent')}% cumulative mass; "
            f"{pipeline.get('high_weight_coverage_minimum_atoms')}--"
            f"{pipeline.get('high_weight_coverage_maximum_atoms')} atoms",
        ),
        ("Artist or cohort conditioning", pipeline.get("artist_or_cohort_conditioning")),
        ("Metadata retrieval", pipeline.get("metadata_retrieval")),
        ("FAISS implementation", pipeline.get("faiss")),
        ("Neighbour description loaded", pipeline.get("neighbor_description_loaded")),
        ("Direct visual verification", pipeline.get("direct_visual_verification")),
        ("Vision role", pipeline.get("direct_vision_role")),
        (
            "Citation constraints",
            f"at least {100 * pipeline.get('minimum_atom_cited_sentence_fraction', 0):.0f}% "
            f"atom-cited sentences; at most "
            f"{100 * pipeline.get('maximum_vision_cited_sentence_fraction', 0):.0f}% "
            "vision-cited sentences",
        ),
        (
            "ReAct",
            f"{pipeline.get('react_rounds_completed')} completed rounds "
            f"(maximum {pipeline.get('react_max_rounds')})",
        ),
        (
            "Coordinator",
            f"{pipeline.get('coordinator_model')} via {pipeline.get('coordinator_backend')}",
        ),
    ]
    return description_list(rows)


def render_prompt_guardrails(data: dict[str, Any]) -> str:
    required = ", ".join(
        str(atom_id) for atom_id in data["pipeline"].get("required_high_weight_atom_ids", [])
    )
    out: list[str] = []
    out.append(
        callout(
            "Concise ReAct revision prompt",
            paragraph(
                "Revise the current formal description using weighted atom evidence, direct "
                "visual observations, and provisional subject and period hypotheses. Weighted "
                "atoms must remain the primary evidence. Direct vision may correct clear "
                "contradictions but must not replace the atom-derived account. Use the tested "
                "subject only as a qualified identification and the period hypothesis only to "
                "contextualize formal qualities. Produce 6--10 self-contained sentences with "
                "inline evidence citations."
            ),
        )
    )
    out.append(section("Generation constraints", 2))
    out.append(
        itemize(
            [
                "Normalized non-negative atom coefficients provide the primary measure of evidential importance.",
                f"All required high-weight atoms ({required}) must be substantively represented and accurately cited.",
                "At least 75% of sentences must cite one or more supplied atoms.",
                "No more than approximately one-third of sentences may cite direct vision.",
                "Every sentence must end with an atom, direct-vision, or contextual evidence tag.",
                "Only atom identifiers present in the supplied sparse code may be cited.",
                "The initial formal synthesis receives atom evidence only and is blind to artist, title, date, culture, period, cohort and nearest-neighbour metadata.",
                "Direct vision is secondary and constraint-only; it cannot replace or wholesale reinterpret the atom-derived account.",
                "The visually tested subject must remain qualified, and unsupported fine iconography cannot be introduced.",
                "Period context may calibrate formal vocabulary but cannot establish attribution, exact date or cultural identity.",
                "Craquelure may support an aged-film observation but cannot independently establish date, authenticity, originality or copy direction.",
                "Rejected nearest-neighbour details cannot re-enter later drafts.",
            ]
        )
    )
    out.append(section("Acceptance audit", 2))
    out.append(
        itemize(
            [
                "Weighted atoms must control the organization and majority of the description.",
                "Stronger atoms must exert visibly greater influence than weaker atoms.",
                "All high-weight atoms must contribute distinct, supported formal tendencies rather than mechanically repeating the dominant atom.",
                "Formal claims must retain atom citations rather than relying only on vision or context.",
                "No attribution claim, unsupported iconography or contradicted neighbour detail may appear.",
                "The candidate is accepted only when its atom identifiers and citation coverage are valid, its unsupported-claim list is empty, and no substantive revision remains.",
                "The recorded observation is a public evidence audit rather than private chain-of-thought.",
            ]
        )
    )
    return "".join(out)


def render_atoms(data: dict[str, Any]) -> str:
    required = set(data["pipeline"].get("required_high_weight_atom_ids", []))
    atoms = data["weighted_original_global_atoms"]
    lines = [
        r"\begin{longtable}{@{}r r r p{0.52\textwidth} c@{}}",
        r"\caption{Active atoms in descending coefficient order. Percentages are normalized "
        r"over the positive coefficient mass.}\label{tab:atoms}\\",
        r"\toprule",
        r"Atom & Coefficient & Mass (\%) & Corpus-derived label & Required\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Atom & Coefficient & Mass (\%) & Corpus-derived label & Required\\",
        r"\midrule",
        r"\endhead",
    ]
    for atom in atoms:
        marker = r"\textbullet" if atom["atom_id"] in required else ""
        lines.append(
            f"{atom['atom_id']} & {atom['coefficient']:.6f} & "
            f"{atom['importance_percent']:.3f} & "
            f"{latex_escape(atom['label'])} & {marker}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines) + "\n"


def render_atom_profiles(data: dict[str, Any]) -> str:
    contributions: dict[int, list[str]] = {}
    for record in data.get("atom_contributions", []):
        contributions.setdefault(record["atom_id"], []).append(record["contribution"])

    output: list[str] = []
    for atom in data["weighted_original_global_atoms"]:
        atom_id = atom["atom_id"]
        output.append(section(f"Atom {atom_id}: {atom['label']}", 2))
        output.append(
            description_list(
                [
                    ("Coefficient", f"{atom['coefficient']:.6f}"),
                    ("Positive mass", f"{atom['importance_percent']:.3f}%"),
                    ("Interpretive confidence", atom.get("confidence", "not recorded")),
                ]
            )
        )
        if contributions.get(atom_id):
            output.append(callout("Painting-level contribution", itemize(contributions[atom_id])))
        output.append(paragraph(atom["painterly_description"]))
    return "".join(output)


def render_direct_vision(data: dict[str, Any]) -> str:
    vision = data["direct_visual_observations"]
    principal = vision["principal_scene_figures"]
    out: list[str] = []
    out.append(
        callout(
            "Binding figure constraint",
            paragraph(vision["subject_constraints"]["constraint_summary"]),
            "warningbox",
        )
    )
    out.append(section("Principal scene figures", 2))
    out.append(paragraph(f"Count: {principal['count']}."))
    for figure in principal.get("figures", []):
        out.append(
            description_list(
                [
                    ("Figure", figure.get("figure_id")),
                    ("Age class", figure.get("age_class")),
                    ("Position", figure.get("position")),
                    ("Pose or action", figure.get("pose_or_action")),
                ]
            )
        )
    out.append(itemize(principal.get("relationships", [])))

    out.append(section("Secondary human images", 2))
    secondary = [
        f"{entry.get('kind', 'uncertain').capitalize()} ({entry.get('count', 0)}): "
        f"{entry.get('location', '')}"
        for entry in vision.get("secondary_human_images", [])
    ]
    out.append(itemize(secondary))
    out.append(section("Setting and objects", 2))
    out.append(itemize(vision.get("setting_and_objects", [])))
    out.append(section("Direct formal observations", 2))
    out.append(itemize(vision.get("formal_observations", [])))

    surface = vision["surface_condition"]
    out.append(section("Surface and condition", 2))
    out.append(
        description_list(
            [
                ("Craquelure visibility", surface["craquelure"].get("visibility")),
                ("Pattern, scale and distribution", surface["craquelure"].get("pattern_scale_distribution")),
                ("Confidence", surface["craquelure"].get("confidence")),
            ]
        )
    )
    out.append(section("Paint and support observations", 3))
    out.append(itemize(surface.get("paint_and_support_observations", [])))
    out.append(section("Possible age indicators", 3))
    out.append(itemize(surface.get("possible_age_indicators", [])))
    out.append(section("Alternative explanations and imaging artefacts", 3))
    out.append(itemize(surface.get("possible_restoration_or_imaging_artifacts", [])))
    out.append(callout("Condition caution", paragraph(surface.get("condition_caution", "")), "warningbox"))

    constraints = vision["subject_constraints"]
    out.append(section("Subject constraints", 2))
    out.append(section("Compatible broad subjects", 3))
    out.append(itemize(constraints.get("compatible_broad_subjects", [])))
    out.append(section("Incompatible subject features", 3))
    out.append(itemize(constraints.get("incompatible_subject_features", [])))
    out.append(section("Visual uncertainties", 2))
    out.append(itemize(vision.get("uncertainties", [])))
    return "".join(out)


def render_context(data: dict[str, Any]) -> str:
    matches = data.get("nearest_metadata_matches") or [data["nearest_metadata_match"]]
    hypotheses = data["l2_hypotheses"]
    out: list[str] = []
    out.append(section("Five-neighbor metadata evidence", 2))
    for match in matches:
        out.append(section(f"Neighbor {match.get('rank', '?')}", 3))
        out.append(
            description_list(
                [
                    ("Painting", match.get("painting_name")),
                    ("Artist", match.get("artist")),
                    ("Date", match.get("date")),
                    ("Culture and region", f"{match.get('culture')}; {match.get('region')}"),
                    ("Period", match.get("art_historical_time_period")),
                    ("Cosine similarity", f"{match.get('cosine_similarity', 0):.6f}"),
                    ("Retrieval features", match.get("retrieval_features")),
                ]
            )
        )
    out.append(
        callout(
            "Retrieval boundary",
            paragraph(
                "Only structured metadata from these five neighbors was used. Their catalogue "
                "descriptions were never loaded. Recurring patterns are weighted by rank and "
                "similarity, and neither an individual match nor the consensus constitutes attribution."
            ),
            "warningbox",
        )
    )
    out.append(section("Visually tested hypotheses", 2))
    out.append(
        description_list(
            [
                ("Subject hypothesis", hypotheses.get("subject_hypothesis")),
                ("Subject confidence", hypotheses.get("subject_confidence")),
                ("Neighbour-proposed subject", hypotheses.get("l2_subject_as_proposed")),
                ("Visual compatibility", hypotheses.get("visual_compatibility")),
                ("Period hypothesis", hypotheses.get("period_hypothesis")),
                ("Period confidence", hypotheses.get("period_confidence")),
                ("Date hypothesis", hypotheses.get("date_hypothesis")),
                ("Date confidence", hypotheses.get("date_confidence")),
                ("Attribution context", hypotheses.get("attribution_context")),
                ("Attribution-context confidence", hypotheses.get("attribution_confidence")),
                ("Five-neighbor consensus", hypotheses.get("neighbor_consensus_summary")),
            ]
        )
    )
    out.append(section("Compatibility assessment", 3))
    out.append(paragraph(hypotheses.get("compatibility_observation", "")))
    out.append(section("Rejected neighbour details", 3))
    out.append(itemize(hypotheses.get("rejected_l2_details", []), empty="None recorded."))
    out.append(section("Binding limitations", 3))
    out.append(itemize(hypotheses.get("limitations", [])))
    return "".join(out)


def render_react(data: dict[str, Any]) -> str:
    output: list[str] = []
    for trace in data["react_trace"]:
        iteration = trace["iteration"]
        observation = trace["observation"]
        status = "accepted" if observation["ready"] else "revision required"
        output.append(section(f"Iteration {iteration}: {status}", 2))
        output.append(callout("Revision action", paragraph(trace["action_summary"])))
        output.append(section("Use of contextual hypotheses", 3))
        output.append(paragraph(trace["hypothesis_usage"]))
        output.append(section("Candidate description", 3))
        output.append(paragraph(trace["candidate_description"]))
        output.append(section("Evidence audit", 3))
        output.append(
            description_list(
                [
                    ("Ready", observation["ready"]),
                    ("Supported claims", len(observation.get("supported_claims", []))),
                    (
                        "Unsupported or overstated claims",
                        len(observation.get("unsupported_or_overstated_claims", [])),
                    ),
                ]
            )
        )
        output.append(r"\paragraph{Supported claims.}" + "\n")
        output.append(itemize(observation.get("supported_claims", [])))
        output.append(r"\paragraph{Unsupported or overstated claims.}" + "\n")
        output.append(
            itemize(
                observation.get("unsupported_or_overstated_claims", []),
                empty="None. The candidate passed this component of the audit.",
            )
        )
        output.append(r"\paragraph{Revision instructions.}" + "\n")
        output.append(
            itemize(
                observation.get("revision_instructions", []),
                empty="None. No further substantive revision was required.",
            )
        )
    return "".join(output)


def render_final_citations(data: dict[str, Any]) -> str:
    out: list[str] = []
    for entry in data["final_formal_description_citations"]:
        labels = "; ".join(entry.get("atom_labels", [])) or "None"
        sources: list[str] = []
        if entry.get("atom_ids"):
            sources.append("Atoms " + ", ".join(str(value) for value in entry["atom_ids"]))
        if entry.get("uses_direct_vision"):
            sources.append("direct vision")
        if entry.get("uses_l2_context"):
            sources.append("context")
        out.append(section(f"Sentence {entry['sentence_number']}", 3))
        out.append(paragraph(entry["sentence"]))
        out.append(
            description_list(
                [
                    ("Evidence sources", "; ".join(sources)),
                    ("Atom labels", labels),
                ]
            )
        )
    return "".join(out)


def render_painting_body(
    data: dict[str, Any],
    *,
    source_json: Path | None = None,
    include_appendix: bool = True,
    title_prefix: str = "",
) -> str:
    """Body sections for one painting report (no preamble / document wrapper)."""
    target = data["target"]
    atom_count = int(data["pipeline"].get("selected_atom_count") or 0)
    prefix = f"{title_prefix}: " if title_prefix else ""
    source_name = source_json.name if source_json is not None else "painting_description_from_global_atoms.json"
    out: list[str] = []
    out.append(section(f"{prefix}Case-study record"))
    out.append(
        description_list(
            [
                ("Target identifier", target.get("painting_name")),
                ("Database row", target.get("database_row")),
                ("External query", data["pipeline"].get("external_query_encoded_live")),
                ("Image file", Path(target.get("image", "")).name),
                ("Embedding file", Path(target.get("query_embedding", "")).name),
                ("Source JSON", source_name),
            ]
        )
    )
    out.append(section(f"{prefix}Pipeline configuration"))
    out.append(render_pipeline(data))
    out.append(section(f"{prefix}Reasoning prompt and guardrails"))
    out.append(render_prompt_guardrails(data))

    final_text = str(
        data.get("final_formal_description") or data.get("formal_visual_description") or ""
    ).strip()
    out.append(section(f"{prefix}Final evidence-grounded description"))
    out.append(callout("Accepted description", paragraph(final_text)))
    out.append(
        paragraph(
            "The accepted description completed "
            f"{data['pipeline'].get('react_rounds_completed', 0)} ReAct audit rounds."
        )
    )

    out.append(section(f"{prefix}Weighted sparse representation"))
    out.append(
        paragraph(
            "The query embedding was encoded against the fixed 1,000-atom global dictionary "
            "using uncapped, non-negative Elastic Net. The "
            f"{atom_count} non-zero coefficients below sum "
            "to 100% of the positive coefficient mass; they do not represent calibrated "
            "probabilities."
        )
    )
    out.append(render_atoms(data))
    out.append(section("Coordinator reranking", 2))
    rerank = data.get("coordinator_rerank") or {}
    out.append(
        description_list(
            [
                ("Reranked atom order", rerank.get("reranked_atom_ids")),
                ("Atoms removed", "None"),
            ]
        )
    )
    out.append(itemize(rerank.get("rationale", [])))

    out.append(section(f"{prefix}Initial atom-only formal account"))
    out.append(
        paragraph(
            "This first description was generated before nearest-neighbour metadata or direct "
            "visual observations were introduced."
        )
    )
    out.append(
        callout(
            "Initial description",
            paragraph(data.get("formal_visual_description", "")),
        )
    )
    out.append(section("Formal elements", 2))
    out.append(
        description_list(
            [(humanize(key), value) for key, value in (data.get("formal_elements") or {}).items()]
        )
    )
    out.append(section("Initial uncertainties", 2))
    out.append(itemize(data.get("uncertainties", [])))

    out.append(section(f"{prefix}Direct visual verification"))
    out.append(render_direct_vision(data))

    out.append(section(f"{prefix}Nearest-neighbour context and tested hypotheses"))
    out.append(render_context(data))

    out.append(section(f"{prefix}ReAct revision and evidence-audit trace"))
    out.append(
        paragraph(
            "Each action produced a revised candidate. A separate observation step then "
            "identified supported and unsupported claims and issued binding revision instructions. "
            "The trace below reports the public evidence audit, not private chain-of-thought."
        )
    )
    out.append(render_react(data))

    out.append(section(f"{prefix}Sentence-level provenance of the final account"))
    out.append(render_final_citations(data))

    if include_appendix:
        out.append(section(f"{prefix}Complete atom profiles"))
        out.append(
            paragraph(
                "The following profiles are frozen corpus-level interpretations generated before "
                "this query was analysed. Artist and subject names inside these profiles describe "
                "retrieval evidence and must not be read as attributions of the query painting."
            )
        )
        out.append(render_atom_profiles(data))
        out.append(section(f"{prefix}Reproducibility and interpretive boundaries"))
        out.append(
            itemize(
                [
                    "The query was encoded live, but the global dictionary and atom descriptions were precomputed.",
                    "Atom selection was blind to the target's artist, title, cohort, period and nearest neighbours.",
                    "The sparse code was uncapped; sparsity was determined by the Elastic Net objective.",
                    "The nearest-neighbour search used cosine similarity over L2-normalized raw pooled embeddings.",
                    "Only structured metadata from the nearest neighbour was loaded; its description was excluded.",
                    "Direct vision served as a secondary contradiction check and condition-observation source.",
                    "Atom coefficients measure reconstruction contribution, not attribution probability.",
                    "The generated account remains an interpretive computational output requiring scholarly review.",
                ]
            )
        )
    return "".join(out)


def build_document(data: dict[str, Any], image_filename: str, source_json: Path) -> str:
    target = data["target"]
    legend = data["citation_legend"]
    today = date.today().isoformat()
    case_label = str(target.get("painting_name") or Path(image_filename).stem).replace("_", " ")
    preamble = r"""\documentclass[10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage[a4paper,margin=22mm,headheight=14pt]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{enumitem}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{fancyhdr}
\usepackage[most]{tcolorbox}
\usepackage{titlesec}
\definecolor{ink}{HTML}{1E2933}
\definecolor{muted}{HTML}{5C6873}
\definecolor{rule}{HTML}{CBD2D8}
\definecolor{atom}{HTML}{E7F0F5}
\definecolor{vision}{HTML}{F1EADB}
\definecolor{context}{HTML}{E8EEE5}
\definecolor{warning}{HTML}{F6ECE8}
\hypersetup{colorlinks=true,linkcolor=ink,urlcolor=muted,pdfborder={0 0 0}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.55em}
\setlist[itemize]{leftmargin=1.5em,itemsep=0.25em,topsep=0.25em}
\renewcommand{\arraystretch}{1.16}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Visual History Agent}
\fancyhead[R]{\small Single-painting evidence report}
\fancyfoot[C]{\small\thepage}
\titleformat{\section}{\Large\bfseries\color{ink}}{\thesection}{0.7em}{}
\titleformat{\subsection}{\large\bfseries\color{ink}}{\thesubsection}{0.7em}{}
\titleformat{\subsubsection}{\normalsize\bfseries\color{ink}}{\thesubsubsection}{0.7em}{}
\newcommand{\atomtag}[1]{\hspace{0.15em}\colorbox{atom}{\scriptsize\textsf{#1}}\hspace{0.1em}}
\newcommand{\visiontag}[1]{\hspace{0.15em}\colorbox{vision}{\scriptsize\textsf{#1}}\hspace{0.1em}}
\newcommand{\contexttag}[1]{\hspace{0.15em}\colorbox{context}{\scriptsize\textsf{#1}}\hspace{0.1em}}
\newtcolorbox{evidencebox}[1]{enhanced,breakable,colback=atom!35,colframe=rule,
  boxrule=0.5pt,arc=0pt,title=#1,fonttitle=\bfseries,coltitle=ink}
\newtcolorbox{warningbox}[1]{enhanced,breakable,colback=warning,colframe=rule,
  boxrule=0.5pt,arc=0pt,title=#1,fonttitle=\bfseries,coltitle=ink}
\begin{document}
"""
    out: list[str] = [preamble]
    out.append(
        r"\begin{center}"
        "\n"
        r"{\LARGE\bfseries Visual History Agent\par}"
        "\n"
        r"\vspace{0.35em}"
        "\n"
        r"{\Large Single-painting evidence report\par}"
        "\n"
        r"\vspace{0.55em}"
        "\n"
        rf"{{\large Case study: {latex_escape(case_label)}\par}}"
        "\n"
        r"\vspace{0.5em}"
        "\n"
        rf"{{\small Generated from the frozen pipeline artifact on {latex_escape(today)}\par}}"
        "\n"
        r"\end{center}"
        "\n\n"
    )
    out.append(
        r"\begin{figure}[h!]\centering"
        "\n"
        rf"\includegraphics[width=0.67\textwidth]{{{latex_escape(image_filename)}}}"
        "\n"
        r"\caption{Query image supplied to the image-only encoder and direct visual-verification stage. "
        r"The title and attribution shown in this report are editorial identifiers and were not used "
        r"to select the sparse atoms.}"
        "\n"
        r"\end{figure}"
        "\n"
    )
    out.append(
        callout(
            "How to read this supplement",
            paragraph(
                "This document reformats the machine-readable analysis artifact as a traceable "
                "scholarly record. Corpus-derived atom evidence is marked in blue, direct visual "
                "observation in tan, and nearest-neighbour context in green. These evidence classes "
                "have different epistemic roles and should not be treated as interchangeable."
            )
            + description_list(
                [
                    ("Atom evidence", legend["Atoms"]),
                    ("Direct vision", legend["Vision"]),
                    ("Context", legend["L2"]),
                ]
            ),
        )
    )
    out.append(r"\tableofcontents\clearpage" + "\n")
    out.append(render_painting_body(data, source_json=source_json, include_appendix=True))
    out.append(r"\end{document}" + "\n")
    return "".join(out)


def compile_tex(tex_path: Path) -> Path:
    tectonic = shutil.which("tectonic")
    xelatex = shutil.which("xelatex")
    pdflatex = shutil.which("pdflatex")
    if tectonic:
        command = [tectonic, "--keep-logs", "--keep-intermediates", tex_path.name]
        subprocess.run(command, cwd=tex_path.parent, check=True)
    elif xelatex or pdflatex:
        compiler = xelatex or pdflatex
        assert compiler is not None
        for _ in range(2):
            subprocess.run(
                [compiler, "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
                cwd=tex_path.parent,
                check=True,
            )
    else:
        raise RuntimeError("No LaTeX compiler found (tectonic, xelatex, or pdflatex)")
    pdf_path = tex_path.with_suffix(".pdf")
    if not pdf_path.is_file():
        raise RuntimeError(f"LaTeX compiler did not create {pdf_path}")
    return pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stem", default="the_blue_room_visual_history_supplement")
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_json = args.json_path.expanduser().resolve()
    data = json.loads(source_json.read_text(encoding="utf-8"))
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else source_json.parent / "supplement"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_image = Path(data["target"]["image"]).expanduser().resolve()
    image_suffix = source_image.suffix.lower() or ".jpg"
    image_path = output_dir / f"the_blue_room_query{image_suffix}"
    shutil.copy2(source_image, image_path)

    tex_path = output_dir / f"{args.stem}.tex"
    tex_path.write_text(
        build_document(data, image_path.name, source_json),
        encoding="utf-8",
    )
    print(f"LaTeX: {tex_path}")
    if args.compile:
        print(f"PDF: {compile_tex(tex_path)}")


if __name__ == "__main__":
    main()
