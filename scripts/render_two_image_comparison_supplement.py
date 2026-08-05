#!/usr/bin/env python3
"""Render a two-image global-atom comparison artifact as a LaTeX supplement."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

from render_painting_description_supplement import (
    callout,
    compile_tex,
    description_list,
    evidence_text,
    humanize,
    itemize,
    latex_escape,
    paragraph,
    section,
)


def image_label(path: str | Path) -> str:
    return Path(path).stem.replace("_", " ").replace("-", " ").strip()


def render_method(method: dict[str, Any]) -> str:
    rows = [(humanize(key), value) for key, value in method.items()]
    return description_list(rows)


def render_quantitative(data: dict[str, Any]) -> str:
    metrics = data["evidence"]["quantitative_similarity"]
    rows = [
        ("Active atoms, A", int(metrics["active_atoms_A"])),
        ("Active atoms, B", int(metrics["active_atoms_B"])),
        ("Shared atoms", int(metrics["shared_atoms"])),
        ("A-specific atoms", int(metrics["A_specific_atoms"])),
        ("B-specific atoms", int(metrics["B_specific_atoms"])),
        ("Shared coefficient mass, A", f"{metrics['shared_mass_A_percent']:.2f}%"),
        ("Shared coefficient mass, B", f"{metrics['shared_mass_B_percent']:.2f}%"),
        ("Weighted Jaccard similarity", f"{metrics['weighted_jaccard']:.4f}"),
        ("Sparse-code cosine similarity", f"{metrics['sparse_code_cosine']:.4f}"),
    ]
    return description_list(rows)


def atom_table(records: list[dict[str, Any]], caption: str, label: str) -> str:
    lines = [
        r"\begin{longtable}{@{}r r r r p{0.46\textwidth}@{}}",
        rf"\caption{{{latex_escape(caption)}}}\label{{{label}}}\\",
        r"\toprule",
        r"Atom & A (\%) & B (\%) & A$-$B & Corpus-derived label\\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Atom & A (\%) & B (\%) & A$-$B & Corpus-derived label\\",
        r"\midrule",
        r"\endhead",
    ]
    for record in records:
        lines.append(
            f"{record['atom_id']} & "
            f"{record.get('importance_A_percent', 0):.3f} & "
            f"{record.get('importance_B_percent', 0):.3f} & "
            f"{record.get('importance_difference_A_minus_B', 0):+.3f} & "
            f"{latex_escape(record['label'])}\\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines) + "\n"


def render_atom_profiles(
    records: list[dict[str, Any]],
    *,
    heading_prefix: str,
) -> str:
    out: list[str] = []
    for record in records:
        out.append(section(f"Atom {record['atom_id']}: {record['label']}", 2))
        out.append(
            description_list(
                [
                    ("A coefficient mass", f"{record.get('importance_A_percent', 0):.3f}%"),
                    ("B coefficient mass", f"{record.get('importance_B_percent', 0):.3f}%"),
                    (
                        "Difference, A minus B",
                        f"{record.get('importance_difference_A_minus_B', 0):+.3f} percentage points",
                    ),
                    ("Evidence class", heading_prefix),
                ]
            )
        )
        out.append(paragraph(record["painterly_description"]))
    return "".join(out)


def render_surface_record(surface: dict[str, Any], image_name: str) -> str:
    out: list[str] = [section(image_name, 2)]
    out.append(description_list([("Craquelure", surface.get("craquelure"))]))
    out.append(section("Paint surface and support", 3))
    out.append(itemize(surface.get("paint_surface_and_support", [])))
    out.append(section("Condition and restoration", 3))
    out.append(itemize(surface.get("condition_and_restoration", [])))
    out.append(section("Imaging artefacts", 3))
    out.append(itemize(surface.get("imaging_artifacts", [])))
    return "".join(out)


def render_comparative_surface(data: dict[str, Any]) -> str:
    surface = data["evidence"]["comparative_surface_inspection"]
    out: list[str] = []
    out.append(render_surface_record(surface["image_A_surface"], "Image A"))
    out.append(render_surface_record(surface["image_B_surface"], "Image B"))
    out.append(section("Comparative age evidence", 2))
    out.append(paragraph(surface["comparative_age_evidence"]))
    out.append(section("Material correspondences", 2))
    out.append(itemize(surface.get("material_correspondences", [])))
    out.append(section("Material differences", 2))
    out.append(itemize(surface.get("material_differences", [])))
    out.append(
        callout(
            "Significance for the copy question",
            paragraph(surface["significance_for_copy_question"]),
            "warningbox",
        )
    )
    out.append(description_list([("Confidence", surface.get("confidence"))]))
    out.append(section("Surface-assessment limitations", 2))
    out.append(itemize(surface.get("limitations", [])))
    return "".join(out)


def render_metadata_context(record: dict[str, Any]) -> str:
    matches = (
        record.get("nearest_embedding_cosine_metadata_matches")
        or record.get("nearest_atom_cosine_metadata_matches")
    )
    if not matches:
        match = record.get(
            "nearest_embedding_cosine_metadata_match",
            record.get("nearest_atom_cosine_metadata_match"),
        )
        matches = [match] if match else []
    hypotheses = record["tested_metadata_hypotheses"]
    out: list[str] = []
    uses_original_embedding = bool(matches) and (
        matches[0].get("retrieval_features") == "raw_whole_2304d_pooled_embedding"
    )
    heading = (
        "Original-embedding cosine metadata neighbors"
        if uses_original_embedding
        else "Atom-cosine metadata neighbors"
    )
    out.append(section(heading, 3))
    for match in matches:
        out.append(section(f"Neighbor {match.get('rank', '?')}", 3))
        out.append(
            description_list(
                [
                    ("Painting", match.get("painting_name")),
                    ("Artist", match.get("artist")),
                    ("Date", match.get("date")),
                    ("Culture", match.get("culture")),
                    ("Region", match.get("region")),
                    ("Period", match.get("art_historical_time_period")),
                    ("Cosine similarity", f"{match.get('cosine_similarity', 0):.6f}"),
                    ("Query active atoms", match.get("query_active_atom_count")),
                    ("Retrieval features", match.get("retrieval_features")),
                ]
            )
        )
    out.append(section("Tested metadata hypotheses", 3))
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
                ("Neighbor consensus", hypotheses.get("neighbor_consensus_summary")),
            ]
        )
    )
    out.append(r"\paragraph{Compatibility assessment.}" + "\n")
    out.append(paragraph(hypotheses.get("compatibility_observation", "")))
    out.append(r"\paragraph{Rejected neighbour details.}" + "\n")
    out.append(itemize(hypotheses.get("rejected_l2_details", []), empty="None recorded."))
    out.append(r"\paragraph{Limitations.}" + "\n")
    out.append(itemize(hypotheses.get("limitations", [])))
    return "".join(out)


def render_atom_neighbor_scene_evidence(record: dict[str, Any]) -> str:
    matches = record.get("supplementary_atom_cosine_matches", [])
    vision = record.get("blind_atom_neighbor_scene_inspection", {})
    if not matches and not vision:
        return ""
    out: list[str] = [section("Supplementary atom-neighbor scene evidence", 3)]
    for match in matches:
        out.append(
            description_list(
                [
                    ("Rank", match.get("rank")),
                    ("Painting", match.get("painting_name")),
                    ("Artist", match.get("artist")),
                    ("Date", match.get("date")),
                    ("Atom cosine", f"{match.get('cosine_similarity', 0):.6f}"),
                ]
            )
        )
    out.append(
        callout(
            "Blind GPT-5.1 scene-type consensus",
            paragraph(vision.get("scene_type_consensus", "Not assessed.")),
        )
    )
    out.append(section("Recurring visual structure", 3))
    out.append(itemize(vision.get("recurring_visual_structure", [])))
    out.append(section("Conflicts across atom neighbors", 3))
    out.append(itemize(vision.get("conflicts_across_neighbors", [])))
    out.append(section("Relevance to atom retrieval", 3))
    out.append(paragraph(vision.get("relevance_to_atom_retrieval", "")))
    out.append(section("Transfer guardrail", 3))
    out.append(paragraph(vision.get("transfer_guardrail", "")))
    out.append(description_list([("Confidence", vision.get("confidence"))]))
    return "".join(out)


def render_direct_vision_compact(vision: dict[str, Any]) -> str:
    principal = vision["principal_scene_figures"]
    constraints = vision["subject_constraints"]
    surface = vision["surface_condition"]
    out: list[str] = []
    out.append(
        callout(
            "Binding figure constraint",
            paragraph(constraints["constraint_summary"]),
            "warningbox",
        )
    )
    out.append(section("Principal figures and relationships", 3))
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
    out.append(section("Secondary images, setting and objects", 3))
    secondary = [
        f"{entry.get('kind', 'uncertain').capitalize()} ({entry.get('count', 0)}): "
        f"{entry.get('location', '')}"
        for entry in vision.get("secondary_human_images", [])
    ]
    out.append(itemize(secondary))
    out.append(itemize(vision.get("setting_and_objects", [])))
    out.append(section("Formal observations", 3))
    out.append(itemize(vision.get("formal_observations", [])))
    out.append(section("Surface observations", 3))
    out.append(
        description_list(
            [
                ("Craquelure visibility", surface["craquelure"].get("visibility")),
                ("Pattern, scale and distribution", surface["craquelure"].get("pattern_scale_distribution")),
                ("Confidence", surface["craquelure"].get("confidence")),
            ]
        )
    )
    out.append(r"\paragraph{Paint and support.}" + "\n")
    out.append(itemize(surface.get("paint_and_support_observations", [])))
    out.append(r"\paragraph{Possible age indicators.}" + "\n")
    out.append(itemize(surface.get("possible_age_indicators", [])))
    out.append(r"\paragraph{Alternative explanations and imaging artefacts.}" + "\n")
    out.append(itemize(surface.get("possible_restoration_or_imaging_artifacts", [])))
    out.append(callout("Condition caution", paragraph(surface.get("condition_caution", "")), "warningbox"))
    out.append(section("Subject boundaries and uncertainties", 3))
    out.append(r"\paragraph{Compatible broad subjects.}" + "\n")
    out.append(itemize(constraints.get("compatible_broad_subjects", [])))
    out.append(r"\paragraph{Incompatible subject features.}" + "\n")
    out.append(itemize(constraints.get("incompatible_subject_features", [])))
    out.append(r"\paragraph{Visual uncertainties.}" + "\n")
    out.append(itemize(vision.get("uncertainties", [])))
    return "".join(out)


def render_individual_analysis(data: dict[str, Any], key: str, label: str) -> str:
    record = data["evidence"][key]
    out: list[str] = [section(label, 2)]
    out.append(section("Atom-only formal description", 3))
    out.append(
        paragraph(
            "This account was generated from the painting's weighted atom evidence before "
            "direct vision or nearest-neighbour context was introduced."
        )
    )
    out.append(callout("Atom-only account", paragraph(record["atom_only_formal_description"])))
    out.append(section("Direct visual observations", 3))
    out.append(render_direct_vision_compact(record["direct_visual_observations"]))
    out.append(render_metadata_context(record))
    out.append(render_atom_neighbor_scene_evidence(record))
    out.append(section("Metadata-grounded formal description", 3))
    out.append(callout("Grounded account", paragraph(record["metadata_grounded_formal_description"])))
    return "".join(out)


def render_shared_interpretation(data: dict[str, Any]) -> str:
    records = data["assessment"]["shared_atom_interpretation"]
    out: list[str] = []
    for record in records:
        ids = ", ".join(str(value) for value in record.get("atom_ids", []))
        out.append(section(f"Atom evidence: {ids}", 2))
        out.append(description_list([("Weight comparison", record.get("weight_comparison"))]))
        out.append(paragraph(record.get("art_historical_significance", "")))
    return "".join(out)


def render_different_interpretation(data: dict[str, Any]) -> str:
    records = data["assessment"]["different_atom_interpretation"]
    out: list[str] = []
    for key, heading in (("image_A", "Image A"), ("image_B", "Image B")):
        out.append(section(heading, 2))
        for record in records.get(key, []):
            ids = ", ".join(str(value) for value in record.get("atom_ids", []))
            out.append(rf"\paragraph{{Atom evidence {latex_escape(ids)}.}}" + "\n")
            out.append(paragraph(record.get("significance", "")))
    return "".join(out)


def render_same_hand_assessment(record: dict[str, Any]) -> str:
    diagnostic = record.get("diagnostic_feature_comparison", {})
    out: list[str] = [
        description_list(
            [
                ("Assessment", record.get("same_hand_assessment")),
                ("Confidence", record.get("confidence")),
            ]
        ),
        callout("Qualified conclusion", paragraph(record.get("summary", "")), "warningbox"),
        section("Evidence consistent with the same hand", 2),
        itemize(record.get("evidence_for_same_hand", []), empty="None recorded."),
        section("Evidence consistent with different hands", 2),
        itemize(record.get("evidence_for_different_hands", []), empty="None recorded."),
        section("Diagnostic feature comparison", 2),
        description_list(
            [(humanize(key), value) for key, value in diagnostic.items()]
        ),
        section("Workshop or collaboration alternatives", 2),
        itemize(record.get("workshop_or_collaboration_alternatives", [])),
        section("Confounds", 2),
        itemize(record.get("confounds", [])),
        section("Required follow-up", 2),
        itemize(record.get("required_follow_up", [])),
    ]
    return "".join(out)


def render_guardrails() -> str:
    return (
        callout(
            "Comparative reasoning task",
            paragraph(
                "Compare how each picture is constructed and how each transforms the shared "
                "design before assessing whether either is likely to be a copy. Treat weighted "
                "atom correspondences, image-specific atoms, direct visual observations, spatial "
                "organization, and surface condition as distinct evidence classes. State copy "
                "direction only as a qualified hypothesis and report counterevidence."
            ),
        )
        + section("Binding guardrails", 2)
        + itemize(
            [
                "Both images are encoded independently against the same fixed global dictionary.",
                "Atom coefficients describe reconstruction contribution and are not attribution probabilities.",
                "Shared atoms establish common formal tendencies; they do not by themselves establish common authorship.",
                "Image-specific atoms must be interpreted as asymmetries, not automatically as evidence of priority.",
                "Nearest-neighbour metadata provides provisional subject and period context and cannot be used as attribution.",
                "Direct vision constrains gross visible content but does not replace weighted atom evidence.",
                "Craquelure, varnish and apparent surface age are supporting but non-determinative evidence.",
                "Differences in photography, color balance, glare and resolution must remain live alternative explanations.",
                "A copy-direction conclusion must include confidence, strongest counterevidence, alternatives and limitations.",
                "Absence of technical imaging prevents definitive claims about underdrawing, pentimenti, support and layer structure.",
            ]
        )
    )


def build_document(
    data: dict[str, Any],
    *,
    image_a_filename: str,
    image_b_filename: str,
    source_json: Path,
) -> str:
    data = dict(data)
    assessment = dict(data["assessment"])
    nested_assessment = assessment.get("surface_and_age_assessment", {})
    for key in (
        "shared_atom_interpretation",
        "different_atom_interpretation",
        "atom_cosine_context_comparison",
        "embedding_context_comparison",
        "atom_neighbor_scene_comparison",
        "strongest_counterevidence",
        "alternative_explanations",
        "limitations",
    ):
        if key not in assessment and key in nested_assessment:
            assessment[key] = nested_assessment[key]
    data["assessment"] = assessment
    images = data["images"]
    today = date.today().isoformat()
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
\setlength{\emergencystretch}{2em}
\setlist[itemize]{leftmargin=1.5em,itemsep=0.25em,topsep=0.25em}
\renewcommand{\arraystretch}{1.16}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small Visual History Agent}
\fancyhead[R]{\small Two-image global-atom comparison}
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
        r"\begin{center}" + "\n"
        r"{\LARGE\bfseries Visual History Agent\par}" + "\n"
        r"\vspace{0.35em}" + "\n"
        r"{\Large Two-image global-atom copy comparison\par}" + "\n"
        r"\vspace{0.5em}" + "\n"
        rf"{{\small Report generated from the frozen comparison artifact on {latex_escape(today)}\par}}"
        + "\n"
        r"\end{center}" + "\n\n"
    )
    out.append(
        r"\begin{figure}[h!]\centering" + "\n"
        r"\begin{minipage}[t]{0.48\textwidth}\centering" + "\n"
        rf"\includegraphics[width=\linewidth,height=0.43\textheight,keepaspectratio]{{{latex_escape(image_a_filename)}}}\\"
        + "\n"
        rf"\textbf{{Image A}}\\{{\small {latex_escape(image_label(images['A']))}}}"
        + "\n"
        r"\end{minipage}\hfill"
        + "\n"
        r"\begin{minipage}[t]{0.48\textwidth}\centering" + "\n"
        rf"\includegraphics[width=\linewidth,height=0.43\textheight,keepaspectratio]{{{latex_escape(image_b_filename)}}}\\"
        + "\n"
        rf"\textbf{{Image B}}\\{{\small {latex_escape(image_label(images['B']))}}}"
        + "\n"
        r"\end{minipage}"
        + "\n"
        r"\caption{The two query images compared independently against the same fixed global atom dictionary.}"
        + "\n"
        r"\end{figure}"
        + "\n"
    )
    out.append(
        callout(
            "Evidence key",
            description_list(
                [
                    ("Shared atoms", "Positive Elastic Net coefficients in both paintings."),
                    ("Image-specific atoms", "Positive coefficient in one painting and zero in the other."),
                    ("Direct vision", "Conservative observations from each supplied reproduction."),
                    ("Context", "Visually tested atom-cosine nearest-neighbour metadata."),
                ]
            ),
        )
    )
    out.append(r"\tableofcontents\clearpage" + "\n")

    out.append(section("Comparison record"))
    out.append(
        description_list(
            [
                ("Image A", Path(images["A"]).name),
                ("Image B", Path(images["B"]).name),
                ("Source artifact", source_json.name),
            ]
        )
    )
    out.append(section("Method configuration"))
    out.append(render_method(data["method"]))
    out.append(section("Comparative prompt and guardrails"))
    out.append(render_guardrails())

    out.append(section("Executive assessment"))
    out.append(
        description_list(
            [
                ("Relationship", assessment.get("relationship_assessment")),
                ("Copy direction", assessment.get("copy_direction")),
                ("Confidence", assessment.get("confidence")),
            ]
        )
    )
    out.append(callout("Conclusion", paragraph(assessment["conclusion"])))
    if data.get("same_hand_assessment"):
        out.append(section("Same-hand assessment"))
        out.append(render_same_hand_assessment(data["same_hand_assessment"]))

    out.append(section("Quantitative comparison"))
    out.append(render_quantitative(data))
    out.append(
        callout(
            "Interpretive boundary",
            paragraph(
                "Cosine similarity measures alignment of the complete sparse coefficient "
                "vectors. Weighted Jaccard measures overlap in normalized coefficient mass. "
                "Neither quantity is a calibrated probability of common authorship or copying."
            ),
            "warningbox",
        )
    )

    evidence = data["evidence"]
    out.append(section("Shared and differentiating atom evidence"))
    out.append(section("Shared atoms", 2))
    out.append(
        atom_table(
            evidence["shared_atoms"],
            "Atoms active in both paintings.",
            "tab:shared",
        )
    )
    out.append(section("Image A-specific atoms", 2))
    out.append(
        atom_table(
            evidence["image_A_specific_atoms"],
            "Atoms active only in Image A.",
            "tab:a-specific",
        )
    )
    out.append(section("Image B-specific atoms", 2))
    out.append(
        atom_table(
            evidence["image_B_specific_atoms"],
            "Atoms active only in Image B.",
            "tab:b-specific",
        )
    )

    out.append(section("Comparative art-historical analysis"))
    out.append(paragraph(assessment["comparative_art_historical_analysis"]))

    out.append(section("Comparative surface and condition assessment"))
    out.append(render_comparative_surface(data))

    out.append(section("Independent painting analyses"))
    out.append(render_individual_analysis(data, "image_A", "Image A"))
    out.append(render_individual_analysis(data, "image_B", "Image B"))

    out.append(section("Interpretation of shared atoms"))
    out.append(render_shared_interpretation(data))

    out.append(section("Interpretation of differentiating atoms"))
    out.append(render_different_interpretation(data))

    context_comparison = assessment.get(
        "embedding_context_comparison",
        assessment.get("atom_cosine_context_comparison", ""),
    )
    context_heading = (
        "Original-embedding context comparison"
        if assessment.get("embedding_context_comparison")
        else "Atom-cosine context comparison"
    )
    out.append(section(context_heading))
    out.append(paragraph(context_comparison))
    if assessment.get("atom_neighbor_scene_comparison"):
        out.append(section("Atom-neighbor scene comparison"))
        out.append(paragraph(assessment["atom_neighbor_scene_comparison"]))

    out.append(section("Counterevidence and alternative explanations"))
    out.append(section("Strongest counterevidence", 2))
    out.append(itemize(assessment.get("strongest_counterevidence", [])))
    out.append(section("Alternative explanations", 2))
    out.append(itemize(assessment.get("alternative_explanations", [])))
    out.append(section("Limitations", 2))
    out.append(itemize(assessment.get("limitations", [])))

    out.append(r"\appendix" + "\n")
    out.append(section("Complete shared-atom profiles"))
    out.append(
        paragraph(
            "These corpus-level atom descriptions were generated independently of this pair. "
            "Artist and subject references characterize retrieved evidence and are not "
            "attributions of either query image."
        )
    )
    out.append(render_atom_profiles(evidence["shared_atoms"], heading_prefix="shared"))
    out.append(section("Complete Image A-specific atom profiles"))
    out.append(render_atom_profiles(evidence["image_A_specific_atoms"], heading_prefix="A-specific"))
    out.append(section("Complete Image B-specific atom profiles"))
    out.append(render_atom_profiles(evidence["image_B_specific_atoms"], heading_prefix="B-specific"))

    metadata_retrieval = str(data["method"].get("metadata_retrieval", ""))
    uses_original_embedding = "2304-D embeddings" in metadata_retrieval
    out.append(section("Reproducibility and interpretive boundaries"))
    out.append(
        itemize(
            [
                "Both images were encoded independently against the same fixed dictionary.",
                "Sparse coding used uncapped, non-negative Elastic Net.",
                "Shared and image-specific atoms were defined from the resulting positive coefficients.",
                f"Metadata retrieval used {metadata_retrieval.rstrip('.')}.",
                (
                    "Exact original-embedding self-matches were excluded."
                    if uses_original_embedding
                    else "Raw pooled-embedding context was excluded."
                ),
                "Atom-cosine neighbors were retained as a supplementary scene-type channel and inspected blindly by coordinator vision when their image files were available.",
                "Atom-neighbor scene evidence was not used as attribution, date, priority, or copy-direction proof.",
                "Metadata neighbours were not used as attribution evidence.",
                "Direct visual observations were binding only for clearly visible content.",
                "Surface condition and craquelure were treated as supporting, non-determinative evidence.",
                "The comparison is based on digital reproductions and does not replace technical examination.",
            ]
        )
    )
    out.append(r"\end{document}" + "\n")
    return "".join(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stem", default="two_image_global_atom_copy_comparison_supplement")
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

    copied_images: dict[str, Path] = {}
    for key in ("A", "B"):
        source = Path(data["images"][key]).expanduser().resolve()
        suffix = source.suffix.lower() or ".jpg"
        destination = output_dir / f"comparison_image_{key.lower()}{suffix}"
        shutil.copy2(source, destination)
        copied_images[key] = destination

    tex_path = output_dir / f"{args.stem}.tex"
    tex_path.write_text(
        build_document(
            data,
            image_a_filename=copied_images["A"].name,
            image_b_filename=copied_images["B"].name,
            source_json=source_json,
        ),
        encoding="utf-8",
    )
    print(f"LaTeX: {tex_path}")
    if args.compile:
        print(f"PDF: {compile_tex(tex_path)}")


if __name__ == "__main__":
    main()
