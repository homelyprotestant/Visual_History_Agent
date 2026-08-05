#!/usr/bin/env python3
"""Render a hand-first comparison package as one combined LaTeX/PDF supplement.

Includes comparison assessments plus the full individual evidence reports for A and B.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_painting_description_supplement import (  # noqa: E402
    callout,
    compile_tex,
    description_list,
    evidence_text,
    itemize,
    latex_escape,
    paragraph,
    render_painting_body,
    section,
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _formal_text(payload: dict[str, Any]) -> str:
    return str(
        payload.get("final_formal_description")
        or payload.get("formal_visual_description")
        or ""
    ).strip()


def _image_label(path: str | Path) -> str:
    return Path(path).stem.replace("_", " ").replace("-", " ").strip()


def _legend(desc_a: dict[str, Any], desc_b: dict[str, Any]) -> dict[str, Any]:
    return (
        desc_a.get("citation_legend")
        or desc_b.get("citation_legend")
        or {
            "Atoms": "Corpus-derived dictionary atom evidence.",
            "Vision": "Direct visual observation of the supplied reproduction.",
            "L2": "Nearest-neighbour metadata context.",
        }
    )


def build_document(
    package: dict[str, Any],
    *,
    image_a_filename: str,
    image_b_filename: str,
    source_json: Path,
) -> str:
    images = package["images"]
    comparison = package["comparison"]
    same_hand = comparison.get("same_hand_assessment") or {}
    copy_assessment = comparison.get("assessment") or {}
    desc_a = package["descriptions"]["A"]
    desc_b = package["descriptions"]["B"]
    legend = _legend(desc_a, desc_b)
    today = date.today().isoformat()
    label_a = _image_label(images["A"])
    label_b = _image_label(images["B"])

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
\fancyhead[R]{\small Hand-first comparison}
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
        r"{\Large Hand-first comparison report\par}" + "\n"
        r"\vspace{0.5em}" + "\n"
        rf"{{\large {latex_escape(label_a)} vs {latex_escape(label_b)}\par}}" + "\n"
        r"\vspace{0.35em}" + "\n"
        rf"{{\small Generated {latex_escape(today)} from {latex_escape(source_json.name)}\par}}"
        + "\n"
        r"\end{center}" + "\n\n"
    )
    out.append(
        r"\begin{figure}[h!]\centering" + "\n"
        r"\begin{minipage}[t]{0.48\textwidth}\centering" + "\n"
        rf"\includegraphics[width=\linewidth,height=0.42\textheight,keepaspectratio]{{{latex_escape(image_a_filename)}}}\\"
        + "\n"
        rf"\textbf{{Image A}}\\{{\small {latex_escape(Path(images['A']).name)}}}"
        + "\n"
        r"\end{minipage}\hfill" + "\n"
        r"\begin{minipage}[t]{0.48\textwidth}\centering" + "\n"
        rf"\includegraphics[width=\linewidth,height=0.42\textheight,keepaspectratio]{{{latex_escape(image_b_filename)}}}\\"
        + "\n"
        rf"\textbf{{Image B}}\\{{\small {latex_escape(Path(images['B']).name)}}}"
        + "\n"
        r"\end{minipage}" + "\n"
        r"\caption{Independent Description Agent runs for each image, then same-hand and copy assessments.}"
        + "\n"
        r"\end{figure}" + "\n"
    )
    out.append(
        callout(
            "How to read this report",
            paragraph(
                "This single document contains the comparative assessments and the complete "
                "individual evidence reports for both paintings. Corpus-derived atom evidence "
                "is marked in blue, direct visual observation in tan, and nearest-neighbour "
                "context in green."
            )
            + description_list(
                [
                    ("Atom evidence", legend.get("Atoms")),
                    ("Direct vision", legend.get("Vision")),
                    ("Context", legend.get("L2")),
                    ("Order", "Same-hand assessment first, then copy relationship."),
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
                ("Coordinator", f"{package.get('coordinator_backend')}/{package.get('coordinator_model')}"),
                ("Source package", source_json.name),
                ("Method", "Independent descriptions; same-hand first; copy conditioned on same-hand"),
            ]
        )
    )

    out.append(section("Executive summary"))
    out.append(
        description_list(
            [
                ("Same-hand assessment", same_hand.get("same_hand_assessment")),
                ("Same-hand confidence", same_hand.get("confidence")),
                ("Formal closeness", same_hand.get("formal_closeness")),
                ("Copy relationship", copy_assessment.get("copy_relationship")),
                ("Copy confidence", copy_assessment.get("confidence")),
            ]
        )
    )
    out.append(callout("Same-hand summary", paragraph(same_hand.get("summary", "")), "warningbox"))
    out.append(callout("Copy summary", paragraph(copy_assessment.get("summary", "")), "warningbox"))

    out.append(section("Same-hand assessment"))
    out.append(
        description_list(
            [
                ("Assessment", same_hand.get("same_hand_assessment")),
                ("Confidence", same_hand.get("confidence")),
                ("Formal closeness", same_hand.get("formal_closeness")),
            ]
        )
    )
    out.append(callout("Summary", paragraph(same_hand.get("summary", "")), "warningbox"))
    out.append(section("Evidence for same hand", 2))
    out.append(itemize(same_hand.get("evidence_for_same_hand", []), empty="None recorded."))
    out.append(section("Evidence for different hands", 2))
    out.append(itemize(same_hand.get("evidence_for_different_hands", []), empty="None recorded."))
    out.append(section("Uncertainties", 2))
    out.append(itemize(same_hand.get("uncertainties", []), empty="None recorded."))
    for key, heading in (
        ("diagnostic_feature_comparison", "Diagnostic feature comparison"),
        ("workshop_or_collaboration_alternatives", "Workshop or collaboration alternatives"),
        ("confounds", "Confounds"),
        ("required_follow_up", "Required follow-up"),
    ):
        value = same_hand.get(key)
        if not value:
            continue
        out.append(section(heading, 2))
        if isinstance(value, dict):
            out.append(description_list([(str(k).replace("_", " ").title(), v) for k, v in value.items()]))
        else:
            out.append(itemize(value if isinstance(value, list) else [value]))

    out.append(section("Copy relationship"))
    out.append(
        description_list(
            [
                ("Relationship", copy_assessment.get("copy_relationship")),
                ("Confidence", copy_assessment.get("confidence")),
            ]
        )
    )
    out.append(callout("Summary", paragraph(copy_assessment.get("summary", "")), "warningbox"))
    if copy_assessment.get("same_hand_conditioning"):
        out.append(section("Same-hand conditioning", 2))
        out.append(paragraph(copy_assessment["same_hand_conditioning"]))
    out.append(section("Evidence for copy or type", 2))
    out.append(
        itemize(copy_assessment.get("evidence_for_copy_or_type", []), empty="None recorded.")
    )
    out.append(section("Evidence against direct copy", 2))
    out.append(
        itemize(
            copy_assessment.get("evidence_against_direct_copy", []),
            empty="None recorded.",
        )
    )
    out.append(section("Uncertainties", 2))
    out.append(itemize(copy_assessment.get("uncertainties", []), empty="None recorded."))

    out.append(section("Side-by-side formal descriptions"))
    out.append(section(f"Image A — {label_a}", 2))
    out.append(
        callout("Final formal description", paragraph(evidence_text(_formal_text(desc_a))))
    )
    out.append(section(f"Image B — {label_b}", 2))
    out.append(
        callout("Final formal description", paragraph(evidence_text(_formal_text(desc_b))))
    )

    out.append(r"\clearpage" + "\n")
    out.append(section(f"Full individual report — Image A ({label_a})"))
    out.append(
        paragraph(
            "The following sections reproduce the complete Description Agent artifact for Image A."
        )
    )
    source_a = Path((package.get("individual_json") or {}).get("A") or source_json)
    out.append(
        render_painting_body(
            desc_a,
            source_json=source_a if source_a.exists() else source_json,
            include_appendix=True,
            title_prefix="A",
        )
    )

    out.append(r"\clearpage" + "\n")
    out.append(section(f"Full individual report — Image B ({label_b})"))
    out.append(
        paragraph(
            "The following sections reproduce the complete Description Agent artifact for Image B."
        )
    )
    source_b = Path((package.get("individual_json") or {}).get("B") or source_json)
    out.append(
        render_painting_body(
            desc_b,
            source_json=source_b if source_b.exists() else source_json,
            include_appendix=True,
            title_prefix="B",
        )
    )

    out.append(r"\end{document}" + "\n")
    return "".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("json_path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stem", default="hand_first_comparison")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    source_json = args.json_path.expanduser().resolve()
    package = _load(source_json)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else source_json.parent / "supplement"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: dict[str, Path] = {}
    for key in ("A", "B"):
        source = Path(package["images"][key]).expanduser().resolve()
        dest = output_dir / f"comparison_image_{key.lower()}{source.suffix.lower() or '.jpg'}"
        shutil.copy2(source, dest)
        copied[key] = dest

    tex_path = output_dir / f"{args.stem}.tex"
    tex_path.write_text(
        build_document(
            package,
            image_a_filename=copied["A"].name,
            image_b_filename=copied["B"].name,
            source_json=source_json,
        ),
        encoding="utf-8",
    )
    print(f"LaTeX: {tex_path}", flush=True)
    if args.compile:
        print(f"PDF: {compile_tex(tex_path)}", flush=True)


if __name__ == "__main__":
    main()
