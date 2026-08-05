"""CLI for the Visual History Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="visual-history-agent", description="Visual History Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    p_desc = sub.add_parser("describe", help="Describe one painting")
    p_desc.add_argument("--image", type=Path, required=True)
    p_desc.add_argument("--out", type=Path)
    p_desc.add_argument("--pdf", action="store_true")

    p_map = sub.add_parser("map", help="Local-Dai spatial maps (cosine LR → HR224)")
    p_map.add_argument("--image", type=Path, required=True)
    p_map.add_argument("--image-b", type=Path)
    p_map.add_argument("--out", type=Path)
    p_map.add_argument("--atom-ids")
    p_map.add_argument("--top-k", type=int)
    p_map.add_argument("--force-recompute", action="store_true")

    p_cmp = sub.add_parser("compare", help="Compare two paintings (hand-first)")
    p_cmp.add_argument("--a", type=Path, required=True)
    p_cmp.add_argument("--b", type=Path, required=True)
    p_cmp.add_argument("--out", type=Path)
    p_cmp.add_argument("--no-pdf", action="store_false", dest="pdf")
    p_cmp.set_defaults(pdf=True)

    p_fr = sub.add_parser("compare-frozen", help="Compare two frozen assessment JSON files")
    p_fr.add_argument("--a", type=Path, required=True)
    p_fr.add_argument("--b", type=Path, required=True)
    p_fr.add_argument("--out", type=Path)

    args = parser.parse_args(argv)
    if args.command == "describe":
        from visual_history_agent.describe import describe_image
        result = describe_image(args.image, output_dir=args.out, compile_pdf=args.pdf)
    elif args.command == "map":
        from visual_history_agent.maps import map_coefficients
        atom_ids = (
            [int(x) for x in args.atom_ids.split(",") if x.strip()]
            if args.atom_ids
            else None
        )
        result = map_coefficients(
            args.image,
            image_b=args.image_b,
            output_dir=args.out,
            atom_ids=atom_ids,
            top_k=args.top_k,
            force_recompute=args.force_recompute,
        )
        result.pop("bundle", None)
    elif args.command == "compare":
        from visual_history_agent.compare import compare_images
        result = compare_images(args.a, args.b, output_dir=args.out, compile_pdf=args.pdf)
    else:
        from visual_history_agent.compare import compare_frozen_assessments
        result = compare_frozen_assessments(args.a, args.b, output_dir=args.out)

    # Drop bulky nested payloads from stdout
    printable = {k: v for k, v in result.items() if k not in {"payload", "comparison", "meta", "encoding", "individual"}}
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
