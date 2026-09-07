"""Signed mode offsets by editor and by architecture, ranked.

Amplitude residuals are restricted to entries measured in the same assay.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, pstdev

from span_editor.io import read_corpus


def centre(span) -> float:
    return 0.5 * (span[0] + span[1])


def collect(evaluation: Path, transfer: Path, corpus_path: Path) -> list[dict]:
    """Signed mode offsets from both scoring runs, joined to the corpus."""
    by_id = {entry.entry_id: entry for entry in read_corpus(corpus_path)}
    rows = []

    if evaluation.exists():
        for row in json.loads(evaluation.read_text("utf-8"))["entries"]:
            entry = by_id.get(row["entry_id"])
            if entry is None:
                continue
            rows.append(
                {
                    "entry_id": row["entry_id"],
                    "source_key": row["source_key"],
                    "editor": entry.editor,
                    "effector": entry.effector,
                    "ortholog": entry.ortholog,
                    "context": entry.context,
                    "anchor_site": entry.anchor_site,
                    "label": entry.label,
                    "offset": int(row["mode_offset"]),
                    "predicted_width": row.get("predicted_width"),
                    "observed_width": row.get("observed_width"),
                    "from": "profile",
                }
            )

    if transfer.exists():
        for row in json.loads(transfer.read_text("utf-8"))["entries"]:
            entry = by_id.get(row["entry_id"])
            if entry is None or not row.get("observed_window"):
                continue
            rows.append(
                {
                    "entry_id": row["entry_id"],
                    "source_key": row["source_key"],
                    "editor": entry.editor,
                    "effector": entry.effector,
                    "ortholog": entry.ortholog,
                    "context": entry.context,
                    "anchor_site": entry.anchor_site,
                    "label": entry.label,
                    "offset": int(round(centre(row["window"]) - centre(row["observed_window"]))),
                    "predicted_width": row.get("width"),
                    "observed_width": entry.observed_width or None,
                    "from": "window",
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--evaluation", type=Path, default=Path("results/evaluation.json"))
    parser.add_argument("--transfer", type=Path, default=Path("results/anchor_transfer.json"))
    parser.add_argument("--out", type=Path, default=Path("results/residuals.json"))
    args = parser.parse_args(argv)

    rows = collect(args.evaluation, args.transfer, args.corpus)
    if not rows:
        print("no scored entries found; run the evaluation and the transfer scoring first")
        return 1

    ranked = sorted(rows, key=lambda row: (-abs(row["offset"]), row["entry_id"]))

    by_editor = defaultdict(list)
    for row in rows:
        by_editor[row["editor"]].append(row["offset"])

    hyperactive = {
        editor: values
        for editor, values in by_editor.items()
        if "8e" in editor.replace(" ", "").lower()
    }
    others = {editor: values for editor, values in by_editor.items() if editor not in hyperactive}

    def described(group):
        return [
            {
                "editor": editor,
                "n": len(values),
                "mean": round(fmean(values), 2),
                "spread": round(pstdev(values), 2) if len(values) > 1 else 0.0,
                "all_one_way": len({v > 0 for v in values if v != 0}) <= 1,
            }
            for editor, values in sorted(group.items())
        ]

    widths = [
        row
        for row in rows
        if row.get("predicted_width") and row.get("observed_width")
    ]
    width_error = [row["predicted_width"] - row["observed_width"] for row in widths]

    print(f"{len(rows)} scored entries from {len({row['source_key'] for row in rows})} sources")
    print()
    print(f"{'entry':32s} {'editor':22s} {'anchor':12s} {'offset':>7s} {'from':>8s}")
    for row in ranked:
        print(
            f"{row['entry_id']:32s} {row['editor']:22s} {row['anchor_site']:12s} "
            f"{row['offset']:+7d} {row['from']:>8s}"
        )
    print()
    offsets = [row["offset"] for row in rows]
    print(f"mean signed offset {fmean(offsets):+.2f}, spread {pstdev(offsets):.2f}")
    print(f"offsets within one nucleotide: {sum(1 for v in offsets if abs(v) <= 1)} of {len(offsets)}")
    print()
    print("per architecture, hyperactive deaminases")
    for item in described(hyperactive):
        print(
            f"  {item['editor']:24s} n {item['n']:3d}  mean {item['mean']:+.2f}  "
            f"spread {item['spread']:.2f}  one way {item['all_one_way']}"
        )
    print("per architecture, the rest")
    for item in described(others):
        print(
            f"  {item['editor']:24s} n {item['n']:3d}  mean {item['mean']:+.2f}  "
            f"spread {item['spread']:.2f}  one way {item['all_one_way']}"
        )
    print()
    if width_error:
        print(f"width, predicted minus observed, over {len(width_error)} entries")
        print(f"  mean {fmean(width_error):+.2f}, spread {pstdev(width_error):.2f}")
        print(f"  exact {sum(1 for v in width_error if v == 0)}, "
              f"within one {sum(1 for v in width_error if abs(v) <= 1)}")
    else:
        print("no entry carries both a predicted and an observed width")

    report = {
        "entries": rows,
        "ranked": [row["entry_id"] for row in ranked],
        "mean_signed_offset": round(fmean(offsets), 3),
        "spread": round(pstdev(offsets), 3),
        "within_one": sum(1 for v in offsets if abs(v) <= 1),
        "hyperactive": described(hyperactive),
        "others": described(others),
        "width_error": {
            "n": len(width_error),
            "mean": round(fmean(width_error), 3) if width_error else None,
            "spread": round(pstdev(width_error), 3) if len(width_error) > 1 else None,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
