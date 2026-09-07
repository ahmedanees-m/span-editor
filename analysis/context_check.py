"""Profile variation across delivery contexts against variation across architectures.

Reports shape distance and centre shift for both.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from span_editor.io import read_corpus


def aligned(first, second) -> tuple[np.ndarray, np.ndarray] | None:
    shared = np.intersect1d(first.indices, second.indices)
    if shared.size < 3:
        return None
    left = first.values[np.isin(first.indices, shared)]
    right = second.values[np.isin(second.indices, shared)]
    return left, right


def shape_distance(first, second) -> float | None:
    """Root mean squared difference between two profiles, each scaled to unit norm."""
    pair = aligned(first, second)
    if pair is None:
        return None
    left, right = pair
    if left.max() <= 0 or right.max() <= 0:
        return None
    left = left / np.linalg.norm(left)
    right = right / np.linalg.norm(right)
    return float(np.sqrt(np.mean((left - right) ** 2)))


def centre(profile) -> float:
    total = float(profile.values.sum())
    return float((profile.indices * profile.values).sum() / total) if total > 0 else float("nan")


def summarise(values: list[float]) -> dict:
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--out", type=Path, default=Path("results/context_check.json"))
    args = parser.parse_args(argv)

    entries = [entry for entry in read_corpus(args.corpus) if entry.observed and entry.context]
    if not entries:
        print("no entries carry both a profile and a context")
        return 1

    by_architecture = defaultdict(dict)
    for entry in entries:
        by_architecture[entry.editor][entry.context] = entry.observed

    within, within_centre = [], []
    for contexts in by_architecture.values():
        for left, right in combinations(sorted(contexts), 2):
            within.append(shape_distance(contexts[left], contexts[right]))
            within_centre.append(abs(centre(contexts[left]) - centre(contexts[right])))

    by_context = defaultdict(dict)
    for entry in entries:
        by_context[entry.context][entry.editor] = entry.observed

    across, across_centre = [], []
    for architectures in by_context.values():
        for left, right in combinations(sorted(architectures), 2):
            across.append(shape_distance(architectures[left], architectures[right]))
            across_centre.append(abs(centre(architectures[left]) - centre(architectures[right])))

    report = {
        "entries": len(entries),
        "architectures": len(by_architecture),
        "contexts": len(by_context),
        "shape": {
            "across_contexts": summarise(within),
            "across_architectures": summarise(across),
        },
        "centre_shift": {
            "across_contexts": summarise(within_centre),
            "across_architectures": summarise(across_centre),
        },
        "peak_by_architecture": {
            editor: sorted({profile.mode for profile in contexts.values()})
            for editor, contexts in by_architecture.items()
        },
    }

    shape_ratio = (
        report["shape"]["across_architectures"]["mean"]
        / report["shape"]["across_contexts"]["mean"]
        if report["shape"]["across_contexts"].get("mean")
        else float("nan")
    )
    report["architecture_over_context"] = shape_ratio

    print(f"{len(entries)} entries, {len(by_architecture)} architectures, {len(by_context)} contexts")
    print()
    print("                       pairs    mean    median     max")
    for label, block in (
        ("shape, same architecture", report["shape"]["across_contexts"]),
        ("shape, same context     ", report["shape"]["across_architectures"]),
        ("centre, same architecture", report["centre_shift"]["across_contexts"]),
        ("centre, same context    ", report["centre_shift"]["across_architectures"]),
    ):
        if block.get("n"):
            print(
                f"{label:26s} {block['n']:5d}  {block['mean']:6.3f}  "
                f"{block['median']:7.3f}  {block['max']:6.3f}"
            )
    print()
    print(f"architecture variance over context variance, in shape: {shape_ratio:.2f}")
    print()
    print("window peak by architecture, across every context measured")
    for editor, peaks in sorted(report["peak_by_architecture"].items()):
        print(f"  {editor:18s} {peaks}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
