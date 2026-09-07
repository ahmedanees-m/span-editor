"""Distance from the sampled displaced-strand path to the deposited one.

Run at residue and atom resolution of the steric map.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from span_editor.io import load_assigned_inputs, load_parameters, read_corpus
from span_editor.model import build_context

NEAR = 6.3  # one nucleotide rise


def score(context, spread: float, distance: float, persistence: float) -> dict:
    context.observed_spread = spread
    context.surface_distance = distance
    substrate = context.substrate(persistence)

    rows = []
    for index, deposited in sorted(context.observed_bubble.items()):
        cloud, weights = substrate.cloud(index)
        weights = weights / weights.sum()
        mean = np.average(cloud, axis=0, weights=weights)
        separations = np.linalg.norm(cloud - deposited, axis=1)
        rows.append(
            {
                "index": int(index),
                "mean_to_deposited": round(float(np.linalg.norm(mean - deposited)), 2),
                "closest": round(float(separations.min()), 2),
                "weight_within_a_rise": round(float(np.sum(weights[separations <= NEAR])), 4),
            }
        )
    if not rows:
        return {"positions": 0}
    return {
        "positions": len(rows),
        "median_mean_to_deposited": round(
            float(np.median([r["mean_to_deposited"] for r in rows])), 2
        ),
        "median_weight_within_a_rise": round(
            float(np.median([r["weight_within_a_rise"] for r in rows])), 4
        ),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--distances", default="0,8,12,16,20,28")
    parser.add_argument("--resolutions", default="residue,atom")
    parser.add_argument(
        "--entries",
        default="kissling-SpCas9-ABE8e-HEK-Plasmid-5d,communbiol-dCas12a-ABE8e",
    )
    parser.add_argument("--out", type=Path, default=Path("results/confinement_check.json"))
    args = parser.parse_args(argv)

    assigned = load_assigned_inputs(args.assigned)
    parameters = load_parameters(args.params)
    corpus = {entry.entry_id: entry for entry in read_corpus(args.corpus)}

    report = {"near": NEAR, "persistence": parameters.ssdna_persistence, "entries": {}}
    for entry_id in args.entries.split(","):
        entry = corpus.get(entry_id)
        if entry is None:
            print(f"{entry_id}: not in the corpus")
            continue
        print(f"{entry_id}, structure {entry.structure_id}")
        print(
            f"{'occupancy':>9s} {'confine':>8s} {'spheres':>9s} "
            f"{'median mean to deposited':>22s} {'median weight within a rise':>25s}"
        )
        arms = []
        for resolution in args.resolutions.split(","):
            inputs = replace(assigned, occupancy_resolution=resolution)
            for distance in (float(v) for v in args.distances.split(",")):
                context = build_context(entry, inputs, args.structures)
                # The resolved coordinates are scored against, never used.
                resolved = dict(context.observed_bubble)
                outcome = score(context, 0.0, distance, parameters.ssdna_persistence)
                outcome["surface_distance"] = distance
                outcome["occupancy_resolution"] = resolution
                outcome["occluding_spheres"] = len(context.occupancy)
                arms.append(outcome)
                if outcome["positions"]:
                    print(
                        f"{resolution:>9s} {distance:8.0f} "
                        f"{outcome['occluding_spheres']:9d} "
                        f"{outcome['median_mean_to_deposited']:22.2f} "
                        f"{outcome['median_weight_within_a_rise']:25.4f}"
                    )
        report["entries"][entry_id] = {
            "structure": entry.structure_id,
            "resolved_bubble_positions": sorted(resolved),
            "arms": arms,
        }
        print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
