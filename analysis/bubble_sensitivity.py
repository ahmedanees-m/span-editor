"""Fit loss under different treatments of the unpaired R-loop nucleotides.

Ranges from pinning every resolved coordinate to ignoring the coordinates
entirely, with intermediate settings that weight sampled conformations by how
closely they follow the deposited path.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from span_editor.fitting import fit_parameters
from span_editor.io import load_assigned_inputs, read_corpus
from span_editor.model import SpanModel, build_context
from span_editor.tether import TetherSampler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--spreads", default="3,6,12,24")
    parser.add_argument("--out", type=Path, default=Path("results/bubble_sensitivity.json"))
    args = parser.parse_args(argv)

    assigned = load_assigned_inputs(args.assigned)
    entries = [entry for entry in read_corpus(args.corpus) if entry.fits_parameters]
    if not entries:
        print("no entry belongs to the fitting subset")
        return 1

    settings = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    bounds = settings["bounds"]
    radii = np.arange(bounds["capture_radius"][0], bounds["capture_radius"][1] + 0.01, 1.0)
    persistences = np.array([6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])
    span = bounds.get("link_alpha", [1.0, 5000.0])
    alphas = np.geomspace(span[0], span[1], 24)

    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=assigned.sampling.get("tether_chains", 20000),
        seed=assigned.sampling.get("seed", 0),
        cache_dir=args.cache,
        target_surviving=assigned.sampling.get("target_surviving", 0),
    )
    model = SpanModel(sampler)

    # Pinning is expressed by declaring no bubble, so the resolved nucleotides
    # go back to being anchors. The others differ only in how loosely the
    # coordinates are believed.
    pinned = {
        accession: {key: value for key, value in settings.items() if key != "bubble"}
        for accession, settings in assigned.structure_geometry.items()
    }

    arms = [("resolved nucleotides pinned", None, pinned)]
    for value in (float(v) for v in args.spreads.split(",")):
        arms.append((f"sampled, coordinates believed to {value:.0f} angstrom", value, None))
    arms.append(("sampled, coordinates ignored", 0.0, None))

    rows = []
    for label, spread, geometry in arms:
        inputs = assigned
        if geometry is not None:
            inputs = replace(assigned, structure_geometry=geometry)
        contexts = []
        for entry in entries:
            try:
                context = build_context(entry, inputs, args.structures)
            except (ValueError, FileNotFoundError) as error:
                print(f"skipped {entry.entry_id}: {error}")
                continue
            if spread is not None:
                context.observed_spread = spread
            contexts.append(context)
        if not contexts:
            continue

        result = fit_parameters(model, contexts, radii, persistences, alphas, n_jobs=args.jobs)
        held = sorted(a.index for a in contexts[0].strand_anchors)
        rows.append(
            {
                "treatment": label,
                "observed_spread": spread,
                "pinned": geometry is not None,
                "held_positions": held,
                "sampled_positions": [
                    i for i in contexts[0].indices if i not in set(held)
                ],
                "capture_radius": result.parameters.capture_radius,
                "ssdna_persistence": result.parameters.ssdna_persistence,
                "link_alpha": round(result.parameters.link_alpha, 3),
                "loss": round(float(result.loss), 6),
            }
        )
        print(
            f"{label:48s} held {len(held):2d}  radius {result.parameters.capture_radius:5.1f}  "
            f"persistence {result.parameters.ssdna_persistence:5.1f}  "
            f"alpha {result.parameters.link_alpha:9.2f}  loss {result.loss:.5f}"
        )

    best = min(rows, key=lambda row: row["loss"])
    print()
    print(f"best loss under {best['treatment']}, {best['loss']}")
    ratios = {
        row["treatment"]: round(row["loss"] / best["loss"], 2) for row in rows
    }
    for treatment, ratio in ratios.items():
        print(f"  {treatment:48s} {ratio:6.2f} times the best")

    report = {
        "arms": rows,
        "best": best["treatment"],
        "loss_ratio_to_best": ratios,
        "fitting_entries": [entry.entry_id for entry in entries],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
