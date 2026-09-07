"""Predicted modes across the assigned persistence range of a linker class.

The fitted parameters are held; only the assigned input changes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from span_editor.io import load_assigned_inputs, load_parameters, read_corpus
from span_editor.model import SpanModel, build_context
from span_editor.tether import TetherSampler


def observed_mode(entry) -> int | None:
    if entry.observed is not None:
        return int(entry.observed.mode)
    if entry.observed_window:
        return int(round(0.5 * sum(entry.observed_window)))
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--composition", default="xten")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--out", type=Path, default=Path("results/composition_sensitivity.json")
    )
    args = parser.parse_args(argv)

    assigned = load_assigned_inputs(args.assigned)
    parameters = load_parameters(args.params)
    corpus = [
        entry
        for entry in read_corpus(args.corpus)
        if (entry.observed is not None or entry.observed_window)
        and entry.linker_composition == args.composition
    ]
    composition = assigned.compositions[args.composition]
    low, high = composition.persistence_range
    values = list(np.linspace(float(low), float(high), args.steps))
    if float(composition.persistence) not in values:
        values.append(float(composition.persistence))
    values = sorted(set(round(v, 2) for v in values))

    print(f"{args.composition}: assigned {composition.persistence} angstrom, "
          f"band {low} to {high}")
    print(f"{len(corpus)} entries with a measurement use this class")
    print(f"walking {values}")
    print()

    modes: dict[str, dict[float, int]] = {}
    for persistence in values:
        swapped = dict(assigned.compositions)
        swapped[args.composition] = replace(composition, persistence=persistence)
        sampler = TetherSampler(
            compositions=swapped,
            n_chains=assigned.sampling.get("tether_chains", 20000),
            seed=assigned.sampling.get("seed", 0),
            target_surviving=assigned.sampling.get("target_surviving", 0),
        )
        model = SpanModel(sampler)
        for entry in corpus:
            try:
                context = build_context(entry, assigned, args.structures)
                profile = model.predict(context, parameters).normalised()
            except (ValueError, FileNotFoundError, RuntimeError):
                continue
            if profile.degenerate:
                continue
            modes.setdefault(entry.entry_id, {})[persistence] = int(profile.mode)
        print(f"  {persistence:5.2f} angstrom done")

    rows = []
    for entry in corpus:
        seen = modes.get(entry.entry_id, {})
        if len(seen) < 2:
            continue
        found = list(seen.values())
        rows.append(
            {
                "entry_id": entry.entry_id,
                "observed_mode": observed_mode(entry),
                "modes": {str(k): v for k, v in sorted(seen.items())},
                "mode_spread": max(found) - min(found),
            }
        )

    print()
    print(f"{'entry':38s} {'obs':>4s}  " + " ".join(f"{v:5.2f}" for v in values) + "  spread")
    for row in rows:
        line = " ".join(
            f"{row['modes'][str(v)]:5d}" if str(v) in row["modes"] else "    ." for v in values
        )
        print(f"{row['entry_id']:38s} {str(row['observed_mode']):>4s}  {line}  {row['mode_spread']:6d}")

    spreads = [row["mode_spread"] for row in rows]
    print()
    if spreads:
        print(
            f"across the band the mode moves by a median of {int(np.median(spreads))} "
            f"nucleotides, at most {max(spreads)}; "
            f"{sum(1 for v in spreads if v == 0)} of {len(spreads)} entries do not move"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "composition": args.composition,
                "assigned": float(composition.persistence),
                "band": [float(low), float(high)],
                "values": values,
                "entries": rows,
                "median_mode_spread": int(np.median(spreads)) if spreads else None,
                "max_mode_spread": int(max(spreads)) if spreads else None,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
