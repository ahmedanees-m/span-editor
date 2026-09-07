"""Predicted modes under each fitting subset's parameters."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from span_editor.io import load_assigned_inputs, read_corpus
from span_editor.model import SpanModel, build_context
from span_editor.occupancy import ModelParameters
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
    parser.add_argument(
        "--sensitivity", type=Path, default=Path("results/fit_sensitivity.json")
    )
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--out", type=Path, default=Path("results/mode_stability.json"))
    args = parser.parse_args(argv)

    if not args.sensitivity.exists():
        print(f"{args.sensitivity} not found; run analysis/fit_sensitivity.py first")
        return 1
    sensitivity = json.loads(args.sensitivity.read_text(encoding="utf-8"))
    subsets = sensitivity["subsets"]

    assigned = load_assigned_inputs(args.assigned)
    corpus = [
        entry
        for entry in read_corpus(args.corpus)
        if entry.observed is not None or entry.observed_window
    ]

    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=assigned.sampling.get("tether_chains", 20000),
        seed=assigned.sampling.get("seed", 0),
        target_surviving=assigned.sampling.get("target_surviving", 0),
    )
    model = SpanModel(sampler)

    contexts = {}
    for entry in corpus:
        try:
            contexts[entry.entry_id] = build_context(entry, assigned, args.structures)
        except (ValueError, FileNotFoundError) as error:
            print(f"skipped {entry.entry_id}: {error}")

    modes = defaultdict(dict)
    for subset in subsets:
        parameters = ModelParameters(
            capture_radius=subset["capture_radius"],
            ssdna_persistence=subset["ssdna_persistence"],
            link_alpha=subset["link_alpha"],
        )
        label = subset["subset"]
        print(f"--- {label}, n={subset['n_entries']}, {parameters.as_dict()}")
        for entry in corpus:
            context = contexts.get(entry.entry_id)
            if context is None:
                continue
            profile = model.predict(context, parameters).normalised()
            modes[entry.entry_id][label] = {
                "mode": int(profile.mode),
                "window": list(profile.window(0.5)),
                "degenerate": bool(profile.degenerate),
            }

    labels = [subset["subset"] for subset in subsets]
    rows = []
    for entry in corpus:
        seen = modes.get(entry.entry_id)
        if not seen:
            continue
        values = [seen[label]["mode"] for label in labels if label in seen]
        rows.append(
            {
                "entry_id": entry.entry_id,
                "ortholog": entry.ortholog,
                "source_key": entry.source_key,
                "observed_mode": observed_mode(entry),
                "modes": {label: seen[label] for label in labels if label in seen},
                "mode_range": [min(values), max(values)],
                "mode_spread": max(values) - min(values),
            }
        )

    print()
    print(f"{'entry':32s} {'obs':>4s} " + " ".join(f"{i:>5d}" for i in range(len(labels))) + "  spread")
    for row in rows:
        line = " ".join(
            f"{row['modes'][label]['mode']:5d}" for label in labels if label in row["modes"]
        )
        print(
            f"{row['entry_id']:32s} {str(row['observed_mode']):>4s} {line}  "
            f"{row['mode_spread']:6d}"
        )
    print()
    for index, label in enumerate(labels):
        print(f"  {index}  {label}")

    spreads = [row["mode_spread"] for row in rows]
    print()
    print(f"mode spread across the five parameter sets: "
          f"median {int(np.median(spreads))}, max {max(spreads)}")
    print(f"entries whose mode never moves: {sum(1 for v in spreads if v == 0)} of {len(rows)}")
    print(f"entries whose mode moves by more than one: {sum(1 for v in spreads if v > 1)}")

    for family in ("Cas12a",):
        subset_rows = [row for row in rows if row["ortholog"] == family]
        if subset_rows:
            print()
            for row in subset_rows:
                print(
                    f"  {row['entry_id']:32s} observed {row['observed_mode']}, "
                    f"predicted {row['mode_range'][0]} to {row['mode_range'][1]}"
                )

    report = {
        "parameter_sets": subsets,
        "entries": rows,
        "median_mode_spread": int(np.median(spreads)),
        "max_mode_spread": int(max(spreads)),
        "unchanged": int(sum(1 for v in spreads if v == 0)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
