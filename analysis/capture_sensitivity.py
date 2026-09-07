"""Predicted modes across a range of capture radii, other parameters held.

Cas12a entries are swept with SpCas9 controls alongside.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from span_editor.io import load_assigned_inputs, load_parameters, read_corpus
from span_editor.model import SpanModel, build_context
from span_editor.occupancy import ModelParameters
from span_editor.tether import TetherSampler

CAS12A = "Cas12a"


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
    parser.add_argument(
        "--radii", type=float, nargs="+",
        default=[8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 25.0],
    )
    parser.add_argument(
        "--controls", type=str, nargs="+",
        default=["hnhx-ABEmax7.10", "huangcp-ABEmax", "komor-linker-16"],
        help="SpCas9 entries swept alongside as controls",
    )
    parser.add_argument("--out", type=Path, default=Path("results/capture_sensitivity.json"))
    args = parser.parse_args(argv)

    assigned = load_assigned_inputs(args.assigned)
    fitted = load_parameters(args.params)
    corpus = {e.entry_id: e for e in read_corpus(args.corpus)}
    chosen = [
        e for e in corpus.values()
        if (e.observed is not None or e.observed_window)
        and (e.ortholog == CAS12A or e.entry_id in set(args.controls))
    ]
    print(f"fitted capture radius {fitted.capture_radius} angstrom, "
          f"sweeping {args.radii}")
    print(f"{len(chosen)} entries: {sum(1 for e in chosen if e.ortholog == CAS12A)} Cas12a, "
          f"{sum(1 for e in chosen if e.ortholog != CAS12A)} SpCas9 controls")
    print()

    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=assigned.sampling.get("tether_chains", 20000),
        seed=assigned.sampling.get("seed", 0),
        target_surviving=assigned.sampling.get("target_surviving", 0),
    )
    model = SpanModel(sampler)

    contexts = {}
    for entry in chosen:
        try:
            contexts[entry.entry_id] = build_context(entry, assigned, args.structures)
        except (ValueError, FileNotFoundError) as error:
            print(f"skipped {entry.entry_id}: {error}")

    rows = {e.entry_id: {} for e in chosen}
    for radius in args.radii:
        parameters = ModelParameters(
            capture_radius=radius,
            ssdna_persistence=fitted.ssdna_persistence,
            link_alpha=fitted.link_alpha,
        )
        for entry in chosen:
            context = contexts.get(entry.entry_id)
            if context is None:
                continue
            profile = model.predict(context, parameters).normalised()
            rows[entry.entry_id][radius] = {
                "mode": None if profile.degenerate else int(profile.mode),
                "window": list(profile.window(0.5)),
                "degenerate": bool(profile.degenerate),
            }
        print(f"  {radius:5.1f} angstrom done")

    print()
    header = " ".join(f"{r:5.0f}" for r in args.radii)
    print(f"{'entry':32s} {'obs':>5s} {'window':>9s}  {header}")
    report = []
    for entry in chosen:
        seen = rows[entry.entry_id]
        line = " ".join(
            f"{seen[r]['mode']:5d}" if seen.get(r) and seen[r]["mode"] is not None else "    ."
            for r in args.radii
        )
        window = entry.observed_window or []
        print(f"{entry.entry_id:32s} {str(observed_mode(entry)):>5s} "
              f"{('%d-%d' % tuple(window)) if window else '':>9s}  {line}")
        report.append(
            {
                "entry_id": entry.entry_id,
                "ortholog": entry.ortholog,
                "observed_mode": observed_mode(entry),
                "observed_window": list(window),
                "by_radius": {str(r): seen.get(r) for r in args.radii},
            }
        )

    # Radii at which a Cas12a mode falls inside its reported window.
    print()
    for row in report:
        if row["ortholog"] != CAS12A or not row["observed_window"]:
            continue
        low, high = row["observed_window"]
        inside = [
            r for r in args.radii
            if (v := row["by_radius"].get(str(r))) and v["mode"] is not None
            and low <= v["mode"] <= high
        ]
        print(f"  {row['entry_id']:32s} reported {low} to {high}: "
              f"{'inside at ' + ', '.join(f'{r:.0f}' for r in inside) if inside else 'never inside'}")

    print()
    for row in report:
        if row["ortholog"] == CAS12A:
            continue
        modes = [v["mode"] for v in row["by_radius"].values() if v and v["mode"] is not None]
        hits = [m for m in modes if abs(m - row["observed_mode"]) <= 1]
        print(f"  control {row['entry_id']:26s} observed {row['observed_mode']}, "
              f"within one nucleotide at {len(hits)} of {len(modes)} radii")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(
            {"fitted_capture_radius": fitted.capture_radius, "radii": args.radii,
             "held": {"ssdna_persistence": fitted.ssdna_persistence,
                      "link_alpha": fitted.link_alpha},
             "entries": report},
            handle, indent=2,
        )
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
