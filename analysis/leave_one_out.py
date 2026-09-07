"""Score each position against an ensemble built without its own coordinate.

Restricted to interior anchors, since dropping the outermost anchor on either
side leaves the chain unbounded rather than interpolated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from span_editor.io import load_assigned_inputs, load_parameters, read_corpus
from span_editor.model import SpanModel, build_context
from span_editor.tether import TetherSampler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path("results/leave_one_out.json"))
    args = parser.parse_args(argv)

    assigned = load_assigned_inputs(args.assigned)
    parameters = load_parameters(args.params)
    corpus = read_corpus(args.corpus)

    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=assigned.sampling.get("tether_chains", 20000),
        seed=assigned.sampling.get("seed", 0),
        target_surviving=assigned.sampling.get("target_surviving", 0),
    )
    model = SpanModel(sampler)

    rows = []
    for entry in corpus:
        if entry.observed is None and not entry.observed_window:
            continue
        try:
            context = build_context(entry, assigned, args.structures)
        except (ValueError, FileNotFoundError) as error:
            print(f"skipped {entry.entry_id}: {error}")
            continue

        anchored = sorted(a.index for a in context.strand_anchors)
        held = model.predict(context, parameters).normalised()
        loose = model.predict_leave_one_out(context, parameters).normalised()

        observed_mode = (
            entry.observed.mode
            if entry.observed is not None
            else int(round(0.5 * sum(entry.observed_window)))
        )
        scored_region = (
            list(range(entry.observed_window[0], entry.observed_window[1] + 1))
            if entry.observed_window
            else [observed_mode]
        )
        read_off = [i for i in scored_region if i in anchored]

        rows.append(
            {
                "entry_id": entry.entry_id,
                "ortholog": entry.ortholog,
                "structure": entry.structure_id,
                "anchors": anchored,
                "window_positions_that_were_anchors": read_off,
                "fraction_of_window_read": round(len(read_off) / max(len(scored_region), 1), 3),
                "observed_mode": observed_mode,
                "held_mode": int(held.mode),
                "held_window": list(held.window(args.threshold)),
                "loose_mode": int(loose.mode),
                "loose_window": list(loose.window(args.threshold)),
                "mode_moved": int(loose.mode) - int(held.mode),
                "held_degenerate": bool(held.degenerate),
                "loose_degenerate": bool(loose.degenerate),
                "positions_scored": int(loose.indices.shape[0]),
            }
        )
        flag = ""
        if held.degenerate:
            flag += "  held profile has no window"
        if loose.degenerate:
            flag += "  leave-one-out profile has no window"
        print(
            f"{entry.entry_id:40s} {entry.ortholog:8s} "
            f"window read {rows[-1]['fraction_of_window_read']:4.0%}  "
            f"held mode {held.mode:2d} -> leave one out {loose.mode:2d}  "
            f"observed {observed_mode:2d}{flag}"
        )

    if rows:
        # A mode that moved because the held profile had no window in the first
        # place is not evidence that the held profile was reading the answer.
        moved = [
            row
            for row in rows
            if row["mode_moved"] != 0 and not row["held_degenerate"]
        ]
        artefact = [
            row for row in rows if row["mode_moved"] != 0 and row["held_degenerate"]
        ]
        print()
        print(f"{len(rows)} entries scored, mode moved on {len(moved)}")
        for row in moved:
            print(
                f"  {row['entry_id']:40s} {row['held_mode']} -> {row['loose_mode']}, "
                f"observed {row['observed_mode']}"
            )
        if artefact:
            print()
            print(
                f"{len(artefact)} more moved, but from a held profile that had no window "
                "at all, so the move is the scoring rule rather than the deposition:"
            )
            for row in artefact:
                print(f"  {row['entry_id']:40s} {row['held_mode']} -> {row['loose_mode']}")

        exposed = [row for row in rows if row["fraction_of_window_read"] > 0]
        print()
        print(
            f"entries whose reported window overlapped the anchors at all: {len(exposed)} "
            f"of {len(rows)}"
        )
        for row in exposed:
            print(
                f"  {row['entry_id']:40s} {row['fraction_of_window_read']:.0%} of the window, "
                f"mode moved {row['mode_moved']:+d}"
            )

    report = {"threshold": args.threshold, "parameters": parameters.as_dict(), "entries": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
