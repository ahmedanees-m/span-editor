"""Entries where the link reaches its numerical ceiling.

Reports the modes read from capture probability alongside those read from the
linked profile.
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

# 1 - exp(-x) reaches exactly 1.0 in double precision a little above this.
SATURATION_EXPONENT = 37.0


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
    parser.add_argument("--out", type=Path, default=Path("results/link_saturation.json"))
    args = parser.parse_args(argv)

    assigned = load_assigned_inputs(args.assigned)
    fitted = load_parameters(args.params)
    corpus = [
        e for e in read_corpus(args.corpus)
        if e.observed is not None or e.observed_window
    ]

    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=assigned.sampling.get("tether_chains", 20000),
        seed=assigned.sampling.get("seed", 0),
        target_surviving=assigned.sampling.get("target_surviving", 0),
    )
    model = SpanModel(sampler)

    # A near-zero alpha makes the link linear, so the profile it returns is the
    # capture probability up to a constant and its argmax is the capture argmax.
    linear = ModelParameters(
        capture_radius=fitted.capture_radius,
        ssdna_persistence=fitted.ssdna_persistence,
        link_alpha=1e-9,
    )
    print(f"fitted alpha {fitted.link_alpha:.2f}; the link reaches exactly 1.0 once "
          f"alpha times capture exceeds about {SATURATION_EXPONENT:.0f}")
    print(f"so any entry whose peak capture exceeds "
          f"{SATURATION_EXPONENT / fitted.link_alpha:.4f} saturates")
    print()

    rows, saturated = [], []
    for entry in corpus:
        try:
            context = build_context(entry, assigned, args.structures)
        except (ValueError, FileNotFoundError) as error:
            print(f"skipped {entry.entry_id}: {error}")
            continue
        linked = model.predict(context, fitted)
        capture = model.predict(context, linear)
        values = np.asarray(linked.values, dtype=float)
        raw = np.asarray(capture.values, dtype=float)
        peak_capture = float(np.nanmax(raw)) / 1e-9 if raw.size else float("nan")
        exponent = fitted.link_alpha * peak_capture
        at_one = int(np.sum(values == 1.0))
        linked_n = linked.normalised()
        capture_n = capture.normalised()
        row = {
            "entry_id": entry.entry_id,
            "source_key": entry.source_key,
            "ortholog": entry.ortholog,
            "observed_mode": observed_mode(entry),
            "peak_capture": round(peak_capture, 8),
            "exponent_at_the_peak": round(float(exponent), 2),
            "positions_at_exactly_one": at_one,
            "linked_mode": None if linked_n.degenerate else int(linked_n.mode),
            "linked_degenerate": bool(linked_n.degenerate),
            "capture_mode": None if capture_n.degenerate else int(capture_n.mode),
            "capture_degenerate": bool(capture_n.degenerate),
            "saturates": bool(at_one > 1),
        }
        rows.append(row)
        if row["saturates"]:
            saturated.append(row)

    print(f"{'entry':38s} {'peak capture':>13s} {'exponent':>9s} {'at 1.0':>7s} "
          f"{'linked':>7s} {'capture':>8s} {'obs':>4s}")
    for row in sorted(rows, key=lambda r: -r["peak_capture"])[:8]:
        print(f"{row['entry_id']:38s} {row['peak_capture']:13.6f} "
              f"{row['exponent_at_the_peak']:9.1f} {row['positions_at_exactly_one']:7d} "
              f"{str(row['linked_mode']):>7s} {str(row['capture_mode']):>8s} "
              f"{str(row['observed_mode']):>4s}")

    print()
    print(f"{len(saturated)} of {len(rows)} entries saturate the link")
    for row in saturated:
        print(f"  {row['entry_id']:38s} linked says "
              f"{'no window' if row['linked_degenerate'] else 'mode ' + str(row['linked_mode'])}, "
              f"capture says mode {row['capture_mode']}, observed {row['observed_mode']}")

    disagree = [r for r in rows if r["linked_degenerate"] != r["capture_degenerate"]
                or (not r["linked_degenerate"] and not r["capture_degenerate"]
                    and r["linked_mode"] != r["capture_mode"])]
    print()
    print(f"entries where the linked and capture profiles disagree: {len(disagree)}")
    for row in disagree:
        print(f"  {row['entry_id']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "fitted_alpha": fitted.link_alpha,
                "saturation_exponent": SATURATION_EXPONENT,
                "peak_capture_that_saturates": SATURATION_EXPONENT / fitted.link_alpha,
                "entries": rows,
                "saturated": [r["entry_id"] for r in saturated],
                "disagreeing": [r["entry_id"] for r in disagree],
            },
            handle, indent=2,
        )
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
