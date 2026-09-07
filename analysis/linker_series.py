"""Predicted profile across a series of contour lengths at one attachment site.

Reports capture probability at the reference radius and its centre of mass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from span_editor.geometry import NUMBERING_FROM_PAM_DISTAL
from span_editor.io import load_assigned_inputs
from span_editor.model import build_context
from span_editor.occupancy import capture_probability_curve
from span_editor.tether import TetherSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from substrate_check import reference_entry  # noqa: E402


def occupancy_profile(context, sampler, persistence: float, radius: float) -> dict:
    field = sampler.sample(
        linker=context.entry.linker,
        effector=context.effector,
        anchor=context.anchor,
        occupancy=context.occupancy,
        structure_id=context.entry.structure_id,
    )
    substrate = context.substrate(persistence)
    indices, curve = capture_probability_curve(
        field, substrate, np.array([radius]), indices=context.indices
    )
    values = curve[:, 0]
    total = float(values.sum())
    centre = float((indices * values).sum() / total) if total > 0 else float("nan")
    return {
        "profile": {int(i): float(v) for i, v in zip(indices, values)},
        "total": total,
        "centre_of_mass": centre,
        "peak_index": int(indices[int(np.argmax(values))]) if total > 0 else None,
        "surviving": field.surviving,
        "sampled": field.sampled,
        "survival": field.survival,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ortholog", default="spcas9")
    parser.add_argument("--effector", default="apobec1")
    parser.add_argument("--lengths", type=int, nargs="+", default=[3, 8, 16, 24, 32, 48])
    parser.add_argument("--composition", default="xten")
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--chains", type=int, default=20000)
    parser.add_argument("--out", type=Path, default=Path("results/linker_series.json"))
    args = parser.parse_args(argv)

    assigned = load_assigned_inputs(args.assigned)
    document = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    persistence = float(document["fitted"]["ssdna_persistence"])
    radius = float(document["link"]["reference_radius"])
    accession = assigned.structures.get(args.ortholog)
    if not accession:
        print(f"no structure assigned for {args.ortholog}")
        return 1

    geometry = assigned.structure_geometry.get(accession, {})
    numbering = geometry.get("numbering") or NUMBERING_FROM_PAM_DISTAL
    span = geometry.get("protospacer_residues") or [1, 20]
    spacer_length = int(span[1]) - int(span[0]) + 1

    assigned.sampling["substrate_samples"] = args.samples
    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=args.chains,
        seed=assigned.sampling.get("seed", 0),
    )

    rows = []
    for length in args.lengths:
        entry = reference_entry(
            args.ortholog,
            accession,
            spacer_length,
            args.effector,
            numbering=numbering,
            linker_residues=length,
        )
        entry.linker_composition = args.composition
        context = build_context(entry, assigned, args.structures)
        measured = occupancy_profile(context, sampler, persistence, radius)
        rows.append({"linker_residues": length, "flexible_tail": context.effector.flexible_tail, **measured})

    report = {
        "ortholog": args.ortholog,
        "structure": accession,
        "effector": args.effector,
        "composition": args.composition,
        "numbering": numbering,
        "persistence": persistence,
        "reference_radius": radius,
        "rows": rows,
    }

    print(f"{args.ortholog} against {accession}, effector {args.effector}, {args.composition}")
    print(
        f"tether length is the linker plus a flexible tail of "
        f"{rows[0]['flexible_tail']} residues; capture radius {radius:.1f} A"
    )
    print()
    shown = sorted(rows[0]["profile"])[:14]
    header = "  ".join(f"{index:5d}" for index in shown)
    print(f" linker  centre  peak   {header}")
    for row in rows:
        line = "  ".join(f"{1000 * row['profile'][index]:5.1f}" for index in shown)
        peak = row["peak_index"] if row["peak_index"] is not None else 0
        print(f" {row['linker_residues']:6d}  {row['centre_of_mass']:6.2f}  {peak:4d}   {line}")
    print()
    print("capture probability in parts per thousand; centre of mass over all positions")
    print(f"centre of mass by linker length: "
          f"{[round(row['centre_of_mass'], 2) for row in rows]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
