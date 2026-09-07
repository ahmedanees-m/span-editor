"""Span, direction and rigidity a connector needs to reach a given position.

Computed for the protospacer positions no candidate tether reaches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from span_editor.io import load_assigned_inputs, load_parameters, read_corpus
from span_editor.model import build_context
from span_editor.polymer import mean_square_end_to_end

HELIX_RISE = 1.5      # angstrom per residue along the axis of an alpha helix
SAMPLES = 20000


def solid_angle_fraction(half_angle_deg: float) -> float:
    """Fraction of the sphere a cone of the given half angle covers."""
    return float(0.5 * (1.0 - np.cos(np.deg2rad(half_angle_deg))))


def residues_for_span(span: float, composition, tolerance: float = 0.5) -> int | None:
    """Contour length whose root mean square end-to-end distance reaches a span."""
    for n_links in range(1, 400):
        reach = np.sqrt(mean_square_end_to_end(n_links, composition.rise, composition.persistence))
        if reach >= span:
            return n_links
    return None


def specify(context, parameters, index: int, rng, samples: int) -> dict | None:
    substrate = context.substrate(parameters.ssdna_persistence)
    cloud, weights = substrate.cloud(index)
    draw = rng.choice(cloud.shape[0], size=samples, p=weights / weights.sum())
    target = cloud[draw]

    # Points within the capture radius of the nucleotide, drawn uniformly in the
    # ball rather than on its surface.
    direction = rng.normal(size=(samples, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    radius = parameters.capture_radius * rng.random(samples) ** (1.0 / 3.0)
    catalytic = target + radius[:, None] * direction

    # The junction sits one active-site offset back from the catalytic centre,
    # in a direction the model leaves free.
    offset = float(np.linalg.norm(context.effector.active_site_offset))
    step = rng.normal(size=(samples, 3))
    step /= np.linalg.norm(step, axis=1, keepdims=True)
    junction = catalytic - offset * step

    allowed = np.ones(samples, dtype=bool)
    if context.occupancy is not None:
        allowed &= ~context.occupancy.occluded(junction)
        allowed &= ~context.occupancy.occluded(catalytic)
    if not np.any(allowed):
        return None

    kept = junction[allowed]
    span = np.linalg.norm(kept, axis=1)
    unit = kept / span[:, None]
    mean_direction = unit.mean(axis=0)
    resultant = float(np.linalg.norm(mean_direction))
    if resultant <= 1e-9:
        return None
    mean_direction /= resultant
    angles = np.rad2deg(np.arccos(np.clip(unit @ mean_direction, -1.0, 1.0)))

    return {
        "index": int(index),
        "sterically_allowed": round(float(np.mean(allowed)), 4),
        "span_mean": round(float(span.mean()), 2),
        "span_range": [round(float(np.percentile(span, 10)), 2), round(float(np.percentile(span, 90)), 2)],
        "direction": [round(float(value), 4) for value in mean_direction],
        "mean_resultant_length": round(resultant, 4),
        "tolerance_68": round(float(np.percentile(angles, 68)), 1),
        "tolerance_90": round(float(np.percentile(angles, 90)), 1),
        "solid_angle_fraction_68": round(solid_angle_fraction(float(np.percentile(angles, 68))), 4),
        "active_site_offset": round(float(np.linalg.norm(context.effector.active_site_offset)), 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--inversion", type=Path, default=Path("results/inversion.json"))
    parser.add_argument("--reference", default="hnhx-ABEmax7.10")
    parser.add_argument("--site", default="n_terminal")
    parser.add_argument("--indices", default="", help="comma separated; defaults to the unreachable set")
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--out", type=Path, default=Path("results/connector_spec.json"))
    args = parser.parse_args(argv)

    fitted = args.params.with_name("fitted.yaml")
    document = yaml.safe_load((fitted if fitted.exists() else args.params).read_text("utf-8"))
    if document.get("status") != "fitted":
        print("no fitted parameters found; run the fit first")
        return 1

    assigned = load_assigned_inputs(args.assigned)
    parameters = load_parameters(args.params)
    entry = next(e for e in read_corpus(args.corpus) if e.entry_id == args.reference)
    from dataclasses import replace

    entry = replace(entry, anchor_site=args.site, return_site="", return_residues=0)
    context = build_context(entry, assigned, args.structures)

    if args.indices:
        targets = [int(value) for value in args.indices.split(",")]
        source = "given on the command line"
    elif args.inversion.exists():
        targets = json.loads(args.inversion.read_text("utf-8"))["unreachable_by_flexible"]
        source = f"the unreachable set in {args.inversion}"
    else:
        print(f"{args.inversion} not found and no indices given")
        return 1
    if not targets:
        print("every position is reachable by a flexible linker; there is nothing to specify")
        report = {"site": args.site, "targets": [], "note": "no unreachable positions"}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 0

    rng = np.random.default_rng(assigned.sampling.get("seed", 0))
    rows = []
    for index in targets:
        record = specify(context, parameters, index, rng, args.samples)
        if record is None:
            rows.append({"index": int(index), "sterically_allowed": 0.0})
            continue
        for name, composition in assigned.compositions.items():
            record[f"residues_{name}"] = residues_for_span(record["span_mean"], composition)
        record["residues_alpha_helix"] = int(np.ceil(record["span_mean"] / HELIX_RISE))
        rows.append(record)

    print(f"attachment site {args.site} on {entry.structure_id}, targets from {source}")
    print(f"effector {entry.effector}, active site {rows[0].get('active_site_offset')} A from the junction")
    print()
    print(
        f"{'index':>5s} {'allowed':>8s} {'span':>7s} {'10 to 90':>14s} "
        f"{'tol 68':>7s} {'tol 90':>7s} {'cone':>7s} {'helix':>6s}"
    )
    for row in rows:
        if not row.get("span_mean"):
            print(f"{row['index']:5d} {row['sterically_allowed']:8.4f}   nothing sterically allowed")
            continue
        print(
            f"{row['index']:5d} {row['sterically_allowed']:8.4f} {row['span_mean']:7.1f} "
            f"{str(row['span_range']):>14s} {row['tolerance_68']:7.1f} {row['tolerance_90']:7.1f} "
            f"{row['solid_angle_fraction_68']:7.4f} {row['residues_alpha_helix']:6d}"
        )
    print()
    print("direction in the anchor frame, and contour length by composition class")
    for row in rows:
        if not row.get("span_mean"):
            continue
        lengths = ", ".join(
            f"{name} {row.get('residues_' + name)}" for name in sorted(assigned.compositions)
        )
        print(f"{row['index']:5d} {row['direction']}  {lengths}")

    report = {
        "site": args.site,
        "reference_entry": args.reference,
        "structure": entry.structure_id,
        "effector": entry.effector,
        "capture_radius": parameters.capture_radius,
        "target_source": source,
        "helix_rise": HELIX_RISE,
        "specifications": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
