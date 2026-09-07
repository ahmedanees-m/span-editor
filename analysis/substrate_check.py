"""Check the displaced-strand ensemble against the structure it is built from.

Verifies that pinned nucleotides do not move and that no ensemble weight falls
inside the excluded volume. Runs at unfitted defaults.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from span_editor.geometry import NUMBERING_FROM_PAM_DISTAL
from span_editor.io import CorpusEntry, load_assigned_inputs
from span_editor.model import SpanModel, build_context
from span_editor.occupancy import ModelParameters
from span_editor.tether import TetherSampler


def reference_entry(
    ortholog: str,
    accession: str,
    spacer_length: int,
    effector: str,
    numbering: str = NUMBERING_FROM_PAM_DISTAL,
    linker_residues: int = 32,
):
    return CorpusEntry(
        entry_id=f"{ortholog}-reference",
        source_key="reference",
        editor=f"{ortholog} reference geometry",
        ortholog=ortholog,
        anchor_site="n_terminal",
        linker_residues=linker_residues,
        linker_composition="xten",
        effector=effector,
        label="none",
        readout_class="amplicon_sequencing",
        tier="A",
        spacer_length=spacer_length,
        numbering=numbering,
        structure_id=accession,
    )


def summarise(context, persistence: float, radii: dict) -> dict:
    substrate = context.substrate(persistence)
    pinned = {anchor.index for anchor in context.strand_anchors}
    rows = []
    for index in context.indices:
        cloud, weights = substrate.cloud(index)
        centre = np.average(cloud, axis=0, weights=weights)
        spread = float(
            np.sqrt(np.average(np.sum((cloud - centre) ** 2, axis=1), weights=weights))
        )
        occluded = (
            float(np.sum(weights[context.occupancy.occluded(cloud)]))
            if context.occupancy is not None
            else 0.0
        )
        rows.append(
            {
                "index": int(index),
                "pinned": index in pinned,
                "spread": spread,
                "distance_from_anchor": float(np.linalg.norm(centre)),
                "weight_inside_the_protein": occluded,
                "effective_samples": float(1.0 / np.sum(weights**2)),
            }
        )
    return {"rows": rows, "pinned_indices": sorted(pinned)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ortholog", default="spcas9")
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--spacer-length", type=int, default=None)
    parser.add_argument("--linker-residues", type=int, default=32)
    parser.add_argument("--effector", default="tada8e")
    parser.add_argument("--samples", type=int, default=4000)
    parser.add_argument("--out", type=Path, default=Path("results/substrate_check.json"))
    args = parser.parse_args(argv)

    assigned = load_assigned_inputs(args.assigned)
    fitted = yaml.safe_load(args.params.read_text(encoding="utf-8"))["fitted"]
    persistence = float(fitted["ssdna_persistence"])

    accession = assigned.structures.get(args.ortholog)
    if not accession:
        print(f"no structure assigned for {args.ortholog}")
        return 1

    geometry = assigned.structure_geometry.get(accession, {})
    numbering = geometry.get("numbering") or NUMBERING_FROM_PAM_DISTAL
    span = geometry.get("protospacer_residues") or [1, 20]
    spacer_length = args.spacer_length or (int(span[1]) - int(span[0]) + 1)

    entry = reference_entry(
        args.ortholog,
        accession,
        spacer_length,
        args.effector,
        numbering=numbering,
        linker_residues=args.linker_residues,
    )
    assigned.sampling["substrate_samples"] = args.samples
    context = build_context(entry, assigned, args.structures)

    summary = summarise(context, persistence, assigned.coarse_radii)
    pinned = summary["pinned_indices"]
    free = [row for row in summary["rows"] if not row["pinned"]]
    held = [row for row in summary["rows"] if row["pinned"]]

    report = {
        "ortholog": args.ortholog,
        "structure": accession,
        "numbering": numbering,
        "spacer_length": spacer_length,
        "linker_residues": args.linker_residues,
        "effector": args.effector,
        "persistence": persistence,
        "samples": args.samples,
        "n_occluding_spheres": len(context.occupancy) if context.occupancy else 0,
        "pinned_indices": pinned,
        "max_spread_pinned": max((row["spread"] for row in held), default=0.0),
        "max_weight_inside_protein": max(
            (row["weight_inside_the_protein"] for row in summary["rows"]), default=0.0
        ),
        "min_effective_samples": min(row["effective_samples"] for row in summary["rows"]),
        "rows": summary["rows"],
    }

    print(f"{args.ortholog} against {accession}, {spacer_length} nt protospacer, {numbering}")
    print(f"  occluding spheres      {report['n_occluding_spheres']}")
    print(f"  pinned indices         {pinned}")
    print(f"  largest spread pinned  {report['max_spread_pinned']:.3f} A")
    print(f"  weight inside protein  {report['max_weight_inside_protein']:.4f}")
    print(f"  smallest effective n   {report['min_effective_samples']:.0f} of {args.samples}")
    print()
    print(" index  pinned  spread A  from anchor A")
    for row in summary["rows"]:
        mark = "yes" if row["pinned"] else "no "
        print(
            f" {row['index']:5d}  {mark}     {row['spread']:7.2f}   {row['distance_from_anchor']:9.2f}"
        )

    if free:
        widest = max(free, key=lambda row: row["spread"])
        print()
        print(
            f"widest point of the bubble: {widest['spread']:.2f} A at index "
            f"{widest['index']}, between pins at {pinned[0]} and {pinned[-1]}"
        )

    # Reachability at the starting parameters, for comparison with the
    # straight-line distance from the anchor reported alongside.
    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=assigned.sampling.get("tether_chains", 20000),
        seed=assigned.sampling.get("seed", 0),
    )
    parameters = ModelParameters.from_dict(fitted)
    field = sampler.sample(
        linker=context.entry.linker,
        effector=context.effector,
        anchor=context.anchor,
        occupancy=context.occupancy,
        structure_id=context.entry.structure_id,
    )
    live = field.positions[field.weights > 0]
    substrate = context.substrate(persistence)

    # Closest approach rather than captured probability, since the capture
    # radius is not yet set at this stage.
    approach = {}
    for index in context.indices:
        cloud, weights = substrate.cloud(index)
        kept = cloud[weights > 0][:600]
        if live.size and kept.size:
            separations = np.linalg.norm(live[:, None, :] - kept[None, :, :], axis=2)
            approach[int(index)] = float(separations.min())
    reachable = min(approach, key=approach.get) if approach else None
    nearest = min(summary["rows"], key=lambda row: row["distance_from_anchor"])["index"]

    report["reach"] = {
        "parameters": parameters.as_dict(),
        "surviving_tether_conformations": field.surviving,
        "sampled_tether_conformations": field.sampled,
        "tether_survival": field.survival,
        "closest_approach": approach,
        "most_reachable_index": reachable,
        "nearest_index_by_straight_line": nearest,
    }

    print()
    print(
        f"tether conformations surviving exclusion "
        f"{report['reach']['surviving_tether_conformations']} of "
        f"{report['reach']['sampled_tether_conformations']} "
        f"({100 * report['reach']['tether_survival']:.1f} per cent)"
    )
    print(f"nearest index by straight-line distance  {nearest}")
    print(f"most reachable index                     {reachable}")
    print(" index  closest approach of the active site, angstrom")
    if approach:
        floor = min(approach.values())
        for index, value in approach.items():
            bar = "#" * max(0, int(round(30 * (1 - (value - floor) / 40))))
            print(f" {index:5d}  {value:7.1f}  {bar}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
