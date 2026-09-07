"""Classify displaced-strand nucleotides as paired or unpaired.

Uses distance from each displaced nucleotide to the nearest target-strand sugar.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from span_editor.io import load_assigned_inputs
from span_editor.structures import StructureGeometry, load_structure

PAIRED_BELOW = 12.0


def pairing(structure, geometry: StructureGeometry, target_chain: str) -> list[dict]:
    displaced = structure[
        (structure.chain_id == geometry.strand_chain) & (structure.atom_name == "C1'")
    ]
    target = structure[(structure.chain_id == target_chain) & (structure.atom_name == "C1'")]
    if displaced.array_length() == 0 or target.array_length() == 0:
        raise ValueError("one of the strands carries no sugars")

    rows = []
    for index in range(displaced.array_length()):
        position = displaced.coord[index]
        separations = np.linalg.norm(target.coord - position, axis=1)
        nearest = int(np.argmin(separations))
        rows.append(
            {
                "residue": int(displaced.res_id[index]),
                "base": str(displaced.res_name[index]),
                "nearest_target_residue": int(target.res_id[nearest]),
                "distance": round(float(separations[nearest]), 2),
                "paired": bool(separations[nearest] < PAIRED_BELOW),
            }
        )
    return sorted(rows, key=lambda row: row["residue"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--accessions", default="5F9R,6I1K")
    parser.add_argument("--out", type=Path, default=Path("results/strand_pairing.json"))
    args = parser.parse_args(argv)

    assigned = load_assigned_inputs(args.assigned)
    report = {"threshold": PAIRED_BELOW, "structures": {}}

    for accession in args.accessions.split(","):
        settings = assigned.structure_geometry.get(accession)
        if settings is None:
            print(f"{accession}: no entry under structure_geometry")
            continue
        geometry = StructureGeometry.from_config(accession, settings)
        target_chain = settings.get("target_chain") or settings.get("distal_closure", {}).get(
            "chain"
        )
        if not target_chain:
            print(f"{accession}: no target strand recorded, so pairing cannot be measured")
            continue

        structure = load_structure(args.structures / f"{accession}.cif.gz")
        rows = pairing(structure, geometry, target_chain)
        to_index = geometry.strand_residue_to_index

        print(f"{accession}, displaced chain {geometry.strand_chain} against {target_chain}")
        print(f"{'residue':>8s} {'base':>5s} {'index':>6s} {'nearest':>8s} {'distance':>9s} {'held':>6s}")
        for row in rows:
            index = to_index.get(row["residue"])
            print(
                f"{row['residue']:8d} {row['base']:>5s} "
                f"{'-' if index is None else index:>6} "
                f"{row['nearest_target_residue']:8d} {row['distance']:9.2f} "
                f"{'yes' if row['paired'] else 'no':>6s}"
            )

        paired = [row["residue"] for row in rows if row["paired"]]
        free = [row["residue"] for row in rows if not row["paired"]]

        # An R-loop has one bubble, so the unpaired nucleotides are contiguous
        # and the two borderline residues that dip under the threshold inside
        # the run are read as stacking against the duplex rather than as a pair
        # of their own. The bubble is the run from the first unpaired residue to
        # the last, and anything outside it may be pinned.
        protospacer = set(range(geometry.protospacer_residues[0], geometry.protospacer_residues[1] + 1)) if getattr(geometry, "protospacer_residues", None) else set()
        inside = [r for r in free if not protospacer or r in protospacer]
        bubble = [min(inside), max(inside)] if inside else None
        pinnable = [
            row["residue"]
            for row in rows
            if row["paired"] and (bubble is None or not bubble[0] <= row["residue"] <= bubble[1])
        ]
        print()
        print(f"  held by pairing   {paired}")
        print(f"  bubble, resolved  {free}")
        print(
            f"  currently pinned  {sorted(to_index)}, of which "
            f"{sorted(set(to_index) & set(free))} are bubble"
        )
        print(f"  bubble spans      {bubble}")
        print(f"  may be pinned     {pinnable}")
        print(
            f"  of those, inside the protospacer "
            f"{sorted(r for r in pinnable if r in to_index)}"
        )
        print()
        report["structures"][accession] = {
            "displaced_chain": geometry.strand_chain,
            "target_chain": target_chain,
            "rows": rows,
            "paired_residues": paired,
            "resolved_bubble_residues": free,
            "currently_pinned": sorted(to_index),
            "pinned_but_in_the_bubble": sorted(set(to_index) & set(free)),
            "bubble": bubble,
            "may_be_pinned": pinnable,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
