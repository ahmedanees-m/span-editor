"""Superpose two Cas12a orthologs on the residues they share.

Reports the separation of the amino termini and their distances to the displaced
strand over the reported window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from biotite.sequence import ProteinSequence
from biotite.sequence.align import SubstitutionMatrix, align_optimal, get_codes
from biotite.structure import filter_amino_acids, superimpose

from span_editor.io import load_assigned_inputs
from span_editor.structures import StructureGeometry, load_structure


def alpha_carbons(structure, chain: str):
    protein = structure[(structure.chain_id == chain) & filter_amino_acids(structure)]
    return protein[protein.atom_name == "CA"]


def matched_pairs(first, second):
    """Alpha carbons paired by an optimal alignment of the two sequences."""
    codes = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
        "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
        "TYR": "Y", "VAL": "V",
    }
    left = ProteinSequence("".join(codes.get(str(n).upper(), "X") for n in first.res_name))
    right = ProteinSequence("".join(codes.get(str(n).upper(), "X") for n in second.res_name))
    matrix = SubstitutionMatrix.std_protein_matrix()
    alignment = align_optimal(left, right, matrix, gap_penalty=(-10, -1))[0]
    trace = alignment.trace
    keep = (trace[:, 0] >= 0) & (trace[:, 1] >= 0)
    return trace[keep, 0], trace[keep, 1], alignment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--reference", default="6I1K")
    parser.add_argument("--other", default="5XUS")
    parser.add_argument("--other-chain", default="A")
    parser.add_argument("--out", type=Path, default=Path("results/ortholog_substitution.json"))
    args = parser.parse_args(argv)

    other_path = args.structures / f"{args.other}.cif.gz"
    if not other_path.exists():
        print(f"{other_path} not found; add {args.other} to data/structures.txt and fetch it")
        return 1

    assigned = load_assigned_inputs(args.assigned)
    settings = assigned.structure_geometry[args.reference]
    geometry = StructureGeometry.from_config(args.reference, settings)

    reference = load_structure(args.structures / f"{args.reference}.cif.gz")
    other = load_structure(other_path)

    first = alpha_carbons(reference, geometry.anchor_chain)
    second = alpha_carbons(other, args.other_chain)
    left, right, alignment = matched_pairs(first, second)
    print(f"{args.reference} chain {geometry.anchor_chain}: {first.array_length()} alpha carbons")
    print(f"{args.other} chain {args.other_chain}: {second.array_length()} alpha carbons")
    print(f"aligned pairs: {len(left)}")

    fitted, transform = superimpose(first[left], second[right])
    deviation = np.linalg.norm(first[left].coord - fitted.coord, axis=1)
    print(f"superposition over the aligned core: {np.sqrt(np.mean(deviation**2)):.2f} angstrom")

    moved = transform.apply(second)

    reference_start = first.coord[0]
    other_start = moved[
        (moved.chain_id == args.other_chain) & (moved.atom_name == "CA")
    ].coord[0]
    separation = float(np.linalg.norm(reference_start - other_start))
    print()
    print(f"amino terminus of {args.reference}: residue {int(first.res_id[0])}")
    print(f"amino terminus of {args.other} after superposition: "
          f"{separation:.1f} angstrom away")

    # Where each terminus sits relative to the substrate the deaminase must reach.
    strand = reference[
        (reference.chain_id == geometry.strand_chain) & (reference.atom_name == "C1'")
    ]
    to_index = geometry.strand_residue_to_index
    rows = []
    print()
    print(f"{'index':>6s} {args.reference + ' N to it':>18s} {args.other + ' N to it':>18s} {'change':>8s}")
    for slot in range(strand.array_length()):
        residue = int(strand.res_id[slot])
        index = to_index.get(residue)
        if index is None or not 1 <= index <= 21:
            continue
        position = strand.coord[slot]
        a = float(np.linalg.norm(position - reference_start))
        b = float(np.linalg.norm(position - other_start))
        rows.append({"index": index, "reference": round(a, 1), "other": round(b, 1),
                     "change": round(b - a, 1)})
        print(f"{index:6d} {a:18.1f} {b:18.1f} {b - a:8.1f}")

    window = [row for row in rows if 8 <= row["index"] <= 12]
    if window:
        print()
        print(
            f"over the reported window, positions 8 to 12: "
            f"{args.reference} {np.mean([r['reference'] for r in window]):.1f} angstrom, "
            f"{args.other} {np.mean([r['other'] for r in window]):.1f}"
        )

    report = {
        "reference": args.reference,
        "other": args.other,
        "aligned_pairs": int(len(left)),
        "core_rmsd": round(float(np.sqrt(np.mean(deviation**2))), 2),
        "amino_terminus_separation": round(separation, 2),
        "distances_to_the_displaced_strand": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
