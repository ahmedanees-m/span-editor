"""Measure active-site offset and body radius for an effector from a structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from biotite.structure import filter_amino_acids

from span_editor.structures import load_structure

BACKBONE = "CA"


def catalytic_centre(structure, chain: str, residue: int) -> np.ndarray:
    selection = structure[(structure.chain_id == chain) & (structure.res_id == residue)]
    if selection.array_length() == 0:
        raise ValueError(f"residue {residue} is absent from chain {chain}")
    return selection.coord.mean(axis=0)


def measure(
    path: Path,
    chain: str,
    active_site_residue: int,
    terminus: str,
    first_residue: int | None = None,
    last_residue: int | None = None,
    tail_cutoff: float = 1.5,
) -> dict:
    structure = load_structure(path)
    domain = structure[(structure.chain_id == chain) & filter_amino_acids(structure)]
    if domain.array_length() == 0:
        raise ValueError(f"chain {chain} carries no amino acids")

    # A fusion construct may carry a crystallisation partner on the same chain,
    # so the domain of interest is bounded explicitly rather than taken as the
    # whole chain.
    if first_residue is not None:
        domain = domain[domain.res_id >= first_residue]
    if last_residue is not None:
        domain = domain[domain.res_id <= last_residue]
    if domain.array_length() == 0:
        raise ValueError(f"no residues left in chain {chain} after the range was applied")

    alpha = domain[domain.atom_name == BACKBONE]
    residues = np.sort(np.unique(alpha.res_id)).astype(int)

    # The folded body is separated from any terminal tail before anything is
    # measured. A residue lying further than tail_cutoff radii of gyration from
    # the centroid is treated as tail: it behaves as flexible chain, so it
    # belongs to the tether rather than to a rigid offset. The cutoff is applied
    # twice so that a long tail cannot inflate the radius that defines it.
    body = alpha
    for _ in range(2):
        centroid = body.coord.mean(axis=0)
        spread = np.linalg.norm(body.coord - centroid, axis=1)
        gyration = float(np.sqrt(np.mean(spread**2)))
        keep = np.linalg.norm(alpha.coord - centroid, axis=1) <= tail_cutoff * gyration
        if not np.any(keep):
            break
        body = alpha[keep]

    body_residues = np.sort(np.unique(body.res_id)).astype(int)
    if terminus == "c_terminal":
        junction_residue = int(body_residues.max())
        exit_residue = int(body_residues.min())
        tail = [int(r) for r in residues if r > junction_residue]
        exit_tail = [int(r) for r in residues if r < exit_residue]
    else:
        junction_residue = int(body_residues.min())
        exit_residue = int(body_residues.max())
        tail = [int(r) for r in residues if r < junction_residue]
        exit_tail = [int(r) for r in residues if r > exit_residue]

    backbone = alpha[alpha.res_id == junction_residue]
    if backbone.array_length() == 0:
        raise ValueError(f"residue {junction_residue} of chain {chain} has no {BACKBONE}")
    junction = backbone.coord[0]

    departure = alpha[alpha.res_id == exit_residue]
    if departure.array_length() == 0:
        raise ValueError(f"residue {exit_residue} of chain {chain} has no {BACKBONE}")
    exit_offset = departure.coord[0] - junction

    centre = catalytic_centre(structure, chain, active_site_residue)
    offset = centre - junction

    centroid = body.coord.mean(axis=0)
    spread = np.linalg.norm(body.coord - centroid, axis=1)
    gyration = float(np.sqrt(np.mean(spread**2)))

    return {
        "chain": chain,
        "terminus": terminus,
        "junction_residue": junction_residue,
        "active_site_residue": active_site_residue,
        "modelled_residues": [int(residues.min()), int(residues.max())],
        "body_residues": [int(body_residues.min()), int(body_residues.max())],
        "resolved_tail": len(tail),
        "tail_cutoff": tail_cutoff,
        "active_site_offset": [round(float(value), 2) for value in offset],
        "offset_magnitude": round(float(np.linalg.norm(offset)), 2),
        "exit_residue": exit_residue,
        "exit_offset": [round(float(value), 2) for value in exit_offset],
        "exit_offset_magnitude": round(float(np.linalg.norm(exit_offset)), 2),
        "exit_tail": len(exit_tail),
        "radius_of_gyration": round(gyration, 2),
        "bounding_radius": round(float(spread.max()), 2),
        "body_radius": round(gyration, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accession", required=True)
    parser.add_argument("--chain", required=True)
    parser.add_argument("--active-site-residue", type=int, required=True)
    parser.add_argument(
        "--terminus",
        choices=["n_terminal", "c_terminal"],
        required=True,
        help="which terminus of the effector the linker leaves from",
    )
    parser.add_argument("--first-residue", type=int, default=None)
    parser.add_argument("--last-residue", type=int, default=None)
    parser.add_argument("--tail-cutoff", type=float, default=1.5)
    parser.add_argument("--name", default=None)
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    path = args.structures / f"{args.accession}.cif.gz"
    if not path.exists():
        print(f"{path} not found; run data/fetch_structures.py")
        return 1

    record = measure(
        path,
        args.chain,
        args.active_site_residue,
        args.terminus,
        args.first_residue,
        args.last_residue,
        args.tail_cutoff,
    )
    record["accession"] = args.accession
    record["effector"] = args.name or args.chain

    print(f"{record['effector']} from {args.accession} chain {args.chain}")
    print(f"  modelled residues        {record['modelled_residues']}")
    print(f"  folded body              {record['body_residues']}")
    print(f"  resolved tail beyond it  {record['resolved_tail']} residues")
    print(f"  fusion junction          {record['terminus']} at residue {record['junction_residue']}")
    print(f"  catalytic centre         residue {record['active_site_residue']}")
    print(f"  offset magnitude         {record['offset_magnitude']} A")
    print(f"  far junction             residue {record['exit_residue']}, "
          f"{record['exit_offset_magnitude']} A away, {record['exit_tail']} residues beyond")
    print(f"  radius of gyration       {record['radius_of_gyration']} A")
    print(f"  bounding radius          {record['bounding_radius']} A")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
