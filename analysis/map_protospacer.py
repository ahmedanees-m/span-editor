"""Locate the protospacer on the displaced strand of a deposited structure.

Writes the residue to protospacer index map used by the model.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from span_editor.geometry import NUMBERING_FROM_PAM_DISTAL, NUMBERING_FROM_PAM_PROXIMAL
from span_editor.structures import load_structure, read_block, sequence_table

COMPLEMENT = str.maketrans("ACGTU", "TGCAA")
PAM_THREE_PRIME = "3prime"
PAM_FIVE_PRIME = "5prime"


@dataclass
class Mapping:
    accession: str
    guide_chain: str
    strand_chain: str
    spacer: str
    protospacer: str
    first_residue: int
    last_residue: int
    numbering: str
    pam_side: str
    pam: str
    matches: int
    index_one_at_first_residue: bool = True
    ordered_indices: list[int] = field(default_factory=list)
    displaced_indices: list[int] = field(default_factory=list)
    reannealed_indices: list[int] = field(default_factory=list)
    modified_residues: dict[int, str] = field(default_factory=dict)

    def index_of(self, residue: int) -> int:
        if self.index_one_at_first_residue:
            return residue - self.first_residue + 1
        return self.last_residue - residue + 1

    def residue_of(self, index: int) -> int:
        if self.index_one_at_first_residue:
            return index + self.first_residue - 1
        return self.last_residue - index + 1

    def as_dict(self) -> dict:
        return {
            "accession": self.accession,
            "guide_chain": self.guide_chain,
            "strand_chain": self.strand_chain,
            "spacer": self.spacer,
            "protospacer": self.protospacer,
            "residue_range": [self.first_residue, self.last_residue],
            "numbering": self.numbering,
            "pam_side": self.pam_side,
            "pam": self.pam,
            "spacer_matches": self.matches,
            "index_one_at_first_residue": self.index_one_at_first_residue,
            "ordered_indices": self.ordered_indices,
            "displaced_indices": self.displaced_indices,
            "reannealed_indices": self.reannealed_indices,
            "modified_residues": {
                str(residue): name for residue, name in self.modified_residues.items()
            },
            "strand_residue_to_index": {
                str(self.residue_of(index)): index for index in self.ordered_indices
            },
        }


def best_alignment(spacer: str, strand: str) -> tuple[int, int]:
    """Offset of the best ungapped match of the spacer in the strand, and its score."""
    window = len(spacer)
    best, score = -1, -1
    for offset in range(len(strand) - window + 1):
        candidate = strand[offset : offset + window]
        matches = sum(a == b for a, b in zip(candidate, spacer))
        if matches > score:
            best, score = offset, matches
    return best, score


def auth_offset(block, chain: str) -> int:
    """Author residue number minus position in the deposited sequence."""
    table = block["atom_site"]
    chains = table["auth_asym_id"].as_array(str)
    auth = table["auth_seq_id"].as_array(str)
    label = table["label_seq_id"].as_array(str)
    mask = chains == chain
    offsets = {
        int(a) - int(l) for a, l in zip(auth[mask], label[mask]) if l not in (".", "?")
    }
    if len(offsets) != 1:
        raise ValueError(f"chain {chain} has an inconsistent numbering offset: {offsets}")
    return offsets.pop()


def map_structure(
    accession: str,
    guide_chain: str,
    strand_chain: str,
    spacer_length: int,
    structure_dir: Path,
    numbering: str = NUMBERING_FROM_PAM_DISTAL,
    pam_side: str = PAM_THREE_PRIME,
    pam_length: int = 3,
) -> Mapping:
    path = Path(structure_dir) / f"{accession}.cif.gz"
    block = read_block(path)
    sequences = sequence_table(block)
    guide = sequences[guide_chain].replace("U", "T")
    strand = sequences[strand_chain]

    # A Cas9 single guide begins with its spacer; a Cas12a crRNA ends with it.
    spacer = guide[:spacer_length] if pam_side == PAM_THREE_PRIME else guide[-spacer_length:]

    forward, forward_score = best_alignment(spacer, strand)
    reverse_strand = strand.translate(COMPLEMENT)[::-1]
    _, reverse_score = best_alignment(spacer, reverse_strand)
    if reverse_score > forward_score:
        raise ValueError(
            f"{accession}: the spacer matches the reverse complement of chain "
            f"{strand_chain} better than the strand itself, so that chain is "
            f"probably the target strand rather than the displaced one"
        )

    shift = auth_offset(block, strand_chain)
    first = forward + 1 + shift
    last = forward + spacer_length + shift

    if pam_side == PAM_THREE_PRIME:
        pam = strand[forward + spacer_length : forward + spacer_length + pam_length]
    else:
        pam = strand[max(0, forward - pam_length) : forward]

    structure = load_structure(path)
    selection = structure[
        (structure.chain_id == strand_chain) & (structure.atom_name == "C1'")
    ]
    present = {int(value) for value in selection.res_id}
    names = {int(r): str(n) for r, n in zip(selection.res_id, selection.res_name)}

    mapping = Mapping(
        accession=accession,
        guide_chain=guide_chain,
        strand_chain=strand_chain,
        spacer=spacer,
        protospacer=strand[forward : forward + spacer_length],
        first_residue=first,
        last_residue=last,
        numbering=numbering,
        pam_side=pam_side,
        pam=pam,
        matches=forward_score,
        index_one_at_first_residue=(numbering == NUMBERING_FROM_PAM_DISTAL)
        == (pam_side == PAM_THREE_PRIME),
    )
    mapping.ordered_indices = sorted(
        mapping.index_of(residue) for residue in range(first, last + 1) if residue in present
    )
    standard = {"DA", "DC", "DG", "DT"}
    mapping.modified_residues = {
        residue: names[residue]
        for residue in range(first, last + 1)
        if residue in names and names[residue] not in standard
    }
    return mapping


def read_deposited_register(
    accession: str,
    guide_chain: str,
    strand_chain: str,
    target_chain: str,
    spacer_length: int,
    structure_dir: Path,
    numbering: str = NUMBERING_FROM_PAM_PROXIMAL,
    pam_side: str = PAM_FIVE_PRIME,
    pam_length: int = 4,
    pairing_cutoff: float = 12.5,
) -> Mapping:
    """Take the register from the deposited numbering and check it three ways.

    Some depositions pair the guide with a displaced strand that is not
    complementary to the target strand, so that the R-loop cannot re-anneal
    during crystallisation. The spacer then cannot be aligned against it and the
    register has to come from the numbering the depositors used, which places
    protospacer position 1 at author residue 1.

    Three checks stand in for the alignment. The nucleotides immediately beyond
    residue 1 must read as the family PAM. The residues on the PAM side must sit
    at base-pairing distance from the target strand while the protospacer
    residues do not. And the strand must re-anneal again beyond the R-loop.
    """
    path = Path(structure_dir) / f"{accession}.cif.gz"
    block = read_block(path)
    sequences = sequence_table(block)
    strand = sequences[strand_chain]
    shift = auth_offset(block, strand_chain)

    def base_at(residue: int) -> str:
        position = residue - shift
        return strand[position - 1] if 1 <= position <= len(strand) else ""

    if pam_side == PAM_FIVE_PRIME:
        pam = "".join(base_at(residue) for residue in range(1 - pam_length, 1))
    else:
        pam = "".join(
            base_at(residue) for residue in range(spacer_length + 1, spacer_length + 1 + pam_length)
        )

    structure = load_structure(path)
    displaced = structure[
        (structure.chain_id == strand_chain) & (structure.atom_name == "C1'")
    ]
    target = structure[(structure.chain_id == target_chain) & (structure.atom_name == "C1'")]

    present, paired = set(), set()
    for row in range(displaced.array_length()):
        residue = int(displaced.res_id[row])
        present.add(residue)
        separations = np.linalg.norm(target.coord - displaced.coord[row], axis=1)
        if separations.size and separations.min() < pairing_cutoff:
            paired.add(residue)

    mapping = Mapping(
        accession=accession,
        guide_chain=guide_chain,
        strand_chain=strand_chain,
        spacer=sequences[guide_chain].replace("U", "T")[-spacer_length:],
        protospacer="".join(base_at(residue) for residue in range(1, spacer_length + 1)),
        first_residue=1,
        last_residue=spacer_length,
        numbering=numbering,
        pam_side=pam_side,
        pam=pam,
        matches=-1,
        index_one_at_first_residue=True,
    )
    mapping.ordered_indices = sorted(
        residue for residue in range(1, spacer_length + 1) if residue in present
    )
    mapping.displaced_indices = sorted(
        residue
        for residue in range(1, spacer_length + 1)
        if residue in present and residue not in paired
    )
    mapping.reannealed_indices = sorted(
        residue for residue in range(1, spacer_length + 1) if residue in paired
    )
    standard = {"DA", "DC", "DG", "DT"}
    names = {int(r): str(n) for r, n in zip(displaced.res_id, displaced.res_name)}
    mapping.modified_residues = {
        residue: names[residue]
        for residue in range(1, spacer_length + 1)
        if residue in names and names[residue] not in standard
    }
    return mapping


def check_geometry(accession: str, mapping: Mapping, structure_dir: Path) -> dict:
    """The PAM-proximal end must sit deeper in the protein than the PAM-distal end."""
    structure = load_structure(Path(structure_dir) / f"{accession}.cif.gz")
    protein = structure[structure.atom_name == "CA"]
    strand = structure[
        (structure.chain_id == mapping.strand_chain) & (structure.atom_name == "C1'")
    ]
    if protein.array_length() == 0 or strand.array_length() == 0:
        return {"available": False}

    centre = protein.coord.mean(axis=0)
    distances = {}
    for index in mapping.ordered_indices:
        residue = mapping.residue_of(index)
        match = strand.coord[strand.res_id == residue]
        if match.shape[0]:
            distances[index] = float(np.linalg.norm(match[0] - centre))
    if len(distances) < 2:
        return {"available": False}

    ordered = sorted(distances)
    return {
        "available": True,
        "distance_from_protein_centre": distances,
        "lowest_index": ordered[0],
        "highest_index": ordered[-1],
        "pam_proximal_is_deeper": (
            distances[ordered[-1]] < distances[ordered[0]]
            if mapping.numbering == NUMBERING_FROM_PAM_DISTAL
            else distances[ordered[0]] < distances[ordered[-1]]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accession")
    parser.add_argument("--guide-chain", required=True)
    parser.add_argument("--strand-chain", required=True)
    parser.add_argument("--spacer-length", type=int, default=20)
    parser.add_argument(
        "--numbering",
        choices=[NUMBERING_FROM_PAM_DISTAL, NUMBERING_FROM_PAM_PROXIMAL],
        default=NUMBERING_FROM_PAM_DISTAL,
    )
    parser.add_argument(
        "--pam-side", choices=[PAM_THREE_PRIME, PAM_FIVE_PRIME], default=PAM_THREE_PRIME
    )
    parser.add_argument("--pam-length", type=int, default=3)
    parser.add_argument(
        "--register",
        choices=["spacer_alignment", "deposited"],
        default="spacer_alignment",
        help="deposited takes position 1 from author residue 1 and validates it",
    )
    parser.add_argument("--target-chain", default=None)
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.register == "deposited":
        if not args.target_chain:
            parser.error("--target-chain is required with --register deposited")
        mapping = read_deposited_register(
            args.accession,
            args.guide_chain,
            args.strand_chain,
            args.target_chain,
            args.spacer_length,
            args.structures,
            args.numbering,
            args.pam_side,
            args.pam_length,
        )
    else:
        mapping = map_structure(
            args.accession,
            args.guide_chain,
            args.strand_chain,
            args.spacer_length,
            args.structures,
            args.numbering,
            args.pam_side,
            args.pam_length,
        )
    geometry = check_geometry(args.accession, mapping, args.structures)

    print(
        f"{mapping.accession}: guide chain {mapping.guide_chain}, "
        f"displaced strand chain {mapping.strand_chain}"
    )
    print(f"  spacer        {mapping.spacer}")
    print(
        f"  protospacer   {mapping.protospacer}  ({mapping.matches} of "
        f"{len(mapping.spacer)} positions match)"
    )
    print(f"  residues      {mapping.first_residue} to {mapping.last_residue}")
    print(f"  pam           {mapping.pam} on the {mapping.pam_side} side")
    print(f"  numbering     {mapping.numbering}")
    print(f"  position 1    {'first' if mapping.index_one_at_first_residue else 'last'} residue")
    for residue, name in mapping.modified_residues.items():
        print(
            f"  modified base {name} at residue {residue}, "
            f"protospacer index {mapping.index_of(residue)}"
        )
    print(f"  ordered       {mapping.ordered_indices}")
    if mapping.displaced_indices:
        print(f"  displaced     {mapping.displaced_indices}")
        print(f"  re-annealed   {mapping.reannealed_indices}")
    if geometry.get("available") and not mapping.displaced_indices:
        verdict = "yes" if geometry["pam_proximal_is_deeper"] else "no"
        print(f"  pam-proximal end sits deeper in the protein: {verdict}")
    elif geometry.get("available"):
        depths = geometry["distance_from_protein_centre"]
        displaced = [depths[i] for i in mapping.displaced_indices if i in depths]
        annealed = [depths[i] for i in mapping.reannealed_indices if i in depths]
        if displaced and annealed:
            print(
                f"  mean depth: displaced {sum(displaced) / len(displaced):.1f} A, "
                f"re-annealed {sum(annealed) / len(annealed):.1f} A"
            )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"mapping": mapping.as_dict(), "geometry": geometry}, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
