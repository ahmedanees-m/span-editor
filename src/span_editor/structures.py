"""Turning a deposited structure into the geometry the model works in.

Three things are read from each structure: the frame of the fusion attachment
residue, a coarse-grained occupancy map of the protein, and the coordinates of
the non-target strand nucleotides that are ordered enough to pin the substrate
ensemble. Which residue is the anchor, and which nucleotides count as ordered,
are assigned inputs rather than anything the file can decide, so they are read
from configs/assigned_inputs.yaml and a missing assignment is an error rather
than a default.
"""

from __future__ import annotations

import gzip
import io
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import Frame, OccupancyMap, StrandAnchor, frame_from_backbone

__all__ = [
    "SIZE_CLASS",
    "StructureGeometry",
    "coarse_occupancy",
    "read_observed_bubble",
    "duplex_partner_offset",
    "entity_descriptions",
    "load_structure",
    "read_block",
    "read_anchor_frame",
    "read_closure_anchor",
    "read_strand_anchors",
    "sequence_table",
]

SIZE_CLASS = {
    "GLY": "glycine",
    "ALA": "small",
    "SER": "small",
    "CYS": "small",
    "PRO": "small",
    "THR": "medium",
    "VAL": "medium",
    "ASN": "medium",
    "ASP": "medium",
    "ILE": "medium",
    "LEU": "medium",
    "MET": "medium",
    "GLN": "medium",
    "GLU": "medium",
    "LYS": "large",
    "ARG": "large",
    "HIS": "large",
    "PHE": "large",
    "TYR": "large",
    "TRP": "large",
}


@dataclass(frozen=True)
class StructureGeometry:
    """Per-structure assignments needed to place the model in the file."""

    accession: str
    anchor_chain: str
    anchor_residue: int
    strand_chain: str
    strand_residue_to_index: dict[int, int]
    anchor_sites: dict[str, int] = field(default_factory=dict)
    protospacer_residues: tuple[int, int] | None = None
    bubble: tuple[int, int] | None = None
    occluding_chains: tuple[str, ...] = ()
    closure_chain: str = ""
    closure_residue: int = 0
    closure_index: int = 0
    closure_pairing_sum: int = 0

    @classmethod
    def from_config(cls, accession: str, values: dict) -> "StructureGeometry":
        missing = [
            key
            for key in ("anchor_chain", "anchor_residue", "strand_chain", "strand_residue_to_index")
            if not values.get(key)
        ]
        if missing:
            raise ValueError(
                f"{accession}: assignments still to be made in assigned_inputs.yaml: "
                + ", ".join(missing)
            )
        return cls(
            accession=accession,
            anchor_chain=str(values["anchor_chain"]),
            anchor_residue=int(values["anchor_residue"]),
            strand_chain=str(values["strand_chain"]),
            strand_residue_to_index={
                int(residue): int(index)
                for residue, index in values["strand_residue_to_index"].items()
            },
            anchor_sites={
                str(name): int(residue)
                for name, residue in (values.get("anchor_sites") or {}).items()
            },
            protospacer_residues=(
                tuple(int(v) for v in values["protospacer_residues"])
                if values.get("protospacer_residues")
                else None
            ),
            bubble=(tuple(int(v) for v in values["bubble"]) if values.get("bubble") else None),
            occluding_chains=tuple(values.get("occluding_chains", ())),
            closure_chain=str(values.get("distal_closure", {}).get("chain", "") or ""),
            closure_residue=int(values.get("distal_closure", {}).get("residue", 0) or 0),
            closure_index=int(values.get("distal_closure", {}).get("index", 0) or 0),
            closure_pairing_sum=int(
                values.get("distal_closure", {}).get("pairing_sum", 0) or 0
            ),
        )


    def holds(self, residue: int) -> bool:
        """Whether a nucleotide may be pinned at its deposited coordinate.

        Only base pairing holds a nucleotide in place. A residue inside the
        R-loop bubble is single-stranded whether or not the deposition resolved
        it, so it belongs to the sampler however good its density was.
        """
        if self.bubble is None:
            return True
        return not self.bubble[0] <= residue <= self.bubble[1]

    def residue_for(self, site: str) -> int:
        """Residue the tether attaches to for a named attachment site.

        A terminal fusion leaves the protein at a terminus; an inserted domain
        leaves it in the middle. Which residue that is depends on the construct,
        so the sites are named in the configuration and a name with no entry is
        an error rather than a fall back to the terminus.
        """
        if not self.anchor_sites:
            return self.anchor_residue
        if site not in self.anchor_sites:
            raise ValueError(
                f"{self.accession}: no residue recorded for attachment site {site!r}"
            )
        return int(self.anchor_sites[site])


def read_block(path: Path | str):
    """Parse a deposited file, gzipped or not, and return its data block."""
    from biotite.structure.io import pdbx

    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return pdbx.CIFFile.read(io.StringIO(handle.read())).block


def sequence_table(block) -> dict[str, str]:
    """One-letter sequences by author chain identifier, as the depositors gave them."""
    table = block["entity_poly"]
    sequences: dict[str, str] = {}
    for chains, sequence in zip(
        table["pdbx_strand_id"].as_array(str),
        table["pdbx_seq_one_letter_code_can"].as_array(str),
    ):
        cleaned = sequence.replace("\n", "").strip()
        for chain in chains.split(","):
            sequences[chain.strip()] = cleaned
    return sequences


def entity_descriptions(block) -> dict[str, str]:
    """Depositor descriptions by author chain identifier."""
    identifiers = list(block["entity"]["id"].as_array(str))
    descriptions = list(block["entity"]["pdbx_description"].as_array(str))
    by_entity = dict(zip(identifiers, descriptions))

    table = block["entity_poly"]
    result: dict[str, str] = {}
    for entity, chains in zip(
        table["entity_id"].as_array(str), table["pdbx_strand_id"].as_array(str)
    ):
        for chain in chains.split(","):
            result[chain.strip()] = by_entity.get(entity, "")
    return result


def load_structure(path: Path | str, model: int = 1):
    """Read a deposited structure, gzipped or not, and return its first model."""
    from biotite.structure.io import pdbx

    return pdbx.get_structure(read_block(path), model=model)


def read_anchor_frame(structure, chain: str, residue: int, frame: Frame | None = None) -> Frame:
    """Frame on the backbone of the fusion attachment residue."""
    selection = structure[(structure.chain_id == chain) & (structure.res_id == residue)]
    if selection.array_length() == 0:
        raise ValueError(f"residue {residue} of chain {chain} is absent from the structure")

    atoms = {name: selection.coord[selection.atom_name == name] for name in ("N", "CA", "C")}
    for name, coordinates in atoms.items():
        if coordinates.shape[0] == 0:
            raise ValueError(f"atom {name} missing from residue {residue} of chain {chain}")

    built = frame_from_backbone(atoms["N"][0], atoms["CA"][0], atoms["C"][0])
    if frame is None:
        return built
    return Frame(origin=frame.to_local(built.origin)[0], basis=built.basis @ frame.basis.T)


ELEMENT_RADII = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "ZN": 1.39,
    "MG": 1.73,
}


def coarse_occupancy(
    structure,
    radii: dict[str, float],
    chains: tuple[str, ...] = (),
    frame: Frame | None = None,
    removed: dict[str, list[tuple[int, int]]] | None = None,
    resolution: str = "residue",
) -> OccupancyMap:
    """Spheres over the chains that obstruct the tether.

    At residue resolution there is one sphere per residue, which is enough to
    say what a domain cannot pass through and not enough to say what shape the
    channel between two domains has. A strand sampled against it stays against
    the surface but slides freely along it, because a union of residue-sized
    spheres has no grooves.

    At atom resolution every atom carries its van der Waals radius. That is
    eight or so times as many spheres and it resolves the channel, which is what
    a displaced strand actually runs in.
    
    Amino acids are placed on CB where it exists and CA otherwise. Nucleotides
    are placed on C1', with their own radius: the guide and the target strand
    are as much of an obstacle to a tethered domain as the protein is, and
    leaving them out would let sampled conformations pass through the duplex.

    The chain carrying the displaced strand is not passed in, since that is the
    substrate the effector is meant to reach rather than something in its way.

    removed names residue ranges to leave out, by chain. Deleting a domain is
    how an architecture that replaces one is modelled: the volume it occupied
    becomes available and nothing else changes, so any predicted change in the
    window comes from steric relief alone.
    """
    selection = structure
    if chains:
        selection = selection[np.isin(selection.chain_id, list(chains))]
    for chain, spans in (removed or {}).items():
        for first, last in spans:
            keep = ~(
                (selection.chain_id == chain)
                & (selection.res_id >= first)
                & (selection.res_id <= last)
            )
            selection = selection[keep]

    if resolution == "atom":
        heavy = selection[selection.element != "H"]
        if heavy.array_length() == 0:
            raise ValueError("no heavy atoms found for the requested chains")
        coordinates = np.asarray(heavy.coord, dtype=float)
        if frame is not None:
            coordinates = frame.to_local(coordinates)
        default = radii.get("atom_default", 1.70)
        sphere = np.asarray(
            [ELEMENT_RADII.get(str(e).upper(), default) for e in heavy.element], dtype=float
        )
        return OccupancyMap(centres=coordinates, radii=sphere)
    if resolution != "residue":
        raise ValueError(f"unknown occupancy resolution: {resolution!r}")

    centres, sizes = [], []
    seen = set()
    sources = [
        (selection[selection.atom_name == "CB"], None),
        (selection[selection.atom_name == "CA"], None),
        (selection[selection.atom_name == "C1'"], "nucleotide"),
    ]
    for source, forced in sources:
        for index in range(source.array_length()):
            key = (str(source.chain_id[index]), int(source.res_id[index]))
            if key in seen:
                continue
            seen.add(key)
            centres.append(source.coord[index])
            sizes.append(
                forced or SIZE_CLASS.get(str(source.res_name[index]).upper(), "default")
            )

    if not centres:
        raise ValueError("no backbone atoms found for the requested chains")

    coordinates = np.asarray(centres, dtype=float)
    if frame is not None:
        coordinates = frame.to_local(coordinates)
    default = radii.get("default", 3.0)
    return OccupancyMap(
        centres=coordinates,
        radii=np.asarray([radii.get(size, default) for size in sizes], dtype=float),
    )


def _strand_frame(position: dict[int, np.ndarray], residue: int):
    """Right-handed frame on a sugar, from its two neighbours along the strand."""
    if residue - 1 not in position or residue + 1 not in position:
        return None
    origin = position[residue]
    along = position[residue + 1] - position[residue - 1]
    along = along / np.linalg.norm(along)
    lateral = position[residue + 1] - origin
    lateral = lateral - np.dot(lateral, along) * along
    norm = np.linalg.norm(lateral)
    if norm < 1e-6:
        return None
    lateral = lateral / norm
    return origin, np.stack([along, lateral, np.cross(along, lateral)])


def duplex_partner_offset(
    structure,
    target_chain: str,
    displaced_chain: str,
    pairing_sum: int,
    cutoff: float = 12.5,
    atom_name: str = "C1'",
) -> tuple[float, np.ndarray, float] | None:
    """Separation and direction from a sugar to its base-paired partner.

    Measured over the duplex pairs a structure does resolve, so that a partner
    which is not resolved can be placed from the geometry of ones that are. The
    separation comes out tight; the direction is returned with its mean
    resultant length so a caller can see how well it is determined.
    """
    target = structure[
        (structure.chain_id == target_chain) & (structure.atom_name == atom_name)
    ]
    displaced = structure[
        (structure.chain_id == displaced_chain) & (structure.atom_name == atom_name)
    ]
    position = {int(r): c for r, c in zip(target.res_id, target.coord)}
    partners = {int(r): c for r, c in zip(displaced.res_id, displaced.coord)}

    separations, directions = [], []
    for residue in sorted(position):
        partner = pairing_sum - residue
        if partner not in partners:
            continue
        built = _strand_frame(position, residue)
        if built is None:
            continue
        origin, basis = built
        step = partners[partner] - origin
        separation = float(np.linalg.norm(step))
        if separation > cutoff:
            continue
        separations.append(separation)
        directions.append(basis @ (step / separation))

    if len(separations) < 3:
        return None
    stacked = np.asarray(directions)
    mean = stacked.mean(axis=0)
    resultant = float(np.linalg.norm(mean))
    if resultant < 1e-6:
        return None
    return float(np.mean(separations)), mean / resultant, resultant


def read_closure_anchor(
    structure,
    geometry: "StructureGeometry",
    frame: Frame | None = None,
    occupancy: OccupancyMap | None = None,
    atom_name: str = "C1'",
) -> StrandAnchor | None:
    """Anchor marking where the R-loop closes at the PAM-distal end.

    Beyond the protospacer the displaced strand re-anneals with the target
    strand, so its PAM-distal end is not free. No structure resolves that
    nucleotide, but the target strand it pairs with is resolved, and the
    position of that partner bounds the bubble. Leaving the end unconstrained
    instead lets it wander about forty angstrom, which no genomic R-loop allows.

    The anchor sits one base-pair separation from the partner's sugar, in the
    direction that the resolved duplex pairs of the same structure put a partner.
    Both are measured there rather than assumed. Putting the anchor on the
    partner's own sugar instead would place it short by that separation.
    """
    if not (geometry.closure_chain and geometry.closure_residue and geometry.closure_index):
        return None
    selection = structure[
        (structure.chain_id == geometry.closure_chain)
        & (structure.res_id == geometry.closure_residue)
        & (structure.atom_name == atom_name)
    ]
    if selection.array_length() == 0:
        raise ValueError(
            f"{geometry.accession}: residue {geometry.closure_residue} of chain "
            f"{geometry.closure_chain} carries no {atom_name} atom"
        )
    position = selection.coord[0]

    if geometry.closure_pairing_sum:
        measured = duplex_partner_offset(
            structure,
            geometry.closure_chain,
            geometry.strand_chain,
            geometry.closure_pairing_sum,
            atom_name=atom_name,
        )
        if measured is not None:
            separation, direction, _ = measured
            target = structure[
                (structure.chain_id == geometry.closure_chain)
                & (structure.atom_name == atom_name)
            ]
            sugars = {int(r): c for r, c in zip(target.res_id, target.coord)}
            built = _strand_frame(sugars, geometry.closure_residue)
            if built is None:
                built = _strand_frame(sugars, geometry.closure_residue - 1)
            if built is not None:
                preferred = built[1].T @ direction
                position = _place_at_separation(
                    position, preferred, separation, frame, occupancy
                )

    if frame is not None:
        position = frame.to_local(position)[0]
    return StrandAnchor(index=geometry.closure_index, position=position)


def _sphere_directions(count: int = 512) -> np.ndarray:
    """Roughly uniform directions on the sphere, by the golden-angle spiral."""
    index = np.arange(count) + 0.5
    cos_theta = 1.0 - 2.0 * index / count
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    phi = np.pi * (1.0 + 5.0**0.5) * index
    return np.stack(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta], axis=1
    )


def _place_at_separation(
    origin: np.ndarray,
    preferred: np.ndarray,
    separation: float,
    frame: Frame | None,
    occupancy: OccupancyMap | None,
) -> np.ndarray:
    """Put a point one separation from origin, as near the preferred direction as sterics allow.

    The separation is well determined by the duplex pairs a structure resolves.
    The direction is not: transferring it between local frames carries a base
    step of helical rotation and the pairs disagree by a median of seventeen
    degrees. Where the preferred direction would bury the point inside the
    protein, the nearest direction that does not is used instead, and where none
    is free the point stays on the partner's own sugar.
    """
    candidate = origin + preferred * separation
    if occupancy is None or frame is None:
        return candidate
    if not occupancy.occluded(frame.to_local(candidate))[0]:
        return candidate

    directions = _sphere_directions()
    order = np.argsort(-(directions @ preferred))
    points = origin + directions[order] * separation
    free = ~occupancy.occluded(frame.to_local(points))
    if np.any(free):
        return points[int(np.argmax(free))]
    return origin


def read_observed_bubble(
    structure,
    chain: str,
    residue_to_index: dict[int, int],
    holds,
    frame: Frame | None = None,
    atom_name: str = "C1'",
) -> dict[int, np.ndarray]:
    """Coordinates of the bubble nucleotides the deposition happens to resolve.

    These are not anchors. They are evidence about the path the displaced strand
    takes, and the ensemble weights sampled conformations by how closely they
    follow them.
    """
    selection = structure[(structure.chain_id == chain) & (structure.atom_name == atom_name)]
    observed: dict[int, np.ndarray] = {}
    for index in range(selection.array_length()):
        residue = int(selection.res_id[index])
        if residue not in residue_to_index or holds(residue):
            continue
        position = selection.coord[index]
        if frame is not None:
            position = frame.to_local(position)[0]
        observed[residue_to_index[residue]] = position
    return observed


def read_strand_anchors(
    structure,
    chain: str,
    residue_to_index: dict[int, int],
    frame: Frame | None = None,
    atom_name: str = "C1'",
    holds=None,
) -> list[StrandAnchor]:
    """Nucleotides of the displaced strand that may anchor the ensemble.

    holds decides which of the resolved nucleotides are held in place. Without
    it every resolved nucleotide anchors, which is the wrong rule: it pins the
    inside of the R-loop bubble wherever a deposition happened to order it, and
    how much of the bubble that is differs from one structure to the next, so
    the ensemble does different amounts of work for different orthologs.
    """
    selection = structure[(structure.chain_id == chain) & (structure.atom_name == atom_name)]
    if selection.array_length() == 0:
        raise ValueError(f"chain {chain} carries no {atom_name} atoms")

    anchors = []
    for index in range(selection.array_length()):
        residue = int(selection.res_id[index])
        if residue not in residue_to_index:
            continue
        if holds is not None and not holds(residue):
            continue
        position = selection.coord[index]
        if frame is not None:
            position = frame.to_local(position)[0]
        anchors.append(StrandAnchor(index=residue_to_index[residue], position=position))

    if not anchors:
        raise ValueError(f"none of the assigned residues were found in chain {chain}")
    return sorted(anchors, key=lambda anchor: anchor.index)
