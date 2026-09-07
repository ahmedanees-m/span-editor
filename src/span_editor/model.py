"""Assembly of one architecture into a prediction.

An ArchitectureContext holds everything about a corpus entry that does not
depend on the fitted parameters: the anchor frame, the excluded-volume map, the
ordered nucleotides of the displaced strand, the effector and its motif term.
The substrate ensemble does depend on one fitted parameter, and on whether an
inserted effector stands in the strand's way, so it is built on demand and
cached against both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import (
    NUMBERING_FROM_PAM_DISTAL,
    Frame,
    OccupancyMap,
    StrandAnchor,
    SubstrateEnsemble,
)
from .io import AssignedInputs, CorpusEntry
from .occupancy import ModelParameters, MotifPreference, Profile, predict_profile
from .structures import (
    StructureGeometry,
    coarse_occupancy,
    load_structure,
    read_anchor_frame,
    read_closure_anchor,
    read_observed_bubble,
    read_strand_anchors,
)
from .tether import EffectorSpec, TetherClosure, TetherSampler

__all__ = ["ArchitectureContext", "SpanModel", "build_context"]


@dataclass
class ArchitectureContext:
    """Everything needed to score one entry, minus the fitted parameters."""

    entry: CorpusEntry
    anchor: Frame
    strand_anchors: list[StrandAnchor]
    effector: EffectorSpec
    indices: list[int]
    occupancy: OccupancyMap | None = None
    motif: MotifPreference | None = None
    sequence: str | None = None
    closure: TetherClosure | None = None
    observed_bubble: dict = field(default_factory=dict)
    observed_spread: float = 0.0
    surface_distance: float = 0.0
    rise: float = 6.3
    n_samples: int = 2000
    seed: int = 0
    _ensembles: dict[tuple, SubstrateEnsemble] = field(default_factory=dict, repr=False)

    @property
    def spacer_length(self) -> int:
        return self.entry.spacer_length

    @property
    def numbering(self) -> str:
        return self.entry.numbering or NUMBERING_FROM_PAM_DISTAL

    def substrate(
        self,
        persistence: float,
        obstacle: tuple[np.ndarray, float] | None = None,
        anchor_jitter: float = 0.0,
    ) -> SubstrateEnsemble:
        """The displaced strand, optionally with a further sphere in its way.

        A domain fused to a terminus sits outside the R-loop and the strand
        never meets it. A domain inserted in place of one the editor used to
        carry sits inside the cavity the strand has to cross, so it obstructs
        the substrate as much as the protein does. Leaving it out would open a
        volume the construct does not actually open.
        """
        centre = None if obstacle is None else np.round(np.asarray(obstacle[0], float), 3)
        key = (
            round(float(persistence), 6),
            None if centre is None else (tuple(centre.tolist()), round(float(obstacle[1]), 3)),
            round(float(anchor_jitter), 3),
            round(float(self.observed_spread), 3),
            round(float(self.surface_distance), 3),
        )
        if key not in self._ensembles:
            occupancy = self.occupancy
            if obstacle is not None and occupancy is not None:
                occupancy = OccupancyMap(
                    centres=np.vstack([occupancy.centres, np.asarray(obstacle[0], float)[None, :]]),
                    radii=np.append(occupancy.radii, float(obstacle[1])),
                    probe_radius=occupancy.probe_radius,
                )
            self._ensembles[key] = SubstrateEnsemble(
                anchors=list(self.strand_anchors),
                spacer_length=self.spacer_length,
                numbering=self.numbering,
                rise=self.rise,
                persistence=key[0],
                n_samples=self.n_samples,
                seed=self.seed,
                occupancy=occupancy,
                anchor_jitter=float(anchor_jitter),
                observed=self.observed_bubble,
                observed_spread=float(self.observed_spread),
                surface_distance=float(self.surface_distance),
            )
        return self._ensembles[key]

    def substrate_excluding(
        self, index: int, persistence: float, anchor_jitter: float = 0.0
    ) -> SubstrateEnsemble:
        """The ensemble built without the anchor at one position.

        Scoring a position against an ensemble that was told where that position
        is measures the deposition, not the model. Dropping the anchor and
        letting the polymer interpolate it from its neighbours makes the value at
        that position a prediction, while everything the structure genuinely
        determines elsewhere is kept.

        Applied at every scored position in turn this is a leave-one-out
        profile, and it is the only construction under which two orthologs whose
        depositions resolve different amounts of the strand are compared on the
        same terms.
        """
        key = ("loo", int(index), round(float(persistence), 6), round(float(anchor_jitter), 3))
        if key not in self._ensembles:
            kept = [anchor for anchor in self.strand_anchors if anchor.index != index]
            if not kept:
                raise ValueError(f"dropping index {index} leaves the strand with no anchor")
            self._ensembles[key] = SubstrateEnsemble(
                anchors=kept,
                spacer_length=self.spacer_length,
                numbering=self.numbering,
                rise=self.rise,
                persistence=round(float(persistence), 6),
                n_samples=self.n_samples,
                seed=self.seed,
                occupancy=self.occupancy,
                anchor_jitter=float(anchor_jitter),
                observed={k: v for k, v in self.observed_bubble.items() if k != index},
                observed_spread=float(self.observed_spread),
                surface_distance=float(self.surface_distance),
            )
        return self._ensembles[key]

    def pam_distances(self) -> np.ndarray:
        substrate = next(iter(self._ensembles.values()), None)
        if substrate is None:
            substrate = self.substrate(15.0)
        return np.asarray([substrate.pam_distance(index) for index in self.indices], dtype=int)


class SpanModel:
    """The forward model: tether field, substrate ensemble, capture, motif, link."""

    def __init__(self, sampler: TetherSampler) -> None:
        self.sampler = sampler

    def predict_leave_one_out(
        self, context: ArchitectureContext, parameters: ModelParameters
    ) -> Profile:
        """A profile in which no position was scored against its own coordinate.

        One ensemble is built per scored position, without that position's
        anchor. It costs as many ensembles as there are positions and it is the
        construction that makes the comparison between orthologs mean what it
        claims, since a deposition that resolves most of its bubble no longer
        gets to answer for the positions being scored.
        """
        field_ = self.sampler.sample(
            linker=context.entry.linker,
            effector=context.effector,
            anchor=context.anchor,
            occupancy=context.occupancy,
            structure_id=context.entry.structure_id or "none",
            closure=context.closure,
        )
        # Dropping an anchor that has no anchor on one side of it does not leave
        # the position to be interpolated, it leaves the strand unbounded from
        # there on, and an unbounded overhang is a weaker constraint than every
        # other position gets rather than a fair one. Only interior anchors are
        # dropped; a position that was never an anchor was never read, so its
        # ensemble is the ordinary one.
        anchored = sorted(anchor.index for anchor in context.strand_anchors)
        interior = set(anchored[1:-1]) if len(anchored) > 2 else set()

        values = np.zeros(len(context.indices))
        capture = np.zeros(len(context.indices))
        for slot, index in enumerate(context.indices):
            if index in anchored and index not in interior:
                values[slot] = np.nan
                capture[slot] = np.nan
                continue
            try:
                substrate = (
                    context.substrate_excluding(index, parameters.ssdna_persistence)
                    if index in interior
                    else context.substrate(parameters.ssdna_persistence)
                )
            except ValueError:
                # Dropping this anchor can leave its neighbours further apart
                # than the chain between them can reach, which happens where the
                # deposition has a gap. Stretching the rise to make it fit would
                # be tuning a physical constant to a sampling convenience, so the
                # position is reported as unscoreable instead.
                values[slot] = np.nan
                capture[slot] = np.nan
                continue
            profile = predict_profile(
                tether=field_,
                substrate=substrate,
                parameters=parameters,
                motif=context.motif,
                sequence=context.sequence,
                indices=[index],
            )
            values[slot] = float(profile.values[0])
            # Carried so the mode is read from capture rather than from a linked
            # value that can underflow to exactly 1.0. See Profile.ranking.
            capture[slot] = (
                float(profile.capture[0]) if profile.capture is not None else np.nan
            )
        keep = ~np.isnan(values)
        return Profile(
            indices=np.asarray(context.indices, dtype=int)[keep],
            values=values[keep],
            capture=capture[keep] if not np.isnan(capture[keep]).any() else None,
        )

    def predict(self, context: ArchitectureContext, parameters: ModelParameters) -> Profile:
        field_ = self.sampler.sample(
            linker=context.entry.linker,
            effector=context.effector,
            anchor=context.anchor,
            occupancy=context.occupancy,
            structure_id=context.entry.structure_id or "none",
            closure=context.closure,
        )
        obstacle = None
        if context.closure is not None and field_.body_centre is not None:
            obstacle = (field_.body_centre, field_.body_radius)
        substrate = context.substrate(parameters.ssdna_persistence, obstacle)
        return predict_profile(
            tether=field_,
            substrate=substrate,
            parameters=parameters,
            motif=context.motif,
            sequence=context.sequence,
            indices=context.indices,
        )


def build_context(
    entry: CorpusEntry,
    assigned: AssignedInputs,
    structure_dir: Path | str = "data/structures",
) -> ArchitectureContext:
    """Assemble one corpus entry against its deposited structure.

    Every choice this needs beyond the file itself is an assigned input: which
    residue the effector attaches to, which chain carries the displaced strand,
    and which of its nucleotides are ordered. A missing assignment stops the
    build instead of falling back to a default.

    An entry that records a return site is an insertion rather than a terminal
    fusion, and picks up a second attachment on the residue the return linker
    runs back to.
    """
    accession = entry.structure_id
    if not accession:
        raise ValueError(f"{entry.entry_id}: no structure recorded")

    settings = assigned.structure_geometry.get(accession)
    if settings is None:
        raise ValueError(f"{entry.entry_id}: {accession} has no entry under structure_geometry")
    geometry = StructureGeometry.from_config(accession, settings)

    path = Path(structure_dir) / f"{accession}.cif.gz"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run data/fetch_structures.py")
    structure = load_structure(path)

    anchor = read_anchor_frame(
        structure, geometry.anchor_chain, geometry.residue_for(entry.anchor_site)
    )
    variant = assigned.scaffold_variants.get(entry.scaffold_variant, {})
    removed = {
        chain: [tuple(span) for span in spans]
        for chain, spans in (variant.get("remove") or {}).items()
    }
    if entry.scaffold_variant and not removed:
        raise ValueError(
            f"{entry.entry_id}: no residue ranges recorded for scaffold variant "
            f"{entry.scaffold_variant!r}"
        )
    occupancy = coarse_occupancy(
        structure,
        assigned.coarse_radii,
        chains=geometry.occluding_chains,
        frame=anchor,
        removed=removed,
        resolution=assigned.occupancy_resolution,
    )
    strand = read_strand_anchors(
        structure,
        geometry.strand_chain,
        geometry.strand_residue_to_index,
        frame=anchor,
        holds=geometry.holds,
    )
    observed_bubble = read_observed_bubble(
        structure,
        geometry.strand_chain,
        geometry.strand_residue_to_index,
        geometry.holds,
        frame=anchor,
    )
    closure = read_closure_anchor(structure, geometry, frame=anchor, occupancy=occupancy)
    if closure is not None and all(a.index != closure.index for a in strand):
        strand = sorted(strand + [closure], key=lambda item: item.index)

    effector = assigned.effectors.get(entry.effector)
    if effector is None:
        raise ValueError(f"{entry.entry_id}: no effector assigned for {entry.effector!r}")

    closure = None
    if entry.is_insertion:
        if effector.exit_offset is None:
            raise ValueError(
                f"{entry.entry_id}: {entry.effector!r} has no exit offset, "
                "so it cannot be modelled as an insertion"
            )
        returned = read_anchor_frame(
            structure,
            geometry.anchor_chain,
            geometry.residue_for(entry.return_site),
            frame=anchor,
        )
        closure = TetherClosure(
            point=returned.origin,
            n_links=entry.return_residues + max(0, effector.exit_tail),
            composition=entry.linker_composition,
        )

    return ArchitectureContext(
        entry=entry,
        anchor=anchor,
        strand_anchors=strand,
        effector=effector,
        indices=list(range(1, entry.spacer_length + 1)),
        occupancy=occupancy,
        motif=assigned.motifs.get(effector.motif),
        closure=closure,
        observed_bubble=observed_bubble,
        observed_spread=float(assigned.observed_spread),
        surface_distance=float(assigned.surface_distance),
        rise=assigned.substrate_rise,
        n_samples=assigned.sampling.get("substrate_samples", 2000),
        seed=assigned.sampling.get("seed", 0),
    )
