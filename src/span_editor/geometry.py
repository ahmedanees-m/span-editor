"""Anchor frames, substrate ensembles and excluded volume.

Everything downstream works in the anchor frame: a right-handed frame placed on
the backbone of the fusion attachment residue of the substrate-bound editor
structure. The displaced non-target strand is represented as a positional
ensemble rather than a set of coordinates, since the segment that carries the
editing window is only partly ordered in R-loop structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .polymer import sample_bridge, sample_free_chain

__all__ = [
    "Frame",
    "OccupancyMap",
    "StrandAnchor",
    "SubstrateEnsemble",
    "frame_from_backbone",
    "pam_distance",
]

NUMBERING_FROM_PAM_DISTAL = "pam_distal"
NUMBERING_FROM_PAM_PROXIMAL = "pam_proximal"


@dataclass(frozen=True)
class Frame:
    """A rigid frame given by an origin and an orthonormal basis in rows."""

    origin: np.ndarray
    basis: np.ndarray

    def to_local(self, points: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(points) - self.origin) @ self.basis.T

    def to_global(self, points: np.ndarray) -> np.ndarray:
        return np.atleast_2d(points) @ self.basis + self.origin

    def rotate_to_local(self, vectors: np.ndarray) -> np.ndarray:
        return np.atleast_2d(vectors) @ self.basis.T


def frame_from_backbone(
    nitrogen: np.ndarray, alpha_carbon: np.ndarray, carbon: np.ndarray
) -> Frame:
    """Build an anchor frame from the N, CA and C atoms of one residue.

    The first axis runs along CA to C, the second lies in the N-CA-C plane and
    the third completes a right-handed set. Fusion linkers leave the chain at
    the terminal residue, so this frame fixes both the position and the
    direction in which the tether departs.
    """
    origin = np.asarray(alpha_carbon, dtype=float)
    first = np.asarray(carbon, dtype=float) - origin
    first /= np.linalg.norm(first)
    in_plane = np.asarray(nitrogen, dtype=float) - origin
    second = in_plane - np.dot(in_plane, first) * first
    second /= np.linalg.norm(second)
    third = np.cross(first, second)
    return Frame(origin=origin, basis=np.stack([first, second, third]))


def pam_distance(index: int, spacer_length: int, numbering: str) -> int:
    """Convert a protospacer index into nucleotides from the PAM-proximal end.

    Cas9 windows are quoted from the PAM-distal end and Cas12a windows from the
    PAM-proximal end. Reporting both makes it possible to separate a genuine
    relocation of the window from a change of convention.
    """
    if numbering == NUMBERING_FROM_PAM_DISTAL:
        return spacer_length - index + 1
    if numbering == NUMBERING_FROM_PAM_PROXIMAL:
        return index
    raise ValueError(f"unknown numbering convention: {numbering}")


class OccupancyMap:
    """Coarse-grained excluded volume as a union of spheres.

    Queries are answered from a uniform grid, so a chain of a few thousand
    joints can be tested against a Cas protein without a spatial index library.
    """

    def __init__(
        self,
        centres: np.ndarray,
        radii: np.ndarray | float,
        probe_radius: float = 0.0,
    ) -> None:
        self.centres = np.atleast_2d(np.asarray(centres, dtype=float))
        if np.isscalar(radii):
            self.radii = np.full(self.centres.shape[0], float(radii))
        else:
            self.radii = np.asarray(radii, dtype=float)
        if self.radii.shape[0] != self.centres.shape[0]:
            raise ValueError("centres and radii must have the same length")
        self.probe_radius = float(probe_radius)
        self._reach = self.radii + self.probe_radius
        self._cell = float(max(self._reach.max(), 1.0))
        self._grid = self._build_grid()

    def _build_grid(self) -> dict[tuple[int, int, int], np.ndarray]:
        keys = np.floor(self.centres / self._cell).astype(np.int64)
        order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))
        grid: dict[tuple[int, int, int], list[int]] = {}
        for idx in order:
            grid.setdefault(tuple(keys[idx]), []).append(int(idx))
        return {key: np.asarray(value, dtype=np.int64) for key, value in grid.items()}

    def __len__(self) -> int:
        return self.centres.shape[0]

    def occluded(self, points: np.ndarray) -> np.ndarray:
        """Return a boolean per point saying whether it falls inside any sphere."""
        query = np.atleast_2d(np.asarray(points, dtype=float))
        hit = np.zeros(query.shape[0], dtype=bool)
        if not self._grid:
            return hit

        keys = np.floor(query / self._cell).astype(np.int64)
        packed, inverse = np.unique(keys, axis=0, return_inverse=True)
        offsets = np.array(
            [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        )
        for cell_id, key in enumerate(packed):
            members = np.flatnonzero(inverse == cell_id)
            candidates = [
                self._grid[neighbour]
                for neighbour in map(tuple, key + offsets)
                if neighbour in self._grid
            ]
            if not candidates:
                continue
            index = np.concatenate(candidates)
            delta = query[members][:, None, :] - self.centres[index][None, :, :]
            distance = np.linalg.norm(delta, axis=2)
            hit[members] = np.any(distance < self._reach[index][None, :], axis=1)
        return hit

    def chain_clashes(self, coords: np.ndarray, skip_first: int = 1) -> np.ndarray:
        """Return a boolean per chain saying whether any joint is occluded.

        The first joints leave the protein surface by construction, so
        skip_first suppresses the self-contact that the fusion residue itself
        would otherwise produce.
        """
        chains = np.asarray(coords, dtype=float)
        tested = chains[:, skip_first:, :]
        flat = tested.reshape(-1, 3)
        return self.occluded(flat).reshape(tested.shape[0], tested.shape[1]).any(axis=1)


@dataclass(frozen=True)
class StrandAnchor:
    """An ordered nucleotide of the displaced strand, resolved in the structure."""

    index: int
    position: np.ndarray


@dataclass
class SubstrateEnsemble:
    """Positional ensemble of the displaced non-target strand.

    Nucleotides between two ordered anchors are sampled as bridges; nucleotides
    beyond the outermost anchor are sampled as free chains. Coordinates are in
    the anchor frame of the editor.

    An anchor is a nucleotide held in place by base pairing, and it is fixed at
    the coordinate the deposition gives it. Nucleotides inside the R-loop bubble
    are not held, and are sampled between the anchors that bound it.

    A bubble nucleotide the deposition nonetheless resolves is neither free nor
    fixed. The coordinate is evidence about where the strand runs, and throwing
    it away replaces measurement with a free bridge; but holding the nucleotide
    at it gives a single-stranded base no conformational freedom, and how much
    of the bubble a deposition orders differs between structures, so pinning
    makes the ensemble do different amounts of work for different orthologs.

    observed carries those coordinates and observed_spread says how loosely to
    believe them. A sampled conformation is weighted by how close it passes to
    each of them, so the spread interpolates between the two treatments: at zero
    it recovers pinning, and as it grows the observation stops constraining
    anything and the bubble is a free bridge. It is an assigned input and results
    are reported across its range.

    surface_distance confines the strand to a band along the protein. Without it
    the ensemble only repels, and a long bubble spreads far enough to reach most
    of the protospacer whatever the tether does.

    anchor_jitter is separate and applies to the pairing-held anchors, and is a
    diagnostic rather than part of the model.
    """

    anchors: list[StrandAnchor]
    spacer_length: int
    numbering: str = NUMBERING_FROM_PAM_DISTAL
    rise: float = 6.3
    persistence: float = 15.0
    n_samples: int = 2000
    seed: int = 0
    occupancy: OccupancyMap | None = None
    anchor_jitter: float = 0.0
    observed: dict[int, np.ndarray] = field(default_factory=dict)
    observed_spread: float = 0.0
    surface_distance: float = 0.0
    _clouds: dict[int, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        if len(self.anchors) < 1:
            raise ValueError("at least one ordered anchor is required")
        self.anchors = sorted(self.anchors, key=lambda a: a.index)
        self._build()

    def _build(self) -> None:
        rng = np.random.default_rng(self.seed)
        for anchor in self.anchors:
            position = np.asarray(anchor.position, dtype=float)
            if self.anchor_jitter <= 0:
                # A nucleotide the deposition resolves is one point, not a
                # thousand copies of one point. Carrying it once is the same
                # distribution and makes every pairwise sum against it cheaper
                # by the sample count, which matters where most of the strand is
                # resolved.
                self._clouds[anchor.index] = (position[None, :], np.ones(1))
                continue
            cloud = np.tile(position, (self.n_samples, 1))
            weights = np.full(self.n_samples, 1.0 / self.n_samples)
            cloud = cloud + rng.normal(scale=self.anchor_jitter, size=cloud.shape)
            if self.occupancy is not None:
                weights = np.where(self.occupancy.occluded(cloud), 0.0, weights)
            self._clouds[anchor.index] = (cloud, self._normalise(weights, f"anchor {anchor.index}"))

        for left, right in zip(self.anchors[:-1], self.anchors[1:]):
            self._fill_bridge(left, right, rng)

        self._fill_overhang(self.anchors[0], direction=-1, rng=rng)
        self._fill_overhang(self.anchors[-1], direction=+1, rng=rng)

    def _fill_bridge(self, left: StrandAnchor, right: StrandAnchor, rng) -> None:
        span = right.index - left.index
        if span <= 1:
            return
        offset = np.asarray(right.position, dtype=float) - np.asarray(
            left.position, dtype=float
        )
        origin = np.asarray(left.position, dtype=float)
        sample = sample_bridge(
            n_links=span,
            link_length=self.rise,
            persistence=self.persistence,
            end_offset=offset,
            n_chains=self.n_samples,
            rng=rng,
            blocked=self._blocked(origin),
        )
        coords = sample.coords + origin
        weights = self._observed_weight(
            sample.weights, coords, left.index, span, f"bridge {left.index} to {right.index}"
        )
        for step in range(1, span):
            self._clouds[left.index + step] = (coords[:, step, :], weights)

    def _fill_overhang(self, anchor: StrandAnchor, direction: int, rng) -> None:
        if direction < 0:
            span = anchor.index - 1
            indices = [anchor.index - step for step in range(1, span + 1)]
        else:
            span = self.spacer_length - anchor.index
            indices = [anchor.index + step for step in range(1, span + 1)]
        if span <= 0:
            return
        origin = np.asarray(anchor.position, dtype=float)
        sample = sample_free_chain(
            n_links=span,
            link_length=self.rise,
            persistence=self.persistence,
            n_chains=self.n_samples,
            rng=rng,
            blocked=self._blocked(origin),
        )
        coords = sample.coords + origin
        weights = self._normalise(sample.weights, f"overhang from {anchor.index}")
        for step, index in enumerate(indices, start=1):
            self._clouds[index] = (coords[:, step, :], weights)

    def _observed_weight(
        self, weights: np.ndarray, coords: np.ndarray, first: int, span: int, label: str
    ) -> np.ndarray:
        """Reweight sampled chains by how closely they follow the resolved path.

        Each resolved bubble nucleotide contributes a Gaussian in the distance
        between where the chain put it and where the deposition did. With no
        resolved nucleotides in the span, or with the spread left at zero, the
        weights are the sampler's own and the bridge is free.
        """
        if not self.observed or self.observed_spread <= 0:
            return self._normalise(weights, label)
        adjusted = np.asarray(weights, dtype=float).copy()
        variance = float(self.observed_spread) ** 2
        for step in range(1, span):
            position = self.observed.get(first + step)
            if position is None:
                continue
            offset = coords[:, step, :] - np.asarray(position, dtype=float)
            adjusted *= np.exp(-0.5 * np.sum(offset**2, axis=1) / variance)
        return self._normalise(adjusted, label)

    def _blocked(self, origin: np.ndarray):
        """Predicate the chain sampler uses to kill a step it may not take.

        Two things rule a step out. It may enter the protein, and it may leave
        the protein behind: the displaced strand of an R-loop runs along the
        surface it was unwound from rather than out into solvent, and a chain
        that wanders off has stopped being a model of it.

        Only the first was applied until now, so the ensemble repelled and never
        confined, and an unconstrained bubble spread far enough to reach most of
        the protospacer. Confinement is stated as a distance from the nearest
        occupied sphere and applies to every structure identically, which is
        what a repulsion-only ensemble could not do.

        Applying both during growth rather than as a filter afterwards is what
        makes a bridge across a protein feasible at all: the guide pulls toward
        the straight line between the pins, and where that line passes through
        the protein every chain grown without the constraint is discarded.
        """
        if self.occupancy is None:
            return None

        shell = None
        if self.surface_distance > 0:
            shell = OccupancyMap(
                centres=self.occupancy.centres,
                radii=self.occupancy.radii,
                probe_radius=float(self.surface_distance),
            )

        def predicate(points: np.ndarray) -> np.ndarray:
            moved = np.atleast_2d(points) + origin
            inside = self.occupancy.occluded(moved)
            if shell is None:
                return inside
            return inside | ~shell.occluded(moved)

        return predicate

    def _normalise(self, weights: np.ndarray, label: str) -> np.ndarray:
        total = float(np.sum(weights))
        if total <= 0:
            raise RuntimeError(
                f"no sampled strand conformation clears the protein for the {label}"
            )
        return weights / total

    @property
    def indices(self) -> list[int]:
        return sorted(self._clouds)

    def cloud(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Sampled positions and weights for one protospacer index."""
        if index not in self._clouds:
            raise KeyError(f"protospacer index {index} is outside the modelled strand")
        return self._clouds[index]

    def mean_position(self, index: int) -> np.ndarray:
        coords, weights = self.cloud(index)
        return np.average(coords, axis=0, weights=weights)

    def pam_distance(self, index: int) -> int:
        return pam_distance(index, self.spacer_length, self.numbering)
