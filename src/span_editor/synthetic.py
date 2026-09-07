"""Synthetic geometries for the implementation control.

The recovery test needs architectures whose true parameters are known. These
helpers build a stand-in editor: a ball of coarse spheres for the protein, an
anchor frame on its surface, and a displaced strand laid on an arc in front of
it with its two ends ordered. The geometry is not a model of any real editor and
is not used for anything that carries a claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Frame, OccupancyMap, StrandAnchor
from .io import CorpusEntry
from .model import ArchitectureContext
from .tether import EffectorSpec

__all__ = [
    "SyntheticGeometry",
    "protein_ball",
    "strand_arc",
    "synthetic_context",
    "synthetic_corpus",
]


@dataclass(frozen=True)
class SyntheticGeometry:
    """Knobs that distinguish one synthetic architecture from another."""

    body_radius: float = 26.0
    lattice_spacing: float = 6.0
    sphere_radius: float = 3.4
    standoff: float = 22.0
    curvature: float = 0.0
    end_to_end_fraction: float = 0.7
    anchor_tilt: float = 0.0


def protein_ball(
    body_radius: float = 26.0,
    lattice_spacing: float = 6.0,
    sphere_radius: float = 3.4,
) -> OccupancyMap:
    """A ball of coarse spheres whose surface passes through the anchor origin."""
    centre = np.array([-body_radius, 0.0, 0.0])
    steps = int(np.ceil(body_radius / lattice_spacing))
    axis = np.arange(-steps, steps + 1) * lattice_spacing
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = np.linalg.norm(grid, axis=1) <= body_radius
    return OccupancyMap(centres=grid[inside] + centre, radii=sphere_radius)


def strand_arc(
    spacer_length: int,
    rise: float,
    geometry: SyntheticGeometry,
) -> np.ndarray:
    """Nucleotide positions for the displaced strand, in the anchor frame."""
    index = np.arange(1, spacer_length + 1, dtype=float)
    centred = index - (spacer_length + 1) / 2.0
    span = geometry.end_to_end_fraction * rise
    coordinates = np.column_stack(
        [
            np.full(spacer_length, geometry.standoff) + geometry.curvature * centred**2,
            centred * span,
            np.full(spacer_length, 0.0),
        ]
    )
    if geometry.anchor_tilt:
        angle = np.deg2rad(geometry.anchor_tilt)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        coordinates = coordinates @ rotation.T
    return coordinates


def synthetic_context(
    entry: CorpusEntry,
    effector: EffectorSpec,
    geometry: SyntheticGeometry | None = None,
    rise: float = 6.3,
    ordered_indices: tuple[int, ...] | None = None,
    n_samples: int = 600,
    seed: int = 0,
) -> ArchitectureContext:
    """Assemble a synthetic architecture ready to be run through the model."""
    geometry = geometry or SyntheticGeometry()
    occupancy = protein_ball(
        geometry.body_radius, geometry.lattice_spacing, geometry.sphere_radius
    )
    coordinates = strand_arc(entry.spacer_length, rise, geometry)
    if ordered_indices is None:
        ordered_indices = (1, entry.spacer_length)
    anchors = [
        StrandAnchor(index=int(index), position=coordinates[int(index) - 1])
        for index in ordered_indices
    ]
    return ArchitectureContext(
        entry=entry,
        anchor=Frame(origin=np.zeros(3), basis=np.eye(3)),
        strand_anchors=anchors,
        effector=effector,
        indices=list(range(1, entry.spacer_length + 1)),
        occupancy=occupancy,
        rise=rise,
        n_samples=n_samples,
        seed=seed,
    )


def synthetic_corpus(
    effector: EffectorSpec,
    linker_lengths: tuple[int, ...] = (8, 16, 24, 32),
    standoffs: tuple[float, ...] = (20.0, 26.0),
    curvatures: tuple[float, ...] = (0.0, 0.12),
    spacer_length: int = 20,
    composition: str = "xten",
    n_samples: int = 600,
    seed: int = 0,
) -> list[ArchitectureContext]:
    """A grid of synthetic architectures spanning linker length and geometry."""
    contexts = []
    counter = 0
    for standoff in standoffs:
        for curvature in curvatures:
            for length in linker_lengths:
                counter += 1
                entry = CorpusEntry(
                    entry_id=f"syn-{counter:03d}",
                    source_key="synthetic",
                    editor=f"synthetic-{composition}-{length}",
                    ortholog="synthetic",
                    anchor_site="n_terminal",
                    linker_residues=length,
                    linker_composition=composition,
                    effector=effector.name,
                    label="tether_varied",
                    readout_class="amplicon_sequencing",
                    tier="A",
                    spacer_length=spacer_length,
                    split="fit",
                    panel="synthetic",
                )
                contexts.append(
                    synthetic_context(
                        entry,
                        effector,
                        geometry=SyntheticGeometry(standoff=standoff, curvature=curvature),
                        n_samples=n_samples,
                        seed=seed + counter,
                    )
                )
    return contexts
