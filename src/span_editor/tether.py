"""Tether sampling and effective-concentration fields.

A tether is a linker of known residue count and composition joining the editor
anchor to the effector. Sampling it gives the distribution of positions the
effector active site can occupy, which is the quantity the occupancy model
convolves against the substrate ensemble.

Persistence lengths are assigned per composition class from the polymer
literature and are never fitted; they enter through CompositionClass and are
loaded from the assigned-inputs configuration.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .geometry import Frame, OccupancyMap
from .polymer import sample_free_chain

__all__ = [
    "CompositionClass",
    "EffectorSpec",
    "LinkerSpec",
    "TetherClosure",
    "TetherField",
    "TetherSampler",
    "random_rotations",
]


@dataclass(frozen=True)
class CompositionClass:
    """Polymer properties of one linker chemistry.

    persistence is in angstrom. departure_cone is the half-angle in degrees over
    which the direction of the first link is spread; a rigid linker leaves the
    anchor in a direction set by backbone geometry that no structure resolves, so
    it is given a wide cone and flagged rather than pinned to a guess.

    persistence_range gives the span of published values. Where that span is
    wide, as it is for the ordered classes, predictions are reported across it
    rather than at the central value alone.
    """

    name: str
    persistence: float
    rise: float
    departure_cone: float = 180.0
    rigid: bool = False
    persistence_range: tuple[float, float] | None = None
    reference: str = ""

    @property
    def spread_in_persistence(self) -> float:
        """Ratio of the widest to the narrowest published persistence length."""
        if not self.persistence_range:
            return 1.0
        low, high = self.persistence_range
        return float(high / low) if low > 0 else float("inf")


@dataclass(frozen=True)
class LinkerSpec:
    """A linker as it is recorded in the corpus.

    n_residues is the contour length read from the construct map. Linker names
    are not unique in the literature, so the residue count is the identifier
    used everywhere downstream.
    """

    n_residues: int
    composition: str

    def key(self) -> str:
        return f"{self.composition}-{self.n_residues}"


@dataclass(frozen=True)
class EffectorSpec:
    """The tethered domain, described from its own fusion terminus.

    active_site_offset points from the fusion junction to the catalytic centre,
    in the effector's own frame. body_radius is the coarse radius used for
    excluded volume, taken as the radius of gyration of the folded body.

    flexible_tail counts the residues between the fusion junction and the start
    of the linker proper, whether they are disordered or resolved as an extended
    tail running away from the fold. Either way they behave as flexible chain, so
    they are added to the linker length rather than buried in a rigid offset.
    Their composition is taken to be that of the linker, which is an
    approximation.

    A domain inserted into the editor leaves by its far junction and returns to
    the protein. exit_offset points from the fusion junction to that far
    junction, in the same frame as active_site_offset, and exit_tail counts the
    flexible residues beyond it. Both are unset for a terminal fusion.
    """

    name: str
    active_site_offset: np.ndarray
    body_radius: float
    motif: str = ""
    flexible_tail: int = 0
    exit_offset: np.ndarray | None = None
    exit_tail: int = 0
    reference: str = ""


@dataclass(frozen=True)
class TetherClosure:
    """The second attachment of a domain inserted into the editor.

    point is the alpha carbon the return linker runs back to, in the anchor
    frame. n_links counts the residues available to reach it, being the return
    linker plus whatever flexible tail the domain carries past its far junction.
    """

    point: np.ndarray
    n_links: int
    composition: str


@dataclass(frozen=True)
class TetherField:
    """Weighted positions of the effector active site in the anchor frame."""

    positions: np.ndarray
    weights: np.ndarray
    linker: LinkerSpec
    effector: str
    directional_uncertainty: bool = False
    sampled: int = 0
    surviving: int = 0
    body_centres: np.ndarray | None = None
    body_radius: float = 0.0

    @property
    def effective_size(self) -> float:
        """Conformations the field is worth, after exclusion reweighting."""
        total = float(np.sum(self.weights))
        if total <= 0:
            return 0.0
        normalised = self.weights / total
        return float(1.0 / np.sum(normalised**2))

    @property
    def survival(self) -> float:
        return self.surviving / self.sampled if self.sampled else 0.0

    @property
    def centroid(self) -> np.ndarray:
        return np.average(self.positions, axis=0, weights=self.weights)

    @property
    def radius_of_gyration(self) -> float:
        offset = self.positions - self.centroid
        return float(np.sqrt(np.average(np.sum(offset**2, axis=1), weights=self.weights)))

    @property
    def body_centre(self) -> np.ndarray | None:
        """Mean position of the folded body over the field."""
        if self.body_centres is None:
            return None
        return np.average(self.body_centres, axis=0, weights=self.weights)

    @property
    def body_spread(self) -> float:
        """How far the folded body moves about that mean.

        A domain held at one end wanders over a volume many times its own size.
        One held at both ends barely moves, and only then is a single sphere at
        the mean a fair stand-in for the space it takes up.
        """
        if self.body_centres is None:
            return float("nan")
        offset = self.body_centres - self.body_centre
        return float(np.sqrt(np.average(np.sum(offset**2, axis=1), weights=self.weights)))


def random_rotations(n_rotations: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation matrices, shape (n_rotations, 3, 3)."""
    quaternion = rng.normal(size=(n_rotations, 4))
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    w, x, y, z = quaternion.T
    return np.stack(
        [
            np.stack([1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)], axis=1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)], axis=1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)], axis=1),
        ],
        axis=1,
    )


def _stable_seed(token: str) -> int:
    """A seed that does not move between processes, unlike the built-in hash."""
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "big") % (2**32)


def _cone_directions(
    axis: np.ndarray, half_angle_deg: float, n_directions: int, rng: np.random.Generator
) -> np.ndarray:
    """Directions drawn uniformly from a cone of the given half angle about axis."""
    unit = np.asarray(axis, dtype=float)
    unit = unit / np.linalg.norm(unit)
    limit = np.cos(np.deg2rad(min(half_angle_deg, 180.0)))
    cos_theta = rng.uniform(limit, 1.0, size=n_directions)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_directions)

    helper = np.array([0.0, 0.0, 1.0]) if abs(unit[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    first = np.cross(unit, helper)
    first /= np.linalg.norm(first)
    second = np.cross(unit, first)
    return (
        cos_theta[:, None] * unit
        + (sin_theta * np.cos(phi))[:, None] * first
        + (sin_theta * np.sin(phi))[:, None] * second
    )


@lru_cache(maxsize=64)
def _return_density(
    n_links: int, link_length: float, persistence: float, n_chains: int = 40000
) -> tuple[np.ndarray, np.ndarray]:
    """Relative likelihood that a free chain of n_links spans a given distance.

    An inserted domain is held at both ends, so a conformation is worth what the
    return linker is worth at the separation it leaves. The end-to-end
    distribution is taken from the same sampler the tether uses rather than from
    a Gaussian approximation, which is wrong in exactly the region that matters
    here: a short return linker spends most of its weight near full extension.

    Returned as bin edges and the density per unit volume in each bin, scaled to
    a peak of one. Separations past the contour length weigh nothing.
    """
    rng = np.random.default_rng(_stable_seed(f"return|{n_links}|{link_length}|{persistence}"))
    chain = sample_free_chain(
        n_links=n_links,
        link_length=link_length,
        persistence=persistence,
        n_chains=n_chains,
        rng=rng,
    )
    distance = np.linalg.norm(chain.endpoints, axis=1)
    edges = np.linspace(0.0, n_links * link_length, 65)
    counts, _ = np.histogram(distance, bins=edges)
    centres = 0.5 * (edges[:-1] + edges[1:])
    radial = counts / (distance.shape[0] * (edges[1] - edges[0]))
    volume = radial / (4.0 * np.pi * centres**2)
    peak = float(volume.max())
    return edges, volume / peak if peak > 0 else volume


def _return_weight(
    separation: np.ndarray, n_links: int, composition: CompositionClass
) -> np.ndarray:
    edges, density = _return_density(n_links, composition.rise, composition.persistence)
    slot = np.digitize(separation, edges) - 1
    inside = (slot >= 0) & (slot < density.shape[0])
    weight = np.zeros(separation.shape[0])
    weight[inside] = density[slot[inside]]
    return weight


class TetherSampler:
    """Builds and caches tether fields for a set of composition classes."""

    def __init__(
        self,
        compositions: dict[str, CompositionClass],
        n_chains: int = 20000,
        seed: int = 0,
        cache_dir: Path | str | None = None,
        target_surviving: int = 0,
        max_chains: int = 0,
    ) -> None:
        self.compositions = compositions
        self.n_chains = int(n_chains)
        self.seed = int(seed)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        # Survival through excluded volume varies by an order of magnitude across
        # architectures, so a fixed chain count buys very different precision for
        # a short linker than for a long one. When a target is set, batches are
        # drawn until that many conformations survive or the cap is reached.
        self.target_surviving = int(target_surviving)
        self.max_chains = int(max_chains or 20 * self.n_chains)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, token: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(token.encode()).hexdigest()[:16]
        return self.cache_dir / f"tether-{digest}.npz"

    def _draw_batch(
        self,
        n_chains: int,
        n_links: int,
        composition: CompositionClass,
        effector: EffectorSpec,
        occupancy: OccupancyMap | None,
        probe: OccupancyMap | None,
        closure: "TetherClosure | None",
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One batch of chains: the surviving active sites, their bodies and their weights."""
        axis = np.array([1.0, 0.0, 0.0])
        departures = _cone_directions(axis, composition.departure_cone, n_chains, rng)

        coords = np.empty((n_chains, n_links + 1, 3))
        block = 4000
        for start in range(0, n_chains, block):
            stop = min(start + block, n_chains)
            chunk = sample_free_chain(
                n_links=n_links,
                link_length=composition.rise,
                persistence=composition.persistence,
                n_chains=stop - start,
                rng=rng,
            )
            coords[start:stop] = chunk.coords

        coords = _reorient(coords, departures)
        terminus = coords[:, -1, :]
        rotations = random_rotations(n_chains, rng)
        offset = np.asarray(effector.active_site_offset, dtype=float)
        active_site = terminus + np.einsum("nij,j->ni", rotations, offset)
        body_centre = terminus + 0.5 * np.einsum("nij,j->ni", rotations, offset)
        weights = np.ones(n_chains)

        if closure is not None:
            exit_offset = np.asarray(effector.exit_offset, dtype=float)
            exit_point = terminus + np.einsum("nij,j->ni", rotations, exit_offset)
            separation = np.linalg.norm(exit_point - np.asarray(closure.point, dtype=float), axis=1)
            weights = _return_weight(
                separation, closure.n_links, self.compositions[closure.composition]
            )

        keep = weights > 0
        if occupancy is not None:
            clashing = occupancy.chain_clashes(coords, skip_first=1)
            if probe is not None:
                clashing |= probe.occluded(body_centre)
            keep &= ~clashing
        return active_site[keep], body_centre[keep], weights[keep]

    # The tether grows into open solvent away from the anchor rather than
    # threading a channel, so rejecting whole chains is affordable here and the
    # per-step constraint the substrate ensemble needs is not used.

    def sample(
        self,
        linker: LinkerSpec,
        effector: EffectorSpec,
        anchor: Frame | None = None,
        occupancy: OccupancyMap | None = None,
        structure_id: str = "none",
        closure: TetherClosure | None = None,
    ) -> TetherField:
        """Sample active-site positions for one linker and effector.

        Chains grow from the anchor along a departure direction drawn from the
        composition cone. The effector is placed at the far end with a uniformly
        sampled orientation, and conformations whose chain or effector body
        intersects the protein are dropped.

        A closure makes the domain an insertion rather than a terminal fusion:
        each conformation is then weighted by what its return linker is worth at
        the separation it has to span, and conformations it cannot span at all
        are dropped. The return path is not itself tested against the protein,
        so the constraint is a bound on reach rather than a full treatment.
        """
        composition = self.compositions.get(linker.composition)
        if composition is None:
            raise KeyError(f"no persistence length assigned for {linker.composition!r}")
        if linker.n_residues < 1:
            raise ValueError("a linker must have at least one residue")
        if closure is not None:
            if closure.composition not in self.compositions:
                raise KeyError(f"no persistence length assigned for {closure.composition!r}")
            if effector.exit_offset is None:
                raise ValueError(f"{effector.name} carries no exit offset, so it cannot be inserted")

        n_links = linker.n_residues + max(0, effector.flexible_tail)
        token = "|".join(
            [
                linker.key(),
                str(n_links),
                effector.name,
                structure_id,
                str(self.n_chains),
                str(self.target_surviving),
                str(self.max_chains),
                str(self.seed),
                f"{composition.persistence:.4f}",
                f"{composition.rise:.4f}",
                f"{composition.departure_cone:.2f}",
                "none"
                if closure is None
                else "{}|{}|{}".format(
                    closure.composition,
                    closure.n_links,
                    ",".join(f"{value:.3f}" for value in np.asarray(closure.point, dtype=float)),
                ),
            ]
        )
        cached = self._cache_path(token)
        if cached is not None and cached.exists():
            stored = np.load(cached)
            return TetherField(
                positions=stored["positions"],
                weights=stored["weights"],
                linker=linker,
                effector=effector.name,
                directional_uncertainty=bool(stored["directional_uncertainty"]),
                sampled=int(stored["sampled"]),
                surviving=int(stored["surviving"]),
                body_centres=stored["body_centres"] if "body_centres" in stored else None,
                body_radius=float(stored["body_radius"]) if "body_radius" in stored else 0.0,
            )

        rng = np.random.default_rng(_stable_seed(token))
        probe = (
            OccupancyMap(occupancy.centres, occupancy.radii, probe_radius=effector.body_radius)
            if occupancy is not None
            else None
        )

        kept: list[np.ndarray] = []
        kept_bodies: list[np.ndarray] = []
        kept_weights: list[np.ndarray] = []
        sampled = 0
        surviving = 0
        target = self.target_surviving or 1
        while True:
            positions, bodies, weights = self._draw_batch(
                self.n_chains, n_links, composition, effector, occupancy, probe, closure, rng
            )
            sampled += self.n_chains
            surviving += positions.shape[0]
            if positions.size:
                kept.append(positions)
                kept_bodies.append(bodies)
                kept_weights.append(weights)
            if surviving >= target or sampled >= self.max_chains:
                break

        if not kept:
            raise RuntimeError(
                f"every conformation of {linker.key()} clashes with the editor, "
                f"over {sampled} chains"
            )

        active_site = np.concatenate(kept)
        bodies = np.concatenate(kept_bodies)
        weight = np.concatenate(kept_weights)
        field = TetherField(
            positions=active_site,
            weights=weight / float(np.sum(weight)),
            linker=linker,
            effector=effector.name,
            directional_uncertainty=composition.rigid,
            sampled=sampled,
            surviving=surviving,
            body_centres=bodies,
            body_radius=effector.body_radius,
        )
        if cached is not None:
            np.savez_compressed(
                cached,
                positions=field.positions,
                weights=field.weights,
                directional_uncertainty=field.directional_uncertainty,
                sampled=field.sampled,
                surviving=field.surviving,
                body_centres=field.body_centres,
                body_radius=field.body_radius,
            )
        return field


def _reorient(coords: np.ndarray, departures: np.ndarray) -> np.ndarray:
    """Rotate each chain so that its first link points along the given direction."""
    first = coords[:, 1, :] - coords[:, 0, :]
    first = first / np.linalg.norm(first, axis=1, keepdims=True)
    rotation = _rotation_between(first, departures)
    return np.einsum("nij,nkj->nki", rotation, coords)


def _perpendicular(vectors: np.ndarray) -> np.ndarray:
    """One unit vector perpendicular to each row."""
    helper = np.tile(np.array([0.0, 0.0, 1.0]), (vectors.shape[0], 1))
    parallel = np.abs(vectors[:, 2]) > 0.9
    helper[parallel] = np.array([1.0, 0.0, 0.0])
    perpendicular = np.cross(vectors, helper)
    return perpendicular / np.linalg.norm(perpendicular, axis=1, keepdims=True)


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation matrices taking each row of source onto the matching row of target."""
    axis = np.cross(source, target)
    sine = np.linalg.norm(axis, axis=1)
    cosine = np.sum(source * target, axis=1)
    degenerate = sine <= 1e-9

    # A pair that is already parallel needs no rotation; an antiparallel pair is
    # turned by pi about any perpendicular axis.
    axis_unit = np.where(
        degenerate[:, None], _perpendicular(source), axis / np.maximum(sine, 1e-12)[:, None]
    )
    angle = np.where(degenerate, np.where(cosine < 0, np.pi, 0.0), np.arctan2(sine, cosine))

    cross = np.zeros((source.shape[0], 3, 3))
    cross[:, 0, 1] = -axis_unit[:, 2]
    cross[:, 0, 2] = axis_unit[:, 1]
    cross[:, 1, 0] = axis_unit[:, 2]
    cross[:, 1, 2] = -axis_unit[:, 0]
    cross[:, 2, 0] = -axis_unit[:, 1]
    cross[:, 2, 1] = axis_unit[:, 0]

    identity = np.tile(np.eye(3), (source.shape[0], 1, 1))
    return (
        identity
        + np.sin(angle)[:, None, None] * cross
        + (1.0 - np.cos(angle))[:, None, None] * (cross @ cross)
    )
