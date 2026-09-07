"""Capture probability, the motif term and the occupancy-to-editing link.

The forward model convolves the tether field against the substrate ensemble
under a capture criterion, multiplies by the fixed intrinsic motif preference of
the effector, and passes the result through a single global saturating link.

Only three quantities are fitted: the capture radius, the flexibility of the
displaced strand, and the steepness of the link. The link is global by
construction, since most held-out datasets contain no architecture from which a
per-dataset activity scale could be estimated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import SubstrateEnsemble
from .tether import TetherField

__all__ = [
    "ModelParameters",
    "MotifPreference",
    "Profile",
    "capture_probability",
    "capture_probability_curve",
    "link_alpha_for_index",
    "predict_profile",
    "saturating_link",
    "saturation_index",
]


@dataclass(frozen=True)
class ModelParameters:
    """The fitted parameter set. Three entries, with a ceiling of four."""

    capture_radius: float
    ssdna_persistence: float
    link_alpha: float

    def as_dict(self) -> dict[str, float]:
        return {
            "capture_radius": self.capture_radius,
            "ssdna_persistence": self.ssdna_persistence,
            "link_alpha": self.link_alpha,
        }

    @classmethod
    def from_dict(cls, values: dict[str, float]) -> "ModelParameters":
        return cls(
            capture_radius=float(values["capture_radius"]),
            ssdna_persistence=float(values["ssdna_persistence"]),
            link_alpha=float(values["link_alpha"]),
        )


@dataclass(frozen=True)
class MotifPreference:
    """Intrinsic sequence preference of the effector, taken from characterisation.

    base_weights maps the base immediately 5' of the edited position to a
    relative rate. Corpus entries that aggregate over thousands of targets are
    evaluated with sequence set to None, which leaves the term flat.
    """

    name: str
    base_weights: dict[str, float] = field(default_factory=dict)
    reference: str = ""

    def per_index(self, indices: np.ndarray, sequence: str | None = None) -> np.ndarray:
        indices = np.asarray(indices, dtype=int)
        if sequence is None or not self.base_weights:
            return np.ones(indices.shape[0])
        weights = np.empty(indices.shape[0])
        for slot, index in enumerate(indices):
            preceding = sequence[index - 2] if 2 <= index <= len(sequence) else "N"
            weights[slot] = self.base_weights.get(preceding.upper(), 1.0)
        return weights


@dataclass(frozen=True)
class Profile:
    """A per-position activity profile over protospacer indices."""

    indices: np.ndarray
    values: np.ndarray
    capture: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.indices.shape != self.values.shape:
            raise ValueError("indices and values must have the same shape")
        if self.capture is not None and self.capture.shape != self.values.shape:
            raise ValueError("capture and values must have the same shape")

    @property
    def ranking(self) -> np.ndarray:
        """Values the mode and the degeneracy test are read from.

        The link is 1 - exp(-alpha * capture), which is strictly increasing, so
        the mode is the same whether it is taken before or after the link. Taken
        after, it underflows: above an exponent of about 37 every position
        evaluates to exactly 1.0 and the profile reads as tied. Capture
        probability is used instead where it is available. Level sets, and so
        the window, remain link-dependent and are taken from the linked values.
        """
        return self.values if self.capture is None else self.capture

    def normalised(self) -> "Profile":
        peak = float(np.max(self.values))
        if peak <= 0:
            return self
        return Profile(
            indices=self.indices, values=self.values / peak, capture=self.capture
        )

    @property
    def mode(self) -> int:
        ranking = self.ranking
        peak = float(np.max(ranking))
        tied = self.indices[ranking >= peak - 1e-12]
        return int(np.rint(np.mean(tied)))

    @property
    def centre_of_mass(self) -> float:
        total = float(np.sum(self.values))
        if total <= 0:
            return float("nan")
        return float(np.sum(self.indices * self.values) / total)

    @property
    def degenerate(self) -> bool:
        """Whether more than half the positions tie at the peak.

        The mode of such a profile is the midpoint of the tie rather than a
        predicted position, so these entries are excluded from reported rates.
        """
        ranking = self.ranking
        peak = float(np.max(ranking))
        if peak <= 0:
            return True
        return int(np.sum(ranking >= peak - 1e-12)) > 0.5 * ranking.shape[0]

    def window(self, threshold: float = 0.5) -> tuple[int, int]:
        """First and last index of the run around the mode above the threshold.

        This is what a published editing window is: the stretch of positions
        carrying activity worth reporting. The threshold makes the convention
        explicit, since papers rarely state one.
        """
        peak = float(np.max(self.values))
        if peak <= 0:
            return (0, 0)
        above = self.values >= threshold * peak
        centre = int(np.argmax(self.values))
        left = centre
        while left - 1 >= 0 and above[left - 1]:
            left -= 1
        right = centre
        while right + 1 < above.shape[0] and above[right + 1]:
            right += 1
        return (int(self.indices[left]), int(self.indices[right]))

    def width(self, threshold: float = 0.5) -> int:
        """Length in nucleotides of the run around the mode above the threshold."""
        first, last = self.window(threshold)
        return 0 if last == first == 0 else int(last - first + 1)


def capture_probability(
    tether: TetherField,
    substrate: SubstrateEnsemble,
    capture_radius: float,
    indices: list[int] | None = None,
    block: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Weighted fraction of tether and substrate pairs closer than the capture radius.

    Returns the protospacer indices and the matching capture probabilities. The
    pairwise sum is taken in blocks over the substrate cloud so that the memory
    footprint stays independent of the sample count.
    """
    positions = tether.positions
    tether_weights = tether.weights
    wanted = substrate.indices if indices is None else list(indices)
    probability = np.zeros(len(wanted))
    radius_squared = float(capture_radius) ** 2

    for slot, index in enumerate(wanted):
        cloud, cloud_weights = substrate.cloud(index)
        total = 0.0
        for start in range(0, cloud.shape[0], block):
            stop = min(start + block, cloud.shape[0])
            delta = positions[None, :, :] - cloud[start:stop, None, :]
            within = np.sum(delta**2, axis=2) <= radius_squared
            total += float(
                np.sum(cloud_weights[start:stop, None] * tether_weights[None, :] * within)
            )
        probability[slot] = total

    return np.asarray(wanted, dtype=int), probability


def capture_probability_curve(
    tether: TetherField,
    substrate: SubstrateEnsemble,
    radii: np.ndarray,
    indices: list[int] | None = None,
    block: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Capture probability at every radius on a grid, from one pass over the pairs.

    Fitting sweeps the capture radius, and recomputing the pairwise distances at
    each value would dominate the cost. Histogramming the distances once and
    taking a cumulative sum gives the whole curve instead, so only the
    flexibility of the displaced strand forces the ensemble to be rebuilt.

    Returns the protospacer indices and an array of shape (n_indices, n_radii).
    """
    grid = np.sort(np.asarray(radii, dtype=float))
    # Squaring is monotone on distances, so binning the squared separation gives
    # the same cumulative counts and skips a square root over every pair.
    edges = np.concatenate([[0.0], grid**2])
    wanted = substrate.indices if indices is None else list(indices)
    curve = np.zeros((len(wanted), grid.shape[0]))

    positions = tether.positions
    tether_weights = tether.weights
    for slot, index in enumerate(wanted):
        cloud, cloud_weights = substrate.cloud(index)
        binned = np.zeros(grid.shape[0])
        for start in range(0, cloud.shape[0], block):
            stop = min(start + block, cloud.shape[0])
            delta = positions[None, :, :] - cloud[start:stop, None, :]
            squared = np.einsum("ijk,ijk->ij", delta, delta)
            pair_weight = cloud_weights[start:stop, None] * tether_weights[None, :]
            counts, _ = np.histogram(squared.ravel(), bins=edges, weights=pair_weight.ravel())
            binned += counts
        curve[slot] = np.cumsum(binned)

    return np.asarray(wanted, dtype=int), curve


def saturating_link(occupancy: np.ndarray, alpha: float) -> np.ndarray:
    """Map capture probability to measured editing fraction.

    Editing accumulates over the residence time of the editor on the target, so
    the measured fraction saturates where capture is high and stays linear where
    it is low. A single global alpha carries that transform everywhere.
    """
    return 1.0 - np.exp(-float(alpha) * np.asarray(occupancy, dtype=float))


def saturation_index(link_alpha: float, peak_occupancy: float) -> float:
    """Link steepness expressed on a scale that does not depend on the sampling.

    Capture probability is a small number whose size depends on how densely the
    tether and the substrate were sampled, so the raw steepness is not
    comparable between runs. Multiplying it by the largest capture probability
    in the training subset gives the argument of the exponential at the peak,
    which is the quantity that decides whether the link is doing anything: well
    below one the link is close to linear and a normalised profile carries
    almost no information about it.

    The peak is always taken at a fixed reference capture radius, recorded as
    link.reference_radius in configs/params.yaml, and never at whichever radius
    the search happens to be visiting. An adaptive reference would make the
    index depend on the grid rather than on the model, so the same fit would
    report different indices under different searches.
    """
    return float(link_alpha) * float(peak_occupancy)


def link_alpha_for_index(index: float, peak_occupancy: float) -> float:
    """Inverse of saturation_index, used to place the fitting grid."""
    if peak_occupancy <= 0:
        raise ValueError("peak capture probability must be positive")
    return float(index) / float(peak_occupancy)


def predict_profile(
    tether: TetherField,
    substrate: SubstrateEnsemble,
    parameters: ModelParameters,
    motif: MotifPreference | None = None,
    sequence: str | None = None,
    indices: list[int] | None = None,
) -> Profile:
    """Full forward model for one architecture."""
    positions, occupancy = capture_probability(
        tether, substrate, parameters.capture_radius, indices=indices
    )
    if motif is not None:
        occupancy = occupancy * motif.per_index(positions, sequence)
    return Profile(
        indices=positions,
        values=saturating_link(occupancy, parameters.link_alpha),
        capture=np.asarray(occupancy, dtype=float),
    )
