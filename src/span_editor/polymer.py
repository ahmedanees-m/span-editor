"""Discrete worm-like chain samplers.

The tether and the displaced non-target strand are both treated as
semi-flexible chains of fixed link length with a bending stiffness set by a
persistence length. Two sampling problems arise: a free chain growing from a
fixed origin, and a bridge whose far end is pinned by the structure. Both are
handled by sequential sampling over links; the bridge adds a Gaussian guide
term and importance weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

__all__ = [
    "ChainSample",
    "bending_concentration",
    "log_von_mises_fisher",
    "mean_square_end_to_end",
    "sample_bridge",
    "sample_free_chain",
    "sample_von_mises_fisher",
]


@dataclass(frozen=True)
class ChainSample:
    """A weighted set of chain conformations.

    coords has shape (n_chains, n_links + 1, 3) and holds every joint position
    including the origin. weights sum to one; a free chain returns uniform
    weights.
    """

    coords: np.ndarray
    weights: np.ndarray

    @property
    def endpoints(self) -> np.ndarray:
        return self.coords[:, -1, :]

    @property
    def effective_size(self) -> float:
        return float(1.0 / np.sum(self.weights**2))


def _mean_cosine(concentration: float) -> float:
    """Mean cosine of the bending angle under a von Mises-Fisher kernel."""
    if concentration < 1e-4:
        return concentration / 3.0
    if concentration > 1e4:
        return 1.0 - 1.0 / concentration
    return 1.0 / np.tanh(concentration) - 1.0 / concentration


@lru_cache(maxsize=4096)
def bending_concentration(link_length: float, persistence: float) -> float:
    """Kernel concentration that reproduces a given persistence length.

    Discretising a worm-like chain into links of length b makes the tangent
    correlation decay as the mean bending cosine raised to the number of links.
    Setting the kernel concentration to the ratio of persistence length to link
    length would leave that decay too fast, and the chain measurably shorter
    than the polymer it is meant to represent. The concentration is solved for
    instead, so that one link decorrelates by exp(-b / lp).
    """
    if persistence <= 0 or link_length <= 0:
        return 0.0
    target = float(np.exp(-link_length / persistence))
    if target >= 1.0 - 1e-12:
        return 1e6
    if target <= 1e-12:
        return 0.0

    low, high = 1e-8, 1e6
    for _ in range(200):
        middle = np.sqrt(low * high)
        if _mean_cosine(middle) < target:
            low = middle
        else:
            high = middle
    return float(np.sqrt(low * high))


def mean_square_end_to_end(n_links: int, link_length: float, persistence: float) -> float:
    """Mean square end-to-end distance of the discretised chain.

    The sum is taken over the discrete links rather than from the continuous
    Kratky-Porod expression, so that it agrees with what the sampler produces.
    Both limits are the expected ones: a stiff chain reaches its contour length
    and a flexible one reaches the ideal-chain value.
    """
    if n_links <= 0:
        return 0.0
    cosine = _mean_cosine(bending_concentration(link_length, persistence))
    separation = np.arange(1, n_links)
    tail = float(np.sum((n_links - separation) * cosine**separation))
    return float(link_length**2 * (n_links + 2.0 * tail))


def _tangent_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane normal to each row of axis."""
    helper = np.tile(np.array([0.0, 0.0, 1.0]), (axis.shape[0], 1))
    parallel = np.abs(axis[:, 2]) > 0.9
    helper[parallel] = np.array([1.0, 0.0, 0.0])
    first = np.cross(axis, helper)
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second = np.cross(axis, first)
    return first, second


def sample_von_mises_fisher(
    mean: np.ndarray, concentration: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw one unit vector per row from a von Mises-Fisher law on the sphere.

    mean has shape (n, 3) and need not be normalised; concentration has shape
    (n,). Rows with a concentration below 1e-8 are drawn uniformly.
    """
    mean = np.asarray(mean, dtype=float)
    norms = np.linalg.norm(mean, axis=1, keepdims=True)
    axis = np.where(norms > 0, mean / np.maximum(norms, 1e-12), np.array([0.0, 0.0, 1.0]))
    kappa = np.asarray(concentration, dtype=float)
    n_rows = axis.shape[0]

    uniform = rng.random(n_rows)
    isotropic = kappa < 1e-8
    safe_kappa = np.where(isotropic, 1.0, kappa)
    cos_theta = 1.0 + np.log(
        uniform + (1.0 - uniform) * np.exp(-2.0 * safe_kappa)
    ) / safe_kappa
    cos_theta = np.where(isotropic, 2.0 * uniform - 1.0, cos_theta)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_rows)
    first, second = _tangent_basis(axis)
    return (
        cos_theta[:, None] * axis
        + (sin_theta * np.cos(phi))[:, None] * first
        + (sin_theta * np.sin(phi))[:, None] * second
    )


def log_von_mises_fisher(
    x: np.ndarray, mean: np.ndarray, concentration: np.ndarray
) -> np.ndarray:
    """Log density of the von Mises-Fisher law on the unit sphere in three dimensions."""
    kappa = np.maximum(np.asarray(concentration, dtype=float), 1e-8)
    norms = np.linalg.norm(mean, axis=1, keepdims=True)
    axis = mean / np.maximum(norms, 1e-12)
    dot = np.sum(x * axis, axis=1)
    # log(kappa / (4 pi sinh kappa)), written so that it stays finite at large kappa.
    log_norm = np.log(kappa) - np.log(2.0 * np.pi) - np.log1p(-np.exp(-2.0 * kappa)) - kappa
    return log_norm + kappa * dot


def _normalise_log_weights(log_weight: np.ndarray) -> np.ndarray:
    shifted = log_weight - np.max(log_weight)
    weights = np.exp(shifted)
    total = np.sum(weights)
    if total <= 0 or not np.isfinite(total):
        return np.full(log_weight.shape[0], 1.0 / log_weight.shape[0])
    return weights / total


def _systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n_rows = weights.shape[0]
    positions = (rng.random() + np.arange(n_rows)) / n_rows
    return np.searchsorted(np.cumsum(weights), positions).clip(0, n_rows - 1)


def _initial_directions(
    n_chains: int, first_direction: np.ndarray | None, rng: np.random.Generator
) -> np.ndarray:
    if first_direction is None:
        directions = rng.normal(size=(n_chains, 3))
        return directions / np.linalg.norm(directions, axis=1, keepdims=True)
    unit = np.asarray(first_direction, dtype=float)
    unit = unit / np.linalg.norm(unit)
    return np.tile(unit, (n_chains, 1))


def sample_free_chain(
    n_links: int,
    link_length: float,
    persistence: float,
    n_chains: int,
    rng: np.random.Generator,
    first_direction: np.ndarray | None = None,
    blocked=None,
    resample_threshold: float = 0.5,
) -> ChainSample:
    """Sample chains growing from the origin with no constraint on the far end.

    blocked, when given, takes an array of points and returns which of them lie
    inside excluded volume. Chains that enter it are killed at the step where
    they do and replaced by resampling from those still alive, rather than being
    grown to full length and discarded. Rejecting at the end wastes almost
    everything once a chain has to thread a narrow channel.
    """
    if n_links < 1:
        raise ValueError("n_links must be at least 1")
    kappa = np.full(n_chains, bending_concentration(link_length, persistence))
    coords = np.zeros((n_chains, n_links + 1, 3))
    direction = _initial_directions(n_chains, first_direction, rng)
    coords[:, 1, :] = link_length * direction
    log_weight = np.zeros(n_chains)

    if blocked is not None:
        log_weight[blocked(coords[:, 1, :])] = -np.inf
        log_weight, coords, direction, extinct = _prune(
            log_weight, coords, direction, rng, resample_threshold, 1
        )
        if extinct:
            return ChainSample(coords=coords, weights=np.zeros(n_chains))

    for joint in range(1, n_links):
        direction = sample_von_mises_fisher(direction, kappa, rng)
        coords[:, joint + 1, :] = coords[:, joint, :] + link_length * direction
        if blocked is None:
            continue
        log_weight[blocked(coords[:, joint + 1, :])] = -np.inf
        log_weight, coords, direction, extinct = _prune(
            log_weight, coords, direction, rng, resample_threshold, joint + 1
        )
        if extinct:
            return ChainSample(coords=coords, weights=np.zeros(n_chains))

    return ChainSample(coords=coords, weights=_normalise_log_weights(log_weight))


def _prune(log_weight, coords, direction, rng, threshold, grown):
    """Resample away chains that have entered excluded volume.

    Returns the surviving state and whether every chain is dead. Only the joints
    already grown are carried, so the resampled copies share a history that is
    valid up to this step.
    """
    alive = np.isfinite(log_weight)
    if not np.any(alive):
        return log_weight, coords, direction, True

    weights = _normalise_log_weights(log_weight)
    if 1.0 / np.sum(weights**2) >= threshold * log_weight.shape[0]:
        return log_weight, coords, direction, False

    index = _systematic_resample(weights, rng)
    coords = coords.copy()
    coords[:, : grown + 1, :] = coords[index][:, : grown + 1, :]
    return np.zeros_like(log_weight), coords, direction[index], False


def sample_bridge(
    n_links: int,
    link_length: float,
    persistence: float,
    end_offset: np.ndarray,
    n_chains: int,
    rng: np.random.Generator,
    first_direction: np.ndarray | None = None,
    closure_tolerance: float = 2.0,
    resample_threshold: float = 0.5,
    blocked=None,
) -> ChainSample:
    """Sample chains from the origin whose far end lands near end_offset.

    Links are proposed from the product of the bending kernel and a tilt toward
    the remaining displacement, then reweighted by a Gaussian guide built from
    the Kratky-Porod end-to-end variance of the links still to be placed. The
    last link closes against a Gaussian of width closure_tolerance rather than a
    delta, so a bridge that cannot close exactly is down-weighted instead of
    rejected outright.

    blocked, when given, kills a chain at the step it enters excluded volume.
    That matters more here than for a free chain: the guide pulls toward the
    straight line between the two pins, and when the protein sits across that
    line every chain grown without the constraint is rejected at the end.
    """
    if n_links < 1:
        raise ValueError("n_links must be at least 1")
    target = np.asarray(end_offset, dtype=float)
    if link_length * n_links < np.linalg.norm(target):
        raise ValueError("contour length is shorter than the end-to-end distance")

    kappa = bending_concentration(link_length, persistence)
    kappa_column = np.full(n_chains, kappa)
    coords = np.zeros((n_chains, n_links + 1, 3))
    direction = _initial_directions(n_chains, first_direction, rng)
    log_weight = np.zeros(n_chains)
    position = np.zeros((n_chains, 3))

    def guide_variance(remaining: int) -> float:
        if remaining <= 0:
            return closure_tolerance**2
        return max(mean_square_end_to_end(remaining, link_length, persistence) / 3.0, 1e-6)

    log_guide = -0.5 * np.sum((target - position) ** 2, axis=1) / guide_variance(n_links)

    for link in range(n_links):
        remaining = n_links - (link + 1)
        variance_next = guide_variance(remaining)
        residual = target - position
        distance = np.linalg.norm(residual, axis=1, keepdims=True)
        pull = np.where(distance > 1e-12, residual / np.maximum(distance, 1e-12), 0.0)
        tilt = (link_length * distance / variance_next) * pull

        if link == 0 and first_direction is not None:
            step = direction
        else:
            proposal_mean = kappa * direction + tilt
            proposal_kappa = np.linalg.norm(proposal_mean, axis=1)
            step = sample_von_mises_fisher(proposal_mean, proposal_kappa, rng)
            log_weight += log_von_mises_fisher(step, direction, kappa_column)
            log_weight -= log_von_mises_fisher(step, proposal_mean, proposal_kappa)

        direction = step
        position = position + link_length * direction
        coords[:, link + 1, :] = position

        log_guide_next = -0.5 * np.sum((target - position) ** 2, axis=1) / variance_next
        log_weight += log_guide_next - log_guide
        log_guide = log_guide_next

        if blocked is not None:
            log_weight[blocked(position)] = -np.inf
            if not np.any(np.isfinite(log_weight)):
                return ChainSample(coords=coords, weights=np.zeros(n_chains))

        weights = _normalise_log_weights(log_weight)
        degenerate = 1.0 / np.sum(weights**2) < resample_threshold * n_chains
        if degenerate and link < n_links - 1:
            index = _systematic_resample(weights, rng)
            coords = coords[index]
            position = position[index]
            direction = direction[index]
            log_guide = log_guide[index]
            log_weight = np.zeros(n_chains)

    return ChainSample(coords=coords, weights=_normalise_log_weights(log_weight))
