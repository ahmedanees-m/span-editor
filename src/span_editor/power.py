"""What margin the corpus can resolve, given the number of source publications.

Entries cluster by laboratory, publication, deaminase and scaffold, so the
resampling unit is the publication and not the architecture. With around ten
independent clusters the interval on a paired difference in window-position
accuracy is wide, and this module exists to find out how wide before anything is
built on it.

Two pairing schemes are simulated and they bracket the answer.

Under comonotonic pairing the predictors are scored against one shared draw per
entry, so the stronger hits wherever the weaker does and paired differences are
zero or one, never minus one. That is the most favourable dependence a
comparison can have. It is also a degenerate null: at zero margin the predictors
agree everywhere, the paired difference is identically zero, and a rejection
rate of zero there is arithmetic rather than coverage.

Under independent pairing the predictors are drawn separately at the same rate.
Discordant pairs occur in both directions, so the difference has variance and the
interval can be calibrated against the nominal level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import wild_cluster_interval

__all__ = [
    "COMONOTONIC",
    "INDEPENDENT",
    "METHODS",
    "PERCENTILE",
    "WILD",
    "Design",
    "bootstrap_excludes_zero",
    "draw_trial",
    "minimum_detectable",
    "power_at",
    "power_curve",
    "rejects_null",
    "wild_cluster_rejects",
]

COMONOTONIC = "comonotonic"
INDEPENDENT = "independent"

PERCENTILE = "percentile"
WILD = "wild"
METHODS = (PERCENTILE, WILD)


@dataclass(frozen=True)
class Design:
    """The resampling design the simulation is run under."""

    n_clusters: int
    entries_per_cluster: float
    base_rate: float
    icc: float
    pairing: str = COMONOTONIC

    def __post_init__(self) -> None:
        if self.pairing not in (COMONOTONIC, INDEPENDENT):
            raise ValueError(f"unknown pairing scheme: {self.pairing}")
        if not 0.0 <= self.icc < 1.0:
            raise ValueError("icc must lie in [0, 1)")

    def cluster_sigma(self) -> float:
        """Logit-scale cluster spread matching the requested correlation."""
        if self.icc <= 0:
            return 0.0
        return float(np.sqrt(self.icc * (np.pi**2 / 3.0) / (1.0 - self.icc)))

    def as_dict(self) -> dict:
        return {
            "n_clusters": self.n_clusters,
            "entries_per_cluster": self.entries_per_cluster,
            "base_rate": self.base_rate,
            "icc": self.icc,
            "pairing": self.pairing,
        }


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _expit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def draw_trial(
    design: Design, delta: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """One synthetic corpus.

    Returns the per-cluster sums of the paired difference and the per-cluster
    entry counts, which is everything the bootstrap needs.
    """
    sizes = np.maximum(1, rng.poisson(design.entries_per_cluster, size=design.n_clusters))
    effects = rng.normal(0.0, design.cluster_sigma(), size=design.n_clusters)

    difference_sums = np.zeros(design.n_clusters)
    for cluster in range(design.n_clusters):
        n_entries = int(sizes[cluster])
        rate_weak = _expit(_logit(design.base_rate) + effects[cluster])
        rate_strong = _expit(_logit(min(design.base_rate + delta, 0.999)) + effects[cluster])
        if design.pairing == COMONOTONIC:
            shared = rng.random(n_entries)
            hits_strong = shared < rate_strong
            hits_weak = shared < rate_weak
        else:
            hits_strong = rng.random(n_entries) < rate_strong
            hits_weak = rng.random(n_entries) < rate_weak
        difference_sums[cluster] = float(np.sum(hits_strong.astype(int) - hits_weak.astype(int)))

    return difference_sums, sizes.astype(float)


def bootstrap_excludes_zero(
    difference_sums: np.ndarray,
    sizes: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> bool:
    """Cluster bootstrap on the pooled paired difference."""
    n_clusters = difference_sums.shape[0]
    index = rng.integers(0, n_clusters, size=(n_bootstrap, n_clusters))
    estimates = difference_sums[index].sum(axis=1) / sizes[index].sum(axis=1)
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return bool(lower > 0 or upper < 0)


def wild_cluster_rejects(
    difference_sums: np.ndarray,
    sizes: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    level: float = 0.95,
) -> bool:
    """Wild cluster bootstrap of the studentised mean, with the null imposed.

    The percentile bootstrap rejects too often when clusters are few, which is
    the regime this corpus sits in. Studentising each resample by its own
    cluster-robust standard error restores the size of the test. The rule here
    is the one the evaluation uses: the interval either excludes zero or does
    not.
    """
    lower, upper = wild_cluster_interval(
        difference_sums, sizes, n_bootstrap, level, seed=int(rng.integers(0, 2**32))
    )
    return bool(lower > 0 or upper < 0)


def rejects_null(
    difference_sums: np.ndarray,
    sizes: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    level: float = 0.95,
    method: str = PERCENTILE,
) -> bool:
    if method == PERCENTILE:
        return bootstrap_excludes_zero(difference_sums, sizes, n_bootstrap, rng, level)
    if method == WILD:
        return wild_cluster_rejects(difference_sums, sizes, n_bootstrap, rng, level)
    raise ValueError(f"unknown interval method: {method}")


def power_at(
    design: Design,
    delta: float,
    n_trials: int,
    n_bootstrap: int,
    rng: np.random.Generator,
    level: float = 0.95,
    method: str = PERCENTILE,
) -> float:
    rejections = 0
    for _ in range(n_trials):
        difference_sums, sizes = draw_trial(design, delta, rng)
        if rejects_null(difference_sums, sizes, n_bootstrap, rng, level, method):
            rejections += 1
    return rejections / n_trials


def power_curve(
    design: Design,
    deltas: np.ndarray,
    n_trials: int,
    n_bootstrap: int,
    seed: int,
    level: float = 0.95,
    method: str = PERCENTILE,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [
        {
            "delta": float(delta),
            "power": power_at(design, float(delta), n_trials, n_bootstrap, rng, level, method),
        }
        for delta in deltas
    ]


def minimum_detectable(curve: list[dict], target: float = 0.8) -> float | None:
    """Smallest margin on the grid whose power reaches the target, by interpolation."""
    ordered = sorted(curve, key=lambda point: point["delta"])
    for previous, current in zip(ordered[:-1], ordered[1:]):
        if previous["power"] < target <= current["power"]:
            span = current["power"] - previous["power"]
            if span <= 0:
                return current["delta"]
            fraction = (target - previous["power"]) / span
            return previous["delta"] + fraction * (current["delta"] - previous["delta"])
    return None
