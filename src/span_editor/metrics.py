"""Scoring, margins and the cluster bootstrap.

Entries cluster by laboratory, publication, deaminase and scaffold, so every
interval is taken by resampling source publications rather than architectures.
Thresholds are margins over the marginal baseline, not absolute hit rates: most
SpCas9 editors peak in a narrow band of positions, so a high absolute hit rate
can be had for nothing.

Cas12a is the exception. The marginal baseline cannot relocate a window, so an
absolute criterion there is self-justifying, and predictions are reported in
distance from the PAM as well as in protospacer index.

Two interval constructions are available and they do not agree at the cluster
counts this corpus supplies. The percentile bootstrap resamples whole clusters
and takes quantiles of the estimate; it rejects a true null about twice as often
as it should at nine clusters. The wild cluster bootstrap gives Rademacher signs
to the cluster residuals and studentises each resample by its own cluster-robust
standard error; it holds its size from six clusters upward, at the cost of a
wider interval. The wild construction is the default.

Imposing the null and inverting the resulting test, which is the usual advice
for few clusters, does not work for this estimand. See saturation_statistic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import pam_distance
from .occupancy import Profile

__all__ = [
    "PERCENTILE",
    "WILD",
    "BootstrapInterval",
    "MarginResult",
    "cluster_bootstrap",
    "cluster_totals",
    "mode_hit",
    "mode_offset",
    "paired_margin",
    "percentile_cluster_interval",
    "rademacher_signs",
    "saturation_statistic",
    "report_in_both_conventions",
    "signed_offsets",
    "wild_cluster_interval",
]

PERCENTILE = "percentile"
WILD = "wild"


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    n_clusters: int
    level: float = 0.95
    method: str = WILD

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0.0 or self.upper < 0.0

    def as_dict(self) -> dict:
        return {
            "estimate": self.estimate,
            "lower": self.lower,
            "upper": self.upper,
            "n_clusters": self.n_clusters,
            "level": self.level,
            "method": self.method,
            "excludes_zero": self.excludes_zero,
        }


@dataclass(frozen=True)
class MarginResult:
    name: str
    reference: str
    interval: BootstrapInterval
    n_entries: int

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "reference": self.reference,
            "n_entries": self.n_entries,
            **self.interval.as_dict(),
        }


def mode_offset(predicted: Profile, observed: Profile) -> int:
    """Signed difference between predicted and observed window mode.

    The primary residual. It survives any monotone link, so it is available on
    every entry, including those measured by functional selection where amplitude
    means nothing.
    """
    return int(predicted.mode - observed.mode)


def mode_hit(predicted: Profile, observed: Profile, tolerance: int = 1) -> bool:
    return abs(mode_offset(predicted, observed)) <= tolerance


def signed_offsets(pairs: list[tuple[Profile, Profile]]) -> np.ndarray:
    return np.asarray([mode_offset(predicted, observed) for predicted, observed in pairs])


def cluster_totals(
    values: np.ndarray, clusters: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-cluster sums and sizes, which is all either bootstrap needs."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    labels = np.unique(clusters)
    if labels.size == 0:
        raise ValueError("no clusters to resample")
    sums = np.asarray([values[clusters == label].sum() for label in labels])
    sizes = np.asarray([np.count_nonzero(clusters == label) for label in labels], dtype=float)
    return labels, sums, sizes


def percentile_cluster_interval(
    sums: np.ndarray,
    sizes: np.ndarray,
    n_bootstrap: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Quantiles of the estimate under resampling of whole clusters."""
    rng = np.random.default_rng(seed)
    n_clusters = sums.shape[0]
    index = rng.integers(0, n_clusters, size=(n_bootstrap, n_clusters))
    estimates = sums[index].sum(axis=1) / sizes[index].sum(axis=1)
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(estimates, [tail, 1.0 - tail])
    return float(lower), float(upper)


def rademacher_signs(
    n_bootstrap: int, n_clusters: int, rng: np.random.Generator
) -> np.ndarray:
    """Sign matrix for the wild bootstrap.

    There are only two to the power of the cluster count distinct sign vectors,
    so at six clusters the smallest attainable p-value is about 0.016. That is
    usable at the five per cent level and would not be at one per cent.
    """
    return rng.choice(np.array([-1.0, 1.0]), size=(n_bootstrap, n_clusters))


def saturation_statistic(sizes: np.ndarray) -> float:
    """The value the studentised statistic approaches for a distant null.

    Studentising a cluster mean puts the candidate value in both the numerator
    and, through the residuals, the standard error. Both grow at the same rate,
    so the statistic tends to the total entry count divided by the root sum of
    squared cluster sizes, which is the root of the cluster count when clusters
    are equal. Inverting the null-imposed test therefore returns unbounded sets
    whenever that limit is not extreme against the bootstrap reference
    distribution, which at nine clusters it usually is not. The interval below
    fixes the residuals at the estimate instead, which keeps it finite.
    """
    total = float(np.sum(sizes))
    scale = float(np.sqrt(np.sum(np.asarray(sizes, dtype=float) ** 2)))
    return total / scale if scale > 0 else float("inf")


def wild_cluster_interval(
    sums: np.ndarray,
    sizes: np.ndarray,
    n_bootstrap: int = 1999,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Studentised wild cluster interval, with residuals taken at the estimate.

    Cluster residuals are given Rademacher signs, each resample is studentised
    by its own cluster-robust standard error, and the quantiles of that
    distribution are applied to the observed standard error.
    """
    total = float(np.sum(sizes))
    if total <= 0:
        return float("nan"), float("nan")

    estimate = float(np.sum(sums)) / total
    residual = np.asarray(sums, dtype=float) - estimate * np.asarray(sizes, dtype=float)
    standard_error = float(np.sqrt(np.sum(residual**2))) / total
    if standard_error <= 0:
        return estimate, estimate

    rng = np.random.default_rng(seed)
    signs = rademacher_signs(n_bootstrap, residual.shape[0], rng)
    drawn = signs * residual[None, :]
    shifts = drawn.sum(axis=1) / total
    residuals = drawn - shifts[:, None] * np.asarray(sizes, dtype=float)[None, :]
    errors = np.sqrt(np.sum(residuals**2, axis=1)) / total

    usable = errors > 0
    if not np.any(usable):
        return estimate, estimate
    statistics = shifts[usable] / errors[usable]

    tail = (1.0 - level) / 2.0
    low, high = np.quantile(statistics, [tail, 1.0 - tail])
    return estimate - float(high) * standard_error, estimate - float(low) * standard_error


def cluster_bootstrap(
    values: np.ndarray,
    clusters: np.ndarray,
    n_bootstrap: int = 1999,
    level: float = 0.95,
    seed: int = 0,
    method: str = WILD,
) -> BootstrapInterval:
    """Interval on a mean, resampling source publications."""
    labels, sums, sizes = cluster_totals(values, clusters)
    if method == WILD:
        lower, upper = wild_cluster_interval(sums, sizes, n_bootstrap, level, seed)
    elif method == PERCENTILE:
        lower, upper = percentile_cluster_interval(sums, sizes, n_bootstrap, level, seed)
    else:
        raise ValueError(f"unknown interval method: {method}")

    return BootstrapInterval(
        estimate=float(np.mean(np.asarray(values, dtype=float))),
        lower=lower,
        upper=upper,
        n_clusters=int(labels.size),
        level=level,
        method=method,
    )


def paired_margin(
    model_hits: np.ndarray,
    reference_hits: np.ndarray,
    clusters: np.ndarray,
    name: str,
    reference: str,
    n_bootstrap: int = 1999,
    level: float = 0.95,
    seed: int = 0,
    method: str = WILD,
) -> MarginResult:
    """Margin of one predictor over another, scored on the same entries."""
    difference = np.asarray(model_hits, dtype=float) - np.asarray(reference_hits, dtype=float)
    interval = cluster_bootstrap(
        difference, clusters, n_bootstrap=n_bootstrap, level=level, seed=seed, method=method
    )
    return MarginResult(
        name=name, reference=reference, interval=interval, n_entries=difference.shape[0]
    )


def report_in_both_conventions(
    mode: int, spacer_length: int, numbering: str
) -> dict[str, int]:
    """A window mode expressed both as a protospacer index and from the PAM."""
    return {
        "protospacer_index": int(mode),
        "nucleotides_from_pam": int(pam_distance(mode, spacer_length, numbering)),
    }
