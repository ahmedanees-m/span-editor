"""Baselines against which the model is scored.

All five are implemented before the model is evaluated. Every hypothesis
threshold is a margin over the marginal baseline, so the marginal case is not a
formality: an absolute hit rate can look high while measuring nothing, because
most SpCas9 editors peak in a narrow band of protospacer positions.

Distance-plus-sterics is the real competitor. It is the logic used for TALE
systems, transposed to an R-loop substrate, and it is given the same
opportunity to fit as the model itself.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .model import ArchitectureContext
from .occupancy import ModelParameters, Profile, saturating_link
from .tether import LinkerSpec

__all__ = [
    "Baseline",
    "DistanceStericsBaseline",
    "LinearLinkerBaseline",
    "MarginalBaseline",
    "ModalLinkerBaseline",
    "SubstrateOnlyBaseline",
    "default_baselines",
]


class Baseline(ABC):
    """Common interface: fit on the training subset, then predict per entry."""

    name: str = "baseline"

    @abstractmethod
    def fit(self, contexts: list[ArchitectureContext]) -> None: ...

    @abstractmethod
    def predict(self, context: ArchitectureContext) -> Profile: ...

    def predict_mode(self, context: ArchitectureContext) -> int:
        return self.predict(context).mode


class MarginalBaseline(Baseline):
    """Predict the corpus-mean profile everywhere."""

    name = "marginal"

    def __init__(self) -> None:
        self._mean: dict[int, float] = {}

    def fit(self, contexts: list[ArchitectureContext]) -> None:
        totals: dict[int, list[float]] = {}
        for context in contexts:
            observed = context.entry.observed
            if observed is None:
                continue
            normalised = observed.normalised()
            for index, value in zip(normalised.indices, normalised.values):
                totals.setdefault(int(index), []).append(float(value))
        if not totals:
            raise ValueError("the marginal baseline needs at least one observed profile")
        self._mean = {index: float(np.mean(values)) for index, values in totals.items()}

    def predict(self, context: ArchitectureContext) -> Profile:
        indices = np.asarray(context.indices, dtype=int)
        fallback = float(np.mean(list(self._mean.values()))) if self._mean else 0.0
        values = np.asarray([self._mean.get(int(i), fallback) for i in indices])
        return Profile(indices=indices, values=values)


class SubstrateOnlyBaseline(Baseline):
    """R-loop exposure with the tether ignored.

    If this matches the full model, the window is set by which nucleotides the
    R-loop presents rather than by where the effector can reach.
    """

    name = "substrate_only"

    def __init__(self, persistence: float = 15.0, alpha: float = 1.0) -> None:
        self.persistence = float(persistence)
        self.alpha = float(alpha)

    def fit(self, contexts: list[ArchitectureContext]) -> None:
        self.alpha = _fit_link_alpha(contexts, self._exposure)

    def _exposure(self, context: ArchitectureContext) -> np.ndarray:
        substrate = context.substrate(self.persistence)
        exposure = np.zeros(len(context.indices))
        for slot, index in enumerate(context.indices):
            cloud, weights = substrate.cloud(index)
            if context.occupancy is None:
                exposure[slot] = 1.0
                continue
            free = ~context.occupancy.occluded(cloud)
            exposure[slot] = float(np.sum(weights[free]))
        return exposure

    def predict(self, context: ArchitectureContext) -> Profile:
        indices = np.asarray(context.indices, dtype=int)
        return Profile(indices=indices, values=saturating_link(self._exposure(context), self.alpha))


class DistanceStericsBaseline(Baseline):
    """Distance from the anchor to each position, with steric exclusion.

    This is the real competitor: the logic used for TALE systems, transposed to
    an R-loop substrate. The effector is a point at a distance from the anchor
    rather than a sampled ensemble, and a nucleotide is edited when it sits at
    that distance.

    An earlier version asked only whether a nucleotide was closer than a fitted
    reach. That is not the same claim and it is not a fair one: everything
    inside the reach ties at the top, the profile is a plateau rather than a
    window, and its mode is the midpoint of the plateau rather than a
    prediction. It scored zero for that reason rather than on the merits.

    A distance rule places the effector at a distance, so the band is two-sided
    and its width is fitted alongside its centre. Three quantities are fitted
    here, the reach, the width and the link, which is what the model itself
    gets. A width factor of one recovers the old one-sided rule exactly, so the
    earlier behaviour is inside this search space and the change can only help
    the baseline.
    """

    name = "distance_sterics"

    def __init__(
        self,
        persistence: float = 15.0,
        reach_factor: float = 0.6,
        width_factor: float = 0.3,
        alpha: float = 1.0,
    ) -> None:
        self.persistence = float(persistence)
        self.reach_factor = float(reach_factor)
        self.width_factor = float(width_factor)
        self.alpha = float(alpha)

    def fit(self, contexts: list[ArchitectureContext]) -> None:
        reaches = np.linspace(0.2, 1.0, 17)
        widths = np.concatenate([np.linspace(0.05, 0.9, 18), [1.0]])
        best = (np.inf, self.reach_factor, self.width_factor, self.alpha)
        for factor in reaches:
            for width in widths:
                self.reach_factor, self.width_factor = float(factor), float(width)
                alpha = _fit_link_alpha(contexts, self._reachable)
                score = _profile_loss(
                    contexts, lambda c: saturating_link(self._reachable(c), alpha)
                )
                if score < best[0]:
                    best = (score, float(factor), float(width), alpha)
        _, self.reach_factor, self.width_factor, self.alpha = best

    def _reach(self, context: ArchitectureContext) -> float:
        linker = context.entry.linker
        contour = linker.n_residues * context.rise
        offset = float(np.linalg.norm(context.effector.active_site_offset))
        return self.reach_factor * (contour + offset)

    def _reachable(self, context: ArchitectureContext) -> np.ndarray:
        substrate = context.substrate(self.persistence)
        reach = self._reach(context)
        band = self.width_factor * reach
        fraction = np.zeros(len(context.indices))
        for slot, index in enumerate(context.indices):
            cloud, weights = substrate.cloud(index)
            distance = np.linalg.norm(cloud, axis=1)
            within = np.abs(distance - reach) <= band
            if context.occupancy is not None:
                within &= ~context.occupancy.occluded(cloud)
            fraction[slot] = float(np.sum(weights[within]))
        return fraction

    def predict(self, context: ArchitectureContext) -> Profile:
        indices = np.asarray(context.indices, dtype=int)
        return Profile(
            indices=indices, values=saturating_link(self._reachable(context), self.alpha)
        )


class LinearLinkerBaseline(Baseline):
    """Window mode as a straight line in linker contour length.

    Published length series are close to linear, so this is a strong baseline
    for the linker-length hypothesis and a weak one everywhere else.
    """

    name = "linear_in_length"

    def __init__(self) -> None:
        self.intercept = 0.0
        self.slope = 0.0
        self.spread = 2.0

    def fit(self, contexts: list[ArchitectureContext]) -> None:
        lengths, modes, widths = [], [], []
        for context in contexts:
            observed = context.entry.observed
            if observed is None:
                continue
            lengths.append(context.entry.linker_residues * context.rise)
            modes.append(observed.mode)
            widths.append(observed.width())
        if len(lengths) < 2:
            raise ValueError("the linear baseline needs at least two observed profiles")
        design = np.column_stack([np.ones(len(lengths)), np.asarray(lengths)])
        solution, *_ = np.linalg.lstsq(design, np.asarray(modes, dtype=float), rcond=None)
        self.intercept, self.slope = float(solution[0]), float(solution[1])
        self.spread = max(float(np.mean(widths)) / 2.0, 1.0)

    def predict(self, context: ArchitectureContext) -> Profile:
        indices = np.asarray(context.indices, dtype=int)
        centre = self.intercept + self.slope * context.entry.linker_residues * context.rise
        values = np.exp(-0.5 * ((indices - centre) / self.spread) ** 2)
        return Profile(indices=indices, values=values)


@dataclass
class ModalLinkerBaseline:
    """The linker the field already uses, for scoring the inverse problem only.

    Published constructs cluster on a small number of linkers, so proposing the
    most common one scores well against any loose tolerance. An inversion result
    only means something if it beats this.
    """

    name: str = "modal_linker"
    linker: LinkerSpec = LinkerSpec(n_residues=16, composition="xten")

    def fit(self, entries) -> None:
        counts: dict[tuple[int, str], int] = {}
        for entry in entries:
            key = (entry.linker_residues, entry.linker_composition)
            counts[key] = counts.get(key, 0) + 1
        if counts:
            residues, composition = max(counts, key=counts.get)
            self.linker = LinkerSpec(n_residues=residues, composition=composition)

    def propose(self, target_mode: int) -> LinkerSpec:
        del target_mode
        return self.linker


def _fit_link_alpha(contexts, occupancy_fn, grid: np.ndarray | None = None) -> float:
    """Choose the single link steepness that best matches observed profile shape."""
    candidates = np.logspace(-2, 2, 41) if grid is None else grid
    best_alpha, best_score = float(candidates[0]), np.inf
    for alpha in candidates:
        score = _profile_loss(contexts, lambda c: saturating_link(occupancy_fn(c), alpha))
        if score < best_score:
            best_alpha, best_score = float(alpha), score
    return best_alpha


def _profile_loss(contexts, predictor) -> float:
    """Sum of squared error between normalised predicted and observed profiles."""
    total = 0.0
    seen = 0
    for context in contexts:
        observed = context.entry.observed
        if observed is None:
            continue
        predicted = Profile(
            indices=np.asarray(context.indices, dtype=int), values=predictor(context)
        ).normalised()
        target = observed.normalised()
        shared = np.intersect1d(predicted.indices, target.indices)
        if shared.size == 0:
            continue
        left = predicted.values[np.isin(predicted.indices, shared)]
        right = target.values[np.isin(target.indices, shared)]
        total += float(np.sum((left - right) ** 2))
        seen += 1
    return total if seen else np.inf


def default_baselines(parameters: ModelParameters) -> list[Baseline]:
    """The four profile baselines, built from the fitted parameter set."""
    return [
        MarginalBaseline(),
        SubstrateOnlyBaseline(persistence=parameters.ssdna_persistence),
        DistanceStericsBaseline(persistence=parameters.ssdna_persistence),
        LinearLinkerBaseline(),
    ]
