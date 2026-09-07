"""The inverse problem: which tether puts the window where it is wanted.

Given a target window, the model is run forward over a grid of candidate
linkers and the candidates are ranked by how closely the predicted profile
matches. The comparison that matters is against the modal linker, since
published constructs cluster on a few lengths and proposing the most common one
already scores well against a loose tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .model import ArchitectureContext, SpanModel
from .occupancy import ModelParameters, Profile
from .tether import LinkerSpec

__all__ = [
    "InversionResult",
    "candidate_linkers",
    "invert_window",
    "reachable_positions",
]


@dataclass(frozen=True)
class InversionResult:
    linker: LinkerSpec
    predicted_mode: int
    predicted_width: int
    mode_error: int
    shape_error: float

    def as_dict(self) -> dict:
        return {
            "linker_residues": self.linker.n_residues,
            "linker_composition": self.linker.composition,
            "predicted_mode": self.predicted_mode,
            "predicted_width": self.predicted_width,
            "mode_error": self.mode_error,
            "shape_error": self.shape_error,
        }


def candidate_linkers(
    compositions: list[str], lengths: range | list[int]
) -> list[LinkerSpec]:
    """Every combination of composition class and contour length in residues."""
    return [
        LinkerSpec(n_residues=int(length), composition=composition)
        for composition in compositions
        for length in lengths
    ]


def _with_linker(context: ArchitectureContext, linker: LinkerSpec) -> ArchitectureContext:
    entry = replace(
        context.entry,
        linker_residues=linker.n_residues,
        linker_composition=linker.composition,
    )
    return replace(context, entry=entry, _ensembles=context._ensembles)


def invert_window(
    model: SpanModel,
    context: ArchitectureContext,
    parameters: ModelParameters,
    target: Profile | int,
    candidates: list[LinkerSpec],
) -> list[InversionResult]:
    """Rank candidate linkers by how well they reproduce a target window.

    target may be a full profile or a single mode. Results are sorted by mode
    error first and shape error second, so a candidate that puts the window in
    the right place is preferred over one that matches an overall shape while
    peaking elsewhere.
    """
    if isinstance(target, Profile):
        target_mode = target.mode
        target_profile: Profile | None = target.normalised()
    else:
        target_mode = int(target)
        target_profile = None

    results = []
    for linker in candidates:
        trial = _with_linker(context, linker)
        predicted = model.predict(trial, parameters).normalised()
        shape_error = float("nan")
        if target_profile is not None:
            shared = np.intersect1d(predicted.indices, target_profile.indices)
            if shared.size:
                left = predicted.values[np.isin(predicted.indices, shared)]
                right = target_profile.values[np.isin(target_profile.indices, shared)]
                shape_error = float(np.mean((left - right) ** 2))
        results.append(
            InversionResult(
                linker=linker,
                predicted_mode=predicted.mode,
                predicted_width=predicted.width(),
                mode_error=abs(predicted.mode - target_mode),
                shape_error=shape_error,
            )
        )

    return sorted(
        results,
        key=lambda item: (
            item.mode_error,
            item.shape_error if np.isfinite(item.shape_error) else 0.0,
        ),
    )


def reachable_positions(
    model: SpanModel,
    context: ArchitectureContext,
    parameters: ModelParameters,
    candidates: list[LinkerSpec],
    threshold: float = 0.5,
) -> dict[int, list[LinkerSpec]]:
    """Map each protospacer position to the linkers that put the window on it."""
    reachable: dict[int, list[LinkerSpec]] = {}
    for linker in candidates:
        trial = _with_linker(context, linker)
        profile = model.predict(trial, parameters).normalised()
        for index, value in zip(profile.indices, profile.values):
            if value >= threshold:
                reachable.setdefault(int(index), []).append(linker)
    return reachable
