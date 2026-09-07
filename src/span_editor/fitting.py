"""Fitting the three parameters on profile shape.

Mode is invariant under any monotone link, so fitting on mode accuracy would
leave the link parameter with nothing to identify it while still adding
flexibility downstream. Capture radius and link steepness are degenerate against
a mode-only objective for the same reason: a larger radius raises every
occupancy and a shallower link absorbs it. Fitting therefore runs on full
per-position profile shape, and mode is kept for the held-out claims.

Only the flexibility of the displaced strand forces the substrate ensemble to be
rebuilt. The capture radius and the link are swept from a single pass over the
pairwise distances.

Sources report editing on whatever scale they use, so profiles are compared as
shapes. How that scale is removed matters more than it looks: dividing by the
peak biases the fitted link toward the linear end, because the peak is a single
order statistic. Solving for the multiplicative constant in closed form uses
every position instead and removes the bias, which is why it is the default.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .model import ArchitectureContext, SpanModel
from .occupancy import ModelParameters, capture_probability_curve, saturating_link

__all__ = [
    "LEAST_SQUARES_SCALE",
    "PEAK_SCALE",
    "SCALES",
    "FitResult",
    "fit_parameters",
    "profile_loss",
    "shape_loss",
]

PEAK_SCALE = "peak"
LEAST_SQUARES_SCALE = "least_squares"
SCALES = (PEAK_SCALE, LEAST_SQUARES_SCALE)


def shape_loss(predicted: np.ndarray, observed: np.ndarray, scale: str) -> np.ndarray:
    """Squared error between a predicted profile and an observed one, up to scale.

    Profiles are compared as shapes because sources report editing on whatever
    scale they use. Two ways of removing that scale are available.

    Dividing by the peak is the obvious one and is what reading a window off a
    published figure amounts to. It is also biased: the observed peak is one
    order statistic and noise inflates it, so every other point is pushed down
    and the profile looks flatter than it is.

    Solving for the multiplicative constant that minimises the squared error
    uses every position rather than one, and removes that bias. It is the same
    operation done better, not an extra fitted quantity: the constant is
    determined in closed form per entry and never carried between entries, so it
    adds nothing to the parameter count and nothing that could travel from the
    training subset to a held-out architecture.

    predicted has shape (positions, candidates); observed has shape (positions,).
    """
    if scale == PEAK_SCALE:
        peak = predicted.max(axis=0)
        peak = np.where(peak > 0, peak, 1.0)
        return np.sum((predicted / peak - observed[:, None]) ** 2, axis=0)
    if scale == LEAST_SQUARES_SCALE:
        weight = predicted.T @ observed
        energy = np.sum(predicted**2, axis=0)
        factor = np.where(energy > 0, weight / np.where(energy > 0, energy, 1.0), 0.0)
        return np.sum((predicted * factor[None, :] - observed[:, None]) ** 2, axis=0)
    raise ValueError(f"unknown profile scale: {scale}")


@dataclass(frozen=True)
class FitResult:
    parameters: ModelParameters
    loss: float
    n_entries: int
    grid_shape: tuple[int, int, int]
    surface: np.ndarray
    scale: str = LEAST_SQUARES_SCALE

    def as_dict(self) -> dict:
        return {
            "parameters": self.parameters.as_dict(),
            "loss": self.loss,
            "n_entries": self.n_entries,
            "grid_shape": list(self.grid_shape),
            "scale": self.scale,
        }


def _observed_vector(context: ArchitectureContext) -> tuple[np.ndarray, np.ndarray]:
    """Observed values aligned to the context indices, with a mask of shared positions."""
    observed = context.entry.observed
    if observed is None:
        raise ValueError(f"{context.entry.entry_id} has no observed profile")
    normalised = observed.normalised()
    indices = np.asarray(context.indices, dtype=int)
    mask = np.isin(indices, normalised.indices)
    lookup = dict(zip(normalised.indices.tolist(), normalised.values.tolist()))
    values = np.asarray([lookup[int(i)] for i in indices[mask]])
    return mask, values


def _context_surface(
    model: SpanModel,
    context: ArchitectureContext,
    radii: np.ndarray,
    persistences: np.ndarray,
    alphas: np.ndarray,
    scale: str = LEAST_SQUARES_SCALE,
) -> np.ndarray:
    """Squared error over the parameter grid for one architecture."""
    mask, observed = _observed_vector(context)
    tether = model.sampler.sample(
        linker=context.entry.linker,
        effector=context.effector,
        anchor=context.anchor,
        occupancy=context.occupancy,
        structure_id=context.entry.structure_id or "none",
    )
    surface = np.zeros((persistences.shape[0], radii.shape[0], alphas.shape[0]))
    for slot, persistence in enumerate(persistences):
        substrate = context.substrate(float(persistence))
        _, curve = capture_probability_curve(tether, substrate, radii, indices=context.indices)
        occupancy = curve[mask]
        for column, alpha in enumerate(alphas):
            predicted = saturating_link(occupancy, float(alpha))
            surface[slot, :, column] = shape_loss(predicted, observed, scale)
    return surface


def fit_parameters(
    model: SpanModel,
    contexts: list[ArchitectureContext],
    radii: np.ndarray,
    persistences: np.ndarray,
    alphas: np.ndarray,
    n_jobs: int = 1,
    scale: str = LEAST_SQUARES_SCALE,
) -> FitResult:
    """Grid search over the three fitted parameters against profile shape.

    Architectures are independent given the grid, so the search is spread over
    processes when more than one is asked for.
    """
    if scale not in SCALES:
        raise ValueError(f"unknown profile scale: {scale}")
    usable = [c for c in contexts if c.entry.observed is not None and c.entry.fits_parameters]
    if not usable:
        raise ValueError("no entry in the fitting subset carries an observed profile")

    radii = np.sort(np.asarray(radii, dtype=float))
    persistences = np.asarray(persistences, dtype=float)
    alphas = np.asarray(alphas, dtype=float)

    if n_jobs > 1 and len(usable) > 1:
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(usable))) as pool:
            futures = [
                pool.submit(
                    _context_surface, model, context, radii, persistences, alphas, scale
                )
                for context in usable
            ]
            surfaces = [future.result() for future in futures]
    else:
        surfaces = [
            _context_surface(model, context, radii, persistences, alphas, scale)
            for context in usable
        ]
    surface = np.sum(surfaces, axis=0)

    best = np.unravel_index(int(np.argmin(surface)), surface.shape)
    parameters = ModelParameters(
        capture_radius=float(radii[best[1]]),
        ssdna_persistence=float(persistences[best[0]]),
        link_alpha=float(alphas[best[2]]),
    )
    return FitResult(
        parameters=parameters,
        loss=float(surface[best]),
        n_entries=len(usable),
        grid_shape=(persistences.shape[0], radii.shape[0], alphas.shape[0]),
        surface=surface,
        scale=scale,
    )


def profile_loss(
    model: SpanModel,
    contexts: list[ArchitectureContext],
    parameters: ModelParameters,
    scale: str = LEAST_SQUARES_SCALE,
) -> float:
    """Squared error between predicted and observed profiles, summed over entries."""
    total = 0.0
    for context in contexts:
        if context.entry.observed is None:
            continue
        predicted = model.predict(context, parameters)
        observed = context.entry.observed.normalised()
        shared = np.intersect1d(predicted.indices, observed.indices)
        if shared.size == 0:
            continue
        left = predicted.values[np.isin(predicted.indices, shared)]
        right = observed.values[np.isin(observed.indices, shared)]
        total += float(shape_loss(left[:, None], right, scale)[0])
    return total
