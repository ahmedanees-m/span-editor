"""Fit the three parameters on the training subset.

Writes the fitted values and a record of the input files they were fitted from.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from span_editor.fitting import fit_parameters
from span_editor.io import load_assigned_inputs, read_corpus, write_fit_record
from span_editor.model import SpanModel, build_context
from span_editor.occupancy import capture_probability_curve
from span_editor.tether import TetherSampler


def peak_occupancy(model, contexts, radius: float, persistence: float) -> float:
    """Largest capture probability over the fitting subset at the reference radius."""
    peak = 0.0
    for context in contexts:
        tether = model.sampler.sample(
            linker=context.entry.linker,
            effector=context.effector,
            anchor=context.anchor,
            occupancy=context.occupancy,
            structure_id=context.entry.structure_id or "none",
        )
        _, curve = capture_probability_curve(
            tether, context.substrate(persistence), np.array([radius]), indices=context.indices
        )
        peak = max(peak, float(curve.max()))
    return peak


def build_contexts(entries, assigned, structure_dir):
    contexts, skipped = [], []
    for entry in entries:
        try:
            contexts.append(build_context(entry, assigned, structure_dir))
        except (ValueError, FileNotFoundError) as error:
            skipped.append((entry.entry_id, str(error)))
    return contexts, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--fitted", type=Path, default=Path("configs/fitted.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("results/fit.json"))
    parser.add_argument(
        "--fit-record", type=Path, default=Path("results/fit_record.json")
    )
    args = parser.parse_args(argv)

    assigned = load_assigned_inputs(args.assigned)
    entries = [entry for entry in read_corpus(args.corpus) if entry.fits_parameters]
    if not entries:
        print(f"no entries in {args.corpus} belong to the fitting subset")
        return 1

    contexts, skipped = build_contexts(entries, assigned, args.structures)
    for entry_id, reason in skipped:
        print(f"skipped {entry_id}: {reason}")
    if not contexts:
        print("no entry could be assembled against a structure")
        return 1

    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=assigned.sampling.get("tether_chains", 20000),
        seed=assigned.sampling.get("seed", 0),
        cache_dir=args.cache,
        target_surviving=assigned.sampling.get("target_surviving", 0),
    )
    model = SpanModel(sampler)

    settings = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    bounds = settings["bounds"]
    radii = np.arange(bounds["capture_radius"][0], bounds["capture_radius"][1] + 0.01, 1.0)
    persistences = np.array([6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])

    # The link is swept on its saturation index, which is the steepness times
    # the largest capture probability at the reference radius. Sweeping the raw
    # steepness would put the grid in the wrong place, since that probability
    # depends on the geometry rather than on the model.
    reference = float(settings["link"]["reference_radius"])
    scale = peak_occupancy(model, contexts, reference, 15.0)
    indices = np.logspace(-2, np.log10(60.0), 25)
    alphas = indices / scale if scale > 0 else np.logspace(-1, 3, 25)

    result = fit_parameters(model, contexts, radii, persistences, alphas, n_jobs=args.jobs)
    fitted = result.parameters

    fitted = {
        "fitted": fitted.as_dict(),
        "saturation_index": float(fitted.link_alpha * scale),
        "peak_capture_probability": float(scale),
        "fit_loss": float(result.loss),
        "status": "fitted",
        "objective": "profile_shape",
        "fitted_on": sorted(context.entry.entry_id for context in contexts),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.fitted.write_text(
        yaml.safe_dump(fitted, sort_keys=False, default_flow_style=False), encoding="utf-8"
    )

    record = write_fit_record([args.params, args.assigned, args.fitted], args.fit_record)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({**result.as_dict(), "fit_record": record}, handle, indent=2)
        handle.write("\n")

    print(f"fitted on {result.n_entries} entries")
    for name, value in fitted.as_dict().items():
        print(f"  {name:20s} {value:.4f}")
    print(f"loss {result.loss:.4f}")
    print(f"fit record written to {args.fit_record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
