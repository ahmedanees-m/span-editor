"""Saturation index and fit loss with and without the link term."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from span_editor.fitting import fit_parameters
from span_editor.io import load_assigned_inputs, read_corpus
from span_editor.model import SpanModel, build_context
from span_editor.occupancy import ModelParameters, saturating_link
from span_editor.tether import TetherSampler


def departure_from_linear(index: float, steps: int = 200) -> float:
    """Largest relative gap between the link and a straight line, over the range seen.

    Both are scaled to agree at the top of the range, since a profile is
    compared as a shape. What is left is curvature.
    """
    if index <= 0:
        return 0.0
    argument = np.linspace(0.0, index, steps)
    linked = 1.0 - np.exp(-argument)
    straight = argument * (linked[-1] / argument[-1])
    return float(np.max(np.abs(linked - straight)) / linked[-1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--ceiling", type=float, default=60.0, help="per cent editing")
    parser.add_argument("--out", type=Path, default=Path("results/link_decision.json"))
    args = parser.parse_args(argv)

    document = yaml.safe_load(
        (args.params.with_name("fitted.yaml") if args.params.with_name("fitted.yaml").exists() else args.params).read_text(encoding="utf-8")
    )
    if document.get("status") != "fitted":
        print("no fitted parameters found; run the fit first")
        return 1

    assigned = load_assigned_inputs(args.assigned)
    fitted = ModelParameters.from_dict(document["fitted"])
    index = float(document.get("saturation_index", float("nan")))
    settings = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    threshold = float(settings["link"]["linear_regime_below"])

    entries = [entry for entry in read_corpus(args.corpus) if entry.observed]
    fitting = [entry for entry in entries if entry.fits_parameters]
    contexts = [build_context(entry, assigned, args.structures) for entry in fitting]

    # Measured efficiency at the centre of the window, over every entry that has
    # one, on the scale the source reports.
    centres = [float(entry.observed.values.max()) for entry in entries]
    at_ceiling = [value for value in centres if value >= args.ceiling]

    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=assigned.sampling.get("tether_chains", 20000),
        seed=assigned.sampling.get("seed", 0),
        cache_dir=args.cache,
        target_surviving=assigned.sampling.get("target_surviving", 0),
    )
    model = SpanModel(sampler)

    # Refit with the link held in the linear regime, so the comparison is
    # between three parameters and two rather than between two fits.
    radii = np.arange(
        settings["bounds"]["capture_radius"][0],
        settings["bounds"]["capture_radius"][1] + 0.01,
        1.0,
    )
    persistences = np.array([6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])
    linear_alpha = np.array([fitted.link_alpha * 1e-3])

    two = fit_parameters(model, contexts, radii, persistences, linear_alpha, n_jobs=args.jobs)
    three_loss = float(document.get("fit_loss", float("nan")))
    if not np.isfinite(three_loss):
        three = fit_parameters(
            model,
            contexts,
            radii,
            persistences,
            np.array([fitted.link_alpha]),
            n_jobs=args.jobs,
        )
        three_loss = three.loss

    below_threshold = index < threshold
    below_ceiling = not at_ceiling
    departure = departure_from_linear(index)
    penalty = two.loss / three_loss if three_loss > 0 else float("inf")

    report = {
        "saturation_index": index,
        "threshold": threshold,
        "index_below_threshold": below_threshold,
        "departure_from_linear": departure,
        "entries_at_or_above_ceiling": len(at_ceiling),
        "ceiling_per_cent": args.ceiling,
        "highest_measured_efficiency": max(centres) if centres else None,
        "loss_with_the_link": three_loss,
        "loss_with_the_link_held_linear": two.loss,
        "loss_penalty_without_the_link": penalty,
        "parameters_with_the_link_held_linear": two.parameters.as_dict(),
    }

    print(f"saturation index at the fit          {index:.4f}")
    print(f"threshold below which it is linear   {threshold}")
    print(f"departure of the link from a line    {100 * departure:.1f} per cent")
    print(f"highest measured efficiency          {max(centres):.1f} per cent")
    print(f"entries at or above {args.ceiling:.0f} per cent      {len(at_ceiling)} of {len(centres)}")
    print()
    print(f"loss with three parameters           {three_loss:.5f}")
    print(f"loss with the link held linear       {two.loss:.5f}")
    print(f"loss penalty for dropping it         {penalty:.2f} times")
    print(f"parameters without the link          {two.parameters.as_dict()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
