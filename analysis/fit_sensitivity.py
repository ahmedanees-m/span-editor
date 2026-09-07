"""Refit the parameters on several subsets and report how far they move."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from span_editor.fitting import fit_parameters
from span_editor.io import load_assigned_inputs, load_parameters, read_corpus
from span_editor.model import SpanModel, build_context
from span_editor.tether import TetherSampler

SUBSETS = {
    "training split, SpCas9-ABE8e plasmid": lambda e: e.fits_parameters,
    "every SpCas9 terminal fusion with a profile": lambda e: (
        e.ortholog == "SpCas9" and e.anchor_site == "n_terminal" and e.observed is not None
    ),
    "ABE8e only": lambda e: e.observed is not None and e.effector == "tada8e",
    "ABEmax only": lambda e: e.observed is not None and e.effector == "abemax",
    "every profile entry": lambda e: e.observed is not None,
}


def contexts_for(entries, assigned, structures):
    built = []
    for entry in entries:
        try:
            built.append(build_context(entry, assigned, structures))
        except (ValueError, FileNotFoundError):
            continue
    return built


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("results/fit_sensitivity.json"))
    args = parser.parse_args(argv)

    assigned = load_assigned_inputs(args.assigned)
    corpus = read_corpus(args.corpus)
    fitted = load_parameters(args.params)
    settings = yaml.safe_load(args.params.read_text(encoding="utf-8"))
    bounds = settings["bounds"]

    radii = np.arange(bounds["capture_radius"][0], bounds["capture_radius"][1] + 0.01, 1.0)
    persistences = np.array([6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 27.0, 33.0, 40.0])
    alphas = np.geomspace(
        bounds.get("link_alpha", [1.0, 5000.0])[0], bounds.get("link_alpha", [1.0, 5000.0])[1], 24
    )

    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=assigned.sampling.get("tether_chains", 20000),
        seed=assigned.sampling.get("seed", 0),
        cache_dir=args.cache,
        target_surviving=assigned.sampling.get("target_surviving", 0),
    )
    model = SpanModel(sampler)

    rows = []
    for label, predicate in SUBSETS.items():
        entries = [entry for entry in corpus if predicate(entry)]
        if not entries:
            continue
        built = contexts_for(entries, assigned, args.structures)
        usable = [c for c in built if c.entry.observed is not None]
        if not usable:
            continue
        # fit_parameters restricts to the fitting subset itself, so a subset that
        # reaches wider is fitted by relabelling. The entry is replaced rather
        # than mutated, since the corpus is read once and shared across subsets
        # and marking an entry here would change which subset the next one sees.
        usable = [
            replace(
                context,
                entry=replace(
                    context.entry, split="fit", readout_class="amplicon_sequencing"
                ),
                _ensembles=context._ensembles,
            )
            for context in usable
        ]
        result = fit_parameters(
            model, usable, radii, persistences, alphas, n_jobs=args.jobs
        )
        rows.append(
            {
                "subset": label,
                "n_entries": len(usable),
                "entries": [c.entry.entry_id for c in usable],
                "capture_radius": result.parameters.capture_radius,
                "ssdna_persistence": result.parameters.ssdna_persistence,
                "link_alpha": round(result.parameters.link_alpha, 3),
                "loss": round(float(result.loss), 6),
            }
        )
        print(
            f"{label:44s} n={len(usable):2d}  radius {result.parameters.capture_radius:5.1f}  "
            f"persistence {result.parameters.ssdna_persistence:5.1f}  "
            f"alpha {result.parameters.link_alpha:9.2f}  loss {result.loss:.5f}"
        )

    print()
    print(
        f"fitted: radius {fitted.capture_radius}, persistence {fitted.ssdna_persistence}, "
        f"alpha {fitted.link_alpha:.2f}"
    )
    for name, attribute in (
        ("capture radius", "capture_radius"),
        ("strand persistence", "ssdna_persistence"),
        ("link steepness", "link_alpha"),
    ):
        values = [row[attribute] for row in rows]
        print(
            f"  {name:20s} across subsets {min(values):9.2f} to {max(values):9.2f}, "
            f"fitted {getattr(fitted, attribute):9.2f}"
        )

    report = {
        "fitted": fitted.as_dict(),
        "subsets": rows,
        "spread": {
            attribute: {
                "min": min(row[attribute] for row in rows),
                "max": max(row[attribute] for row in rows),
                "fitted": getattr(fitted, attribute),
            }
            for attribute in ("capture_radius", "ssdna_persistence", "link_alpha")
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
