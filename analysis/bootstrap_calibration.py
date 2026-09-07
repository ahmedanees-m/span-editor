"""Rejection rate of the cluster bootstrap at zero margin, by cluster count.

Compares the studentised wild bootstrap used for reporting against the
percentile construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from span_editor.power import (
    COMONOTONIC,
    INDEPENDENT,
    METHODS,
    Design,
    draw_trial,
    power_at,
)


def degenerate_fraction(design: Design, n_trials: int, rng: np.random.Generator) -> float:
    """Fraction of trials in which every paired difference is zero."""
    degenerate = 0
    for _ in range(n_trials):
        sums, _ = draw_trial(design, 0.0, rng)
        if not np.any(sums):
            degenerate += 1
    return degenerate / n_trials


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clusters", type=int, nargs="+", default=[6, 9, 12, 16, 30, 60, 100]
    )
    parser.add_argument("--entries-per-cluster", type=float, default=5.0)
    parser.add_argument("--base-rate", type=float, default=0.45)
    parser.add_argument("--icc", type=float, default=0.3)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--bootstrap", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--out", type=Path, default=Path("results/bootstrap_calibration.json"))
    args = parser.parse_args(argv)

    nominal = 1.0 - args.level
    rows = []
    for method in args.methods:
        for pairing in (INDEPENDENT, COMONOTONIC):
            for count in args.clusters:
                design = Design(
                    n_clusters=count,
                    entries_per_cluster=args.entries_per_cluster,
                    base_rate=args.base_rate,
                    icc=args.icc,
                    pairing=pairing,
                )
                rng = np.random.default_rng(args.seed + count)
                rate = power_at(
                    design, 0.0, args.trials, args.bootstrap, rng, args.level, method
                )
                degenerate = degenerate_fraction(
                    design, min(args.trials, 200), np.random.default_rng(args.seed + count)
                )
                rows.append(
                    {
                        "method": method,
                        "pairing": pairing,
                        "n_clusters": count,
                        "rejection_at_zero_margin": rate,
                        "degenerate_trials": degenerate,
                    }
                )

    report = {
        "nominal_rejection": nominal,
        "methods": args.methods,
        "level": args.level,
        "entries_per_cluster": args.entries_per_cluster,
        "base_rate": args.base_rate,
        "icc": args.icc,
        "trials": args.trials,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "rows": rows,
    }

    print(f"nominal rejection at a {args.level:.2f} interval: {nominal:.3f}")
    print()
    print("method      pairing        clusters  rejection  trials with no discordant pair")
    for row in rows:
        print(
            f"{row['method']:11s} {row['pairing']:13s} {row['n_clusters']:8d}"
            f"     {row['rejection_at_zero_margin']:.3f}     {row['degenerate_trials']:.3f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
