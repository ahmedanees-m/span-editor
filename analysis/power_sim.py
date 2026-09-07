"""Detectable margin as a function of the number of source publications."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from span_editor.power import (
    COMONOTONIC,
    INDEPENDENT,
    METHODS,
    PERCENTILE,
    Design,
    minimum_detectable,
    power_curve,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters", type=int, default=9)
    parser.add_argument("--entries-per-cluster", type=float, default=5.0)
    parser.add_argument("--base-rate", type=float, default=0.45)
    parser.add_argument("--icc", type=float, default=0.3)
    parser.add_argument("--pairing", choices=[COMONOTONIC, INDEPENDENT], default=COMONOTONIC)
    parser.add_argument("--delta-max", type=float, default=0.4)
    parser.add_argument("--delta-steps", type=int, default=17)
    parser.add_argument("--trials", type=int, default=600)
    parser.add_argument("--bootstrap", type=int, default=1500)
    parser.add_argument("--method", choices=METHODS, default=PERCENTILE)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument("--out", type=Path, default=Path("results/power_sim.json"))
    args = parser.parse_args(argv)

    design = Design(
        n_clusters=args.clusters,
        entries_per_cluster=args.entries_per_cluster,
        base_rate=args.base_rate,
        icc=args.icc,
        pairing=args.pairing,
    )
    deltas = np.linspace(0.0, args.delta_max, args.delta_steps)
    curve = power_curve(
        design, deltas, args.trials, args.bootstrap, args.seed, method=args.method
    )
    detectable = minimum_detectable(curve)

    report = {
        "design": design.as_dict(),
        "method": args.method,
        "trials": args.trials,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "curve": curve,
        "rejection_at_zero_margin": curve[0]["power"],
        "minimum_detectable_delta": detectable,
    }

    print(f"clusters {design.n_clusters}, entries per cluster {design.entries_per_cluster}")
    print(
        f"base rate {design.base_rate:.2f}, icc {design.icc:.2f}, "
        f"pairing {design.pairing}, interval {args.method}"
    )
    print(f"rejection at zero margin              {report['rejection_at_zero_margin']:.3f}")
    if detectable is None:
        print(f"no margin up to {args.delta_max:.2f} reaches 80 per cent power")
    else:
        print(f"smallest margin at 80 per cent power   {detectable:.3f}")
    print()
    print(" delta   power")
    for point in curve:
        print(f" {point['delta']:.3f}   {point['power']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
