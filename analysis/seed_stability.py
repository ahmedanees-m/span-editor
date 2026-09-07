"""Predicted modes at several sampling seeds.

Reports the per-entry spread and the mean signed offset at each seed. The
configured seed leads the list by default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from span_editor.io import load_assigned_inputs, load_parameters, read_corpus
from span_editor.model import SpanModel, build_context
from span_editor.tether import TetherSampler

PROFILE_SOURCE = "kissling2025"


def observed_mode(entry) -> int | None:
    if entry.observed is not None:
        return int(entry.observed.mode)
    if entry.observed_window:
        return int(round(0.5 * sum(entry.observed_window)))
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--architectures-only",
        action="store_true",
        default=True,
        help="the entries that carry claims, rather than the concentrated profile source",
    )
    parser.add_argument("--all-entries", dest="architectures_only", action="store_false")
    parser.add_argument("--out", type=Path, default=Path("results/seed_stability.json"))
    args = parser.parse_args(argv)

    assigned = load_assigned_inputs(args.assigned)
    parameters = load_parameters(args.params)
    # The configured seed leads the list, so the sweep includes the setting
    # the stored results were computed at.
    configured = int(assigned.sampling.get("seed", 0))
    if args.seeds is None:
        args.seeds = [configured, 1, 2, 3, 4]
    corpus = [
        entry
        for entry in read_corpus(args.corpus)
        if (entry.observed is not None or entry.observed_window)
        and not (args.architectures_only and entry.source_key == PROFILE_SOURCE)
    ]
    print(f"{len(corpus)} entries, seeds {args.seeds}")
    print()

    contexts = {}
    for entry in corpus:
        try:
            contexts[entry.entry_id] = build_context(entry, assigned, args.structures)
        except (ValueError, FileNotFoundError) as error:
            print(f"skipped {entry.entry_id}: {error}")

    seen: dict[str, dict[int, dict]] = {}
    for seed in args.seeds:
        sampler = TetherSampler(
            compositions=assigned.compositions,
            n_chains=assigned.sampling.get("tether_chains", 20000),
            seed=seed,
            target_surviving=assigned.sampling.get("target_surviving", 0),
        )
        model = SpanModel(sampler)
        for entry in corpus:
            context = contexts.get(entry.entry_id)
            if context is None:
                continue
            field = sampler.sample(
                linker=entry.linker,
                effector=context.effector,
                anchor=context.anchor,
                occupancy=context.occupancy,
                structure_id=entry.structure_id or "none",
                closure=context.closure,
            )
            profile = model.predict(context, parameters)
            normalised = profile.normalised()
            capture = np.asarray(profile.values, dtype=float)
            peak = float(np.nanmax(capture)) if capture.size else float("nan")
            seen.setdefault(entry.entry_id, {})[seed] = {
                "mode": None if normalised.degenerate else int(normalised.mode),
                "degenerate": bool(normalised.degenerate),
                "peak": round(peak, 6),
                "effective_conformations": round(float(field.effective_size), 1),
            }
        print(f"  seed {seed} done")

    rows = []
    for entry in corpus:
        by_seed = seen.get(entry.entry_id, {})
        modes = [v["mode"] for v in by_seed.values() if v["mode"] is not None]
        if not by_seed:
            continue
        rows.append(
            {
                "entry_id": entry.entry_id,
                "source_key": entry.source_key,
                "ortholog": entry.ortholog,
                "observed_mode": observed_mode(entry),
                "by_seed": {str(k): v for k, v in sorted(by_seed.items())},
                "modes": modes,
                "mode_spread": (max(modes) - min(modes)) if len(modes) > 1 else None,
                "degenerate_seeds": sum(1 for v in by_seed.values() if v["degenerate"]),
                "effective_conformations": by_seed[args.seeds[0]]["effective_conformations"],
            }
        )

    print()
    header = " ".join(f"s{s}" for s in args.seeds)
    print(f"{'entry':38s} {'obs':>4s}  {header}  {'spread':>6s} {'effective':>10s}")
    for row in rows:
        line = " ".join(
            f"{row['by_seed'][str(s)]['mode'] if row['by_seed'][str(s)]['mode'] is not None else '.':>2}"
            for s in args.seeds
        )
        spread = "-" if row["mode_spread"] is None else str(row["mode_spread"])
        print(
            f"{row['entry_id']:38s} {str(row['observed_mode']):>4s}  {line}  {spread:>6s} "
            f"{row['effective_conformations']:10.0f}"
        )

    # Mean signed offset per seed, on the held treatment this sweep computes.
    # analysis/bias_seed_stability.py reports the same quantity under
    # leave-one-out scoring.
    print()
    print("mean signed offset per seed, predicted minus observed, held treatment")
    per_seed = {}
    for seed in args.seeds:
        offsets, families = [], {}
        for r in rows:
            v = r["by_seed"].get(str(seed))
            if not v or v["mode"] is None or r["observed_mode"] is None:
                continue
            d = v["mode"] - r["observed_mode"]
            offsets.append(d)
            families.setdefault(
                "Cas12a" if r["ortholog"] == "Cas12a" else "SpCas9 family", []
            ).append(d)
        if not offsets:
            continue
        per_seed[str(seed)] = {
            "n": len(offsets),
            "mean": round(float(np.mean(offsets)), 3),
            "by_family": {k: round(float(np.mean(v)), 3) for k, v in families.items()},
        }
        parts = "  ".join(f"{k} {np.mean(v):+.2f} (n={len(v)})" for k, v in sorted(families.items()))
        print(f"  seed {seed}: all {np.mean(offsets):+.3f} nt over {len(offsets)} entries   {parts}")
    if per_seed:
        means = [v["mean"] for v in per_seed.values()]
        print(f"  across seeds the mean signed offset spans {min(means):+.3f} to {max(means):+.3f}, "
              f"range {max(means) - min(means):.3f} nt")

    spreads = [r["mode_spread"] for r in rows if r["mode_spread"] is not None]
    print()
    if spreads:
        print(
            f"mode spread across seeds: median {int(np.median(spreads))}, max {max(spreads)}; "
            f"{sum(1 for v in spreads if v == 0)} of {len(spreads)} entries never move, "
            f"{sum(1 for v in spreads if v > 1)} move by more than one"
        )
    unstable = [r["entry_id"] for r in rows if (r["mode_spread"] or 0) > 1]
    if unstable:
        print(f"entries whose mode is not seed-stable: {unstable}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "seeds": args.seeds,
                "entries": rows,
                "median_mode_spread": int(np.median(spreads)) if spreads else None,
                "max_mode_spread": int(max(spreads)) if spreads else None,
                "mean_signed_offset_per_seed": per_seed,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
