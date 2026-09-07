"""Mean signed mode offset under leave-one-out scoring at several sampling seeds.

Seeds run in parallel. The configured seed leads the list by default.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from span_editor.io import load_assigned_inputs, load_parameters, read_corpus
from span_editor.metrics import cluster_bootstrap
from span_editor.model import SpanModel, build_context
from span_editor.tether import TetherSampler

CAS12A = "Cas12a"


def observed_mode(entry) -> int | None:
    if entry.observed is not None:
        return int(entry.observed.mode)
    if entry.observed_window:
        return int(round(0.5 * sum(entry.observed_window)))
    return None


def score_one_seed(job: tuple) -> tuple[int, list]:
    """Leave-one-out modes for every entry at one seed."""
    seed, corpus_path, assigned_path, params_path, structures = job
    assigned = load_assigned_inputs(assigned_path)
    parameters = load_parameters(params_path)
    corpus = [
        e for e in read_corpus(corpus_path)
        if e.observed is not None or e.observed_window
    ]
    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=assigned.sampling.get("tether_chains", 20000),
        seed=seed,
        target_surviving=assigned.sampling.get("target_surviving", 0),
    )
    model = SpanModel(sampler)
    rows = []
    for entry in corpus:
        try:
            context = build_context(entry, assigned, structures)
            profile = model.predict_leave_one_out(context, parameters).normalised()
        except (ValueError, FileNotFoundError, RuntimeError):
            continue
        observed = observed_mode(entry)
        if observed is None:
            continue
        rows.append(
            {
                "entry_id": entry.entry_id,
                "source_key": entry.source_key,
                "ortholog": entry.ortholog,
                "family": CAS12A if entry.ortholog == CAS12A else "SpCas9 family",
                "observed_mode": observed,
                "mode": None if profile.degenerate else int(profile.mode),
                "degenerate": bool(profile.degenerate),
            }
        )
    return seed, rows


def summarise(rows: list, label: str) -> dict:
    keep = [r for r in rows if not r["degenerate"]]
    if not keep:
        return {}
    offsets = np.asarray([r["mode"] - r["observed_mode"] for r in keep], dtype=float)
    clusters = np.asarray([r["source_key"] for r in keep])
    record = {"label": label, "n": len(keep), "publications": len(set(clusters)),
              "mean": round(float(offsets.mean()), 3)}
    if len(set(clusters)) >= 2:
        interval = cluster_bootstrap(offsets, clusters)
        record.update(lower=round(float(interval.lower), 3),
                      upper=round(float(interval.upper), 3),
                      excludes_zero=bool(interval.excludes_zero))
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="defaults to the configured seed followed by four others",
    )
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("results/bias_seed_stability.json"))
    args = parser.parse_args(argv)

    # The configured seed leads the list by default.
    configured = int(load_assigned_inputs(args.assigned).sampling.get("seed", 0))
    if args.seeds is None:
        args.seeds = [configured, 1, 2, 3, 4]
    elif configured not in args.seeds:
        print(f"note: the configured seed {configured} is not in {args.seeds}, "
              f"so none of these is the baseline")
    jobs = [
        (seed, args.corpus, args.assigned, args.params, args.structures)
        for seed in args.seeds
    ]
    print(f"leave-one-out at seeds {args.seeds}, {args.jobs} in parallel")
    by_seed = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for seed, rows in pool.map(score_one_seed, jobs):
            by_seed[seed] = rows
            print(f"  seed {seed} done, {len(rows)} entries scored", flush=True)

    print()
    print(f"{'seed':>5s} {'set':16s} {'n':>3s} {'pubs':>5s} {'mean':>8s} "
          f"{'interval':>20s}  excludes zero")
    summary = {}
    for seed in args.seeds:
        rows = by_seed.get(seed, [])
        summary[str(seed)] = {}
        for label, sel in (
            ("all entries", rows),
            ("SpCas9 family", [r for r in rows if r["family"] != CAS12A]),
            (CAS12A, [r for r in rows if r["family"] == CAS12A]),
        ):
            rec = summarise(sel, label)
            if not rec:
                continue
            summary[str(seed)][label] = rec
            span = (f"[{rec['lower']:+.3f}, {rec['upper']:+.3f}]"
                    if "lower" in rec else "one publication")
            print(f"{seed:5d} {label:16s} {rec['n']:3d} {rec['publications']:5d} "
                  f"{rec['mean']:+8.3f} {span:>20s}  "
                  f"{rec.get('excludes_zero', '')}")

    print()
    for label in ("all entries", "SpCas9 family"):
        means = [summary[s][label]["mean"] for s in summary if label in summary[s]]
        excl = [summary[s][label].get("excludes_zero") for s in summary if label in summary[s]]
        if not means:
            continue
        print(f"{label}: mean spans {min(means):+.3f} to {max(means):+.3f} "
              f"(range {max(means) - min(means):.3f} nt); "
              f"interval excludes zero at {sum(1 for e in excl if e)} of {len(excl)} seeds")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"seeds": args.seeds, "summary": summary,
                   "entries": {str(k): v for k, v in by_seed.items()}},
                  handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
