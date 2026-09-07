"""Hit rate against a marginal baseline on the architecture entries.

The profile source the parameters were fitted on is excluded, since every
architecture in it peaks at the same position. Intervals resample publications.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from span_editor.io import load_assigned_inputs, load_parameters, read_corpus
from span_editor.metrics import cluster_bootstrap
from span_editor.model import SpanModel, build_context
from span_editor.tether import TetherSampler

TOLERANCE = 1
PROFILE_SOURCE = "kissling2025"


def observed_mode(entry) -> int:
    if entry.observed is not None:
        return int(entry.observed.mode)
    return int(round(0.5 * sum(entry.observed_window)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument(
        "--leave-one-out",
        type=Path,
        default=Path("results/leave_one_out.json"),
        help="modes scored with each position's own coordinate withheld",
    )
    parser.add_argument("--out", type=Path, default=Path("results/architecture_margin.json"))
    args = parser.parse_args(argv)

    assigned = load_assigned_inputs(args.assigned)
    parameters = load_parameters(args.params)
    corpus = read_corpus(args.corpus)

    profiles = [e for e in corpus if e.source_key == PROFILE_SOURCE and e.observed is not None]
    architectures = [
        e
        for e in corpus
        if e.source_key != PROFILE_SOURCE and (e.observed is not None or e.observed_window)
    ]
    # The marginal baseline predicts the corpus mean, and the only corpus it has
    # seen is the profile source the parameters were fitted on.
    marginal = int(round(float(np.mean([observed_mode(e) for e in profiles]))))
    print(f"profile source supplies {len(profiles)} entries, all peaking at "
          f"{sorted(Counter(observed_mode(e) for e in profiles))}")
    print(f"the marginal baseline therefore predicts {marginal} everywhere")
    print(f"architecture entries: {len(architectures)} from "
          f"{len({e.source_key for e in architectures})} publications")
    print()

    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=assigned.sampling.get("tether_chains", 20000),
        seed=assigned.sampling.get("seed", 0),
        target_surviving=assigned.sampling.get("target_surviving", 0),
    )
    model = SpanModel(sampler)

    # Scoring a position against an ensemble that was told where it is measures
    # the deposition. Where leave-one-out modes exist they are used instead.
    withheld = {}
    if args.leave_one_out.exists():
        for row in json.loads(args.leave_one_out.read_text(encoding="utf-8"))["entries"]:
            withheld[row["entry_id"]] = row
        print(f"leave-one-out modes available for {len(withheld)} entries")

    rows, skipped = [], []
    for entry in architectures:
        try:
            context = build_context(entry, assigned, args.structures)
            profile = model.predict(context, parameters).normalised()
        except (ValueError, FileNotFoundError, RuntimeError) as error:
            skipped.append({"entry_id": entry.entry_id, "reason": str(error)})
            continue
        observed = observed_mode(entry)
        held_mode, held_degenerate = int(profile.mode), bool(profile.degenerate)
        record = withheld.get(entry.entry_id)
        if record is not None:
            predicted_mode = int(record["loose_mode"])
            degenerate = bool(record.get("loose_degenerate", False))
            source = "leave one out"
        else:
            predicted_mode, degenerate, source = held_mode, held_degenerate, "held"
        rows.append(
            {
                "entry_id": entry.entry_id,
                "source_key": entry.source_key,
                "ortholog": entry.ortholog,
                "observed_mode": observed,
                "predicted_mode": predicted_mode,
                "scored_from": source,
                "held_mode": held_mode,
                "degenerate": degenerate,
                "model_hit": float(abs(predicted_mode - observed) <= TOLERANCE),
                "marginal_hit": float(abs(marginal - observed) <= TOLERANCE),
            }
        )
        print(
            f"{entry.entry_id:32s} {entry.source_key:16s} observed {observed:2d}  "
            f"model {predicted_mode:2d} {'hit ' if rows[-1]['model_hit'] else 'miss'}  "
            f"marginal {marginal:2d} {'hit ' if rows[-1]['marginal_hit'] else 'miss'}"
            f"{'  no window' if degenerate else ''}"
        )

    usable = [row for row in rows if not row["degenerate"]]
    print()
    print(f"{len(rows)} scored, {len(rows) - len(usable)} with no window and set aside")
    for scope, chosen in (("all architecture entries", rows), ("those with a window", usable)):
        if not chosen:
            continue
        clusters = np.asarray([row["source_key"] for row in chosen])
        model_hits = np.asarray([row["model_hit"] for row in chosen])
        marginal_hits = np.asarray([row["marginal_hit"] for row in chosen])
        interval = cluster_bootstrap(model_hits - marginal_hits, clusters)
        print()
        print(f"{scope}: n={len(chosen)}, publications={len(set(clusters))}")
        print(f"  model    {model_hits.mean():.3f}")
        print(f"  marginal {marginal_hits.mean():.3f}")
        print(
            f"  margin   {interval.estimate:+.3f} "
            f"[{interval.lower:+.3f}, {interval.upper:+.3f}], "
            f"excludes zero: {interval.excludes_zero}"
        )

    families = {}
    for row in usable:
        families.setdefault(
            "Cas12a" if row["ortholog"] == "Cas12a" else "SpCas9 family", []
        ).append(row)
    print()
    for family, chosen in sorted(families.items()):
        hits = np.mean([row["model_hit"] for row in chosen])
        offsets = [row["predicted_mode"] - row["observed_mode"] for row in chosen]
        print(
            f"  {family:16s} n={len(chosen):2d}  within one nucleotide {hits:.2f}  "
            f"offsets {min(offsets):+d} to {max(offsets):+d}"
        )

    report = {
        "marginal_prediction": marginal,
        "tolerance": TOLERANCE,
        "profile_entries": len(profiles),
        "entries": rows,
        "skipped": skipped,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
