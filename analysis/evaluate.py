"""Score the model and the baselines on the profile corpus."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from span_editor.baselines import default_baselines
from span_editor.io import load_assigned_inputs, load_parameters, read_corpus
from span_editor.metrics import (
    mode_hit,
    mode_offset,
    paired_margin,
    report_in_both_conventions,
)
from span_editor.model import SpanModel, build_context
from span_editor.tether import TetherSampler

TOLERANCE = 1


def build_contexts(entries, assigned, structure_dir):
    contexts, skipped = [], []
    for entry in entries:
        try:
            contexts.append(build_context(entry, assigned, structure_dir))
        except (ValueError, FileNotFoundError) as error:
            skipped.append((entry.entry_id, str(error)))
    return contexts, skipped


def score(model, parameters, baselines, contexts):
    """Per-entry hits and signed mode offsets for the model and every baseline."""
    rows = []
    for context in contexts:
        observed = context.entry.observed
        if observed is None:
            continue
        predicted = model.predict(context, parameters)
        row = {
            "entry_id": context.entry.entry_id,
            "source_key": context.entry.source_key,
            "ortholog": context.entry.ortholog,
            "tier": context.entry.tier,
            "label": context.entry.label,
            "readout_class": context.entry.readout_class,
            "observed_mode": observed.mode,
            "predicted_mode": predicted.mode,
            "mode_offset": mode_offset(predicted, observed),
            "hit_span": float(mode_hit(predicted, observed, TOLERANCE)),
            "observed_width": observed.width(),
            "predicted_width": predicted.width(),
            "observed_profile": {
                int(index): round(float(value), 4)
                for index, value in zip(observed.normalised().indices, observed.normalised().values)
            },
            "predicted_profile": {
                int(index): round(float(value), 4)
                for index, value in zip(
                    predicted.normalised().indices, predicted.normalised().values
                )
            },
            "reported": {
                "observed": report_in_both_conventions(
                    observed.mode, context.spacer_length, context.numbering
                ),
                "predicted": report_in_both_conventions(
                    predicted.mode, context.spacer_length, context.numbering
                ),
            },
        }
        for baseline in baselines:
            guess = baseline.predict(context)
            row[f"hit_{baseline.name}"] = float(mode_hit(guess, observed, TOLERANCE))
            row[f"offset_{baseline.name}"] = mode_offset(guess, observed)
            # A baseline that ties most of its positions at the peak has not put
            # a window anywhere, and its mode is the midpoint of the tie. Scoring
            # that as a wrong window would flatter the model, so it is counted
            # separately and reported.
            row[f"degenerate_{baseline.name}"] = float(guess.degenerate)
        row["degenerate_span"] = float(predicted.degenerate)
        rows.append(row)
    return rows


def margins(rows, baseline_names, label):
    """Margin of the model over each baseline, resampling source publications."""
    if not rows:
        return []
    clusters = np.asarray([row["source_key"] for row in rows])
    model_hits = np.asarray([row["hit_span"] for row in rows])
    results = []
    for name in baseline_names:
        reference = np.asarray([row[f"hit_{name}"] for row in rows])
        margin = paired_margin(model_hits, reference, clusters, label, name)
        results.append(margin.as_dict())
    return results


def subset(rows, **conditions):
    return [
        row
        for row in rows
        if all(row.get(key) in values for key, values in conditions.items())
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--out", type=Path, default=Path("results/evaluation.json"))
    args = parser.parse_args(argv)

    assigned = load_assigned_inputs(args.assigned)
    parameters = load_parameters(args.params)
    entries = read_corpus(args.corpus)
    if not entries:
        print(f"{args.corpus} holds no entries")
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

    training = [c for c in contexts if c.entry.fits_parameters]
    baselines = default_baselines(parameters)
    for baseline in baselines:
        baseline.fit(training)
    names = [baseline.name for baseline in baselines]

    rows = score(model, parameters, baselines, contexts)
    held_out = subset(rows, tier={"A", "B", "C"})
    sequencing = [row for row in rows if row["readout_class"] == "amplicon_sequencing"]

    by_ortholog = defaultdict(list)
    for row in sequencing:
        by_ortholog[row["ortholog"]].append(row)

    report = {
        "parameters": parameters.as_dict(),
        "n_entries": len(rows),
        "n_clusters": len({row["source_key"] for row in rows}),
        "baselines": names,
        "h1_mode_sufficiency": margins(
            subset(sequencing, ortholog={"SpCas9"}, label={"tether_varied"}), names, "h1"
        ),
        "h2_anchor_transfer": {
            "all": margins(
                subset(sequencing, label={"tether_varied", "scaffold_perturbed"}), names, "h2"
            ),
            "without_scaffold_perturbed": margins(
                subset(sequencing, label={"tether_varied"}), names, "h2_clean"
            ),
        },
        "h3_ortholog_transfer": {
            ortholog: margins(subset_rows, names, f"h3_{ortholog}")
            for ortholog, subset_rows in by_ortholog.items()
        },
        "specificity_control": {
            "n": len(subset(rows, label={"effector_varied"})),
            "mode_offsets": [
                row["mode_offset"] for row in subset(rows, label={"effector_varied"})
            ],
        },
        "residuals": {
            "signed_mode_offsets": [row["mode_offset"] for row in rows],
            "by_ortholog": {
                ortholog: [row["mode_offset"] for row in subset_rows]
                for ortholog, subset_rows in by_ortholog.items()
            },
        },
        "width": {
            "note": "descriptive only; comparisons restricted to one assay at comparable efficiency",
            "pairs": [
                {"observed": row["observed_width"], "predicted": row["predicted_width"]}
                for row in sequencing
            ],
        },
        "entries": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")

    print(f"entries {len(rows)} across {report['n_clusters']} source publications")
    print(f"held out {len(held_out)}")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
