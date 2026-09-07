"""Cas12a windows predicted from parameters fitted on SpCas9.

Reports a band over attachment site and contour length alongside the prediction
for the deposited construct.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import yaml

from span_editor.geometry import pam_distance
from span_editor.io import load_assigned_inputs, load_parameters, read_corpus
from span_editor.model import SpanModel, build_context
from span_editor.tether import TetherSampler

LENGTHS = (8, 16, 24, 48)
SITES = ("n_terminal", "c_terminal")
THRESHOLD = 0.5


def overlap(first, second) -> float:
    low = max(first[0], second[0])
    high = min(first[1], second[1])
    shared = max(0, high - low + 1)
    union = max(first[1], second[1]) - min(first[0], second[0]) + 1
    return shared / union if union > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--ortholog", default="Cas12a")
    parser.add_argument("--skip", default="", help="comma separated entry ids to leave out")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--target-surviving", type=int, default=0)
    parser.add_argument("--max-chains", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/ortholog_transfer.json"))
    args = parser.parse_args(argv)

    fitted = args.params.with_name("fitted.yaml")
    document = yaml.safe_load((fitted if fitted.exists() else args.params).read_text("utf-8"))
    if document.get("status") != "fitted":
        print("no fitted parameters found; run the fit first")
        return 1

    assigned = load_assigned_inputs(args.assigned)
    parameters = load_parameters(args.params)
    entries = [
        entry
        for entry in read_corpus(args.corpus)
        if entry.ortholog == args.ortholog
        and entry.observed_window
        and entry.entry_id not in set(filter(None, args.skip.split(",")))
    ]
    if not entries:
        print(f"no {args.ortholog} entry carries a reported window")
        return 1

    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=assigned.sampling.get("tether_chains", 20000),
        seed=assigned.sampling.get("seed", 0),
        target_surviving=args.target_surviving or assigned.sampling.get("target_surviving", 0),
        max_chains=args.max_chains,
    )
    model = SpanModel(sampler)

    rows = []
    for entry in entries:
        band = []
        for site in SITES:
            context = build_context(
                replace(entry, anchor_site=site), assigned, args.structures
            )
            for length in LENGTHS:
                trial = replace(
                    context,
                    entry=replace(context.entry, linker_residues=length),
                    _ensembles=context._ensembles,
                )
                try:
                    profile = model.predict(trial, parameters).normalised()
                except RuntimeError as error:
                    band.append(
                        {
                            "site": site,
                            "length": length,
                            "buildable": False,
                            "reason": str(error),
                        }
                    )
                    continue
                window = profile.window(args.threshold)
                if window == (0, 0):
                    band.append(
                        {
                            "site": site,
                            "length": length,
                            "buildable": False,
                            "reason": "no position is captured at all",
                        }
                    )
                    continue
                band.append(
                    {
                        "site": site,
                        "length": length,
                        "buildable": True,
                        "mode": int(profile.mode),
                        "window": list(window),
                        "width": int(profile.width(args.threshold)),
                        "from_pam": [
                            pam_distance(window[0], entry.spacer_length, entry.numbering),
                            pam_distance(window[1], entry.spacer_length, entry.numbering),
                        ],
                    }
                )

        built = [item for item in band if item["buildable"]]
        if not built:
            rows.append({"entry_id": entry.entry_id, "band": band, "buildable": False})
            continue
        modes = [item["mode"] for item in built]
        starts = [item["window"][0] for item in built]
        ends = [item["window"][1] for item in built]
        observed = tuple(entry.observed_window)
        rows.append(
            {
                "entry_id": entry.entry_id,
                "editor": entry.editor,
                "effector": entry.effector,
                "structure": entry.structure_id,
                "numbering": entry.numbering,
                "spacer_length": entry.spacer_length,
                "linker_reported": bool(entry.linker_residues),
                "buildable": True,
                "band": band,
                "mode_band": [min(modes), max(modes)],
                "window_band": [min(starts), max(ends)],
                "observed_window": list(observed),
                "observed_from_pam": [
                    pam_distance(observed[0], entry.spacer_length, entry.numbering),
                    pam_distance(observed[1], entry.spacer_length, entry.numbering),
                ],
                "overlap_at_each_setting": {
                    f"{item['site']}-{item['length']}": round(
                        overlap(tuple(item["window"]), observed), 3
                    )
                    for item in built
                },
                "mode_inside_reported_window": {
                    f"{item['site']}-{item['length']}": observed[0] <= item["mode"] <= observed[1]
                    for item in built
                },
            }
        )

    print(f"fitted parameters: {parameters.as_dict()}")
    print(f"attachment sites swept: {list(SITES)}")
    print(f"contour lengths swept: {list(LENGTHS)}")
    print()
    print(
        f"{'entry':32s} {'effector':10s} {'mode band':>10s} {'window band':>13s} "
        f"{'reported':>10s} {'from PAM':>12s}"
    )
    for row in rows:
        if not row["buildable"]:
            print(f"{row['entry_id']:32s} no conformation clears the protein at any length")
            continue
        print(
            f"{row['entry_id']:32s} {row['effector']:10s} "
            f"{'{}-{}'.format(*row['mode_band']):>10s} "
            f"{'{}-{}'.format(*row['window_band']):>13s} "
            f"{'{}-{}'.format(*row['observed_window']):>10s} "
            f"{'{}-{}'.format(*row['observed_from_pam']):>12s}"
        )
    print()
    for row in rows:
        if not row["buildable"]:
            continue
        print(f"{row['entry_id']}")
        for item in row["band"]:
            if not item["buildable"]:
                print(f"  {item['site']:12s} {item['length']:3d} residues  {item['reason'][:44]}")
                continue
            key = f"{item['site']}-{item['length']}"
            print(
                f"  {item['site']:12s} {item['length']:3d} residues  mode {item['mode']:2d}  "
                f"window {item['window'][0]:2d}-{item['window'][1]:2d}  "
                f"{item['from_pam'][0]:2d} to {item['from_pam'][1]:2d} from the PAM  "
                f"overlap {row['overlap_at_each_setting'][key]:.2f}"
            )
    print()
    inside = [
        sum(1 for value in row["mode_inside_reported_window"].values() if value)
        for row in rows
        if row["buildable"]
    ]
    total = [len(row["mode_inside_reported_window"]) for row in rows if row["buildable"]]
    if total:
        print(
            f"predicted mode falls inside the reported window in {sum(inside)} of {sum(total)} "
            "entry and setting combinations"
        )

    report = {
        "ortholog": args.ortholog,
        "threshold": args.threshold,
        "lengths": list(LENGTHS),
        "parameters": parameters.as_dict(),
        "entries": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
