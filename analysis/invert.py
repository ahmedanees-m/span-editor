"""Enumerate candidate tethers and report which positions their modes cover.

The grid spans attachment site, linker composition and contour length. Candidates
are single-ended fusions on the intact scaffold. Also runs the inverse problem,
asking which contour lengths are consistent with a measured window, scored
against a modal-length baseline.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np
import yaml

from span_editor.io import load_assigned_inputs, load_parameters, read_corpus
from span_editor.model import SpanModel, build_context
from span_editor.tether import TetherSampler

LENGTHS = (4, 8, 12, 16, 24, 32, 48)
COMPOSITIONS = ("xten", "polyproline", "helical")
THRESHOLD = 0.5


def sites_for(assigned, accession: str) -> list[str]:
    settings = assigned.structure_geometry.get(accession, {})
    return sorted((settings.get("anchor_sites") or {}))


def template(entry, site: str):
    """The entry as a terminal fusion at one attachment site.

    A search over where to attach asks for one connection, so an entry that
    carries two is reduced to the first. Insertions are therefore proposed as
    single-ended fusions and would need a second specification before they could
    be built.
    """
    return replace(
        entry,
        entry_id=f"{entry.entry_id}@{site}",
        anchor_site=site,
        return_site="",
        return_residues=0,
        effector=entry.effector.replace("_inserted", ""),
    )


def scan_site(
    site: str,
    reference,
    assigned,
    parameters,
    structures: Path,
    target_surviving: int,
    threshold: float,
    seed: int | None = None,
) -> list[dict]:
    """Every composition and contour length at one attachment site.

    One worker holds one site, so the occupancy map and the substrate ensemble
    are built once and shared across the whole scan at that site.
    """
    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=assigned.sampling.get("tether_chains", 20000),
        seed=assigned.sampling.get("seed", 0) if seed is None else seed,
        target_surviving=target_surviving,
    )
    model = SpanModel(sampler)
    try:
        context = build_context(template(reference, site), assigned, structures)
    except (ValueError, FileNotFoundError) as error:
        return [{"site": site, "buildable": False, "reason": str(error), "composition": "", "length": 0}]

    rows = []
    for composition in COMPOSITIONS:
        for length in LENGTHS:
            trial = replace(
                context,
                entry=replace(
                    context.entry,
                    linker_residues=length,
                    linker_composition=composition,
                ),
                _ensembles=context._ensembles,
            )
            try:
                profile = model.predict(trial, parameters).normalised()
            except RuntimeError as error:
                rows.append(
                    {
                        "site": site,
                        "composition": composition,
                        "length": length,
                        "buildable": False,
                        "reason": str(error),
                    }
                )
                continue
            rows.append(
                {
                    "site": site,
                    "composition": composition,
                    "length": length,
                    "buildable": True,
                    "mode": int(profile.mode),
                    "window": list(profile.window(threshold)),
                    "width": int(profile.width(threshold)),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--reference", default="hnhx-ABEmax7.10")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--target-surviving", type=int, default=3000)
    parser.add_argument("--tolerance", type=int, default=8, help="residues, for the recovery check")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument(
        "--seed", type=int, default=None,
        help="override the configured sampling seed",
    )
    parser.add_argument(
        "--mode-tolerance",
        type=int,
        default=1,
        help="nucleotides; how close a candidate window has to sit to count as consistent",
    )
    parser.add_argument(
        "--width-tolerance",
        type=int,
        default=1,
        help="nucleotides; how close a candidate window has to be in width",
    )
    parser.add_argument(
        "--reuse",
        type=Path,
        default=None,
        help="read the grid from an earlier run instead of computing it again",
    )
    parser.add_argument("--out", type=Path, default=Path("results/inversion.json"))
    args = parser.parse_args(argv)

    fitted = args.params.with_name("fitted.yaml")
    document = yaml.safe_load((fitted if fitted.exists() else args.params).read_text("utf-8"))
    if document.get("status") != "fitted":
        print("no fitted parameters found; run the fit first")
        return 1

    assigned = load_assigned_inputs(args.assigned)
    parameters = load_parameters(args.params)
    corpus = read_corpus(args.corpus)
    reference = next((e for e in corpus if e.entry_id == args.reference), None)
    if reference is None:
        print(f"{args.reference} is not in the corpus")
        return 1

    sites = sites_for(assigned, reference.structure_id)
    print(f"attachment sites on {reference.structure_id}: {', '.join(sites)}")
    print(f"contour lengths: {list(LENGTHS)}")
    print(f"composition classes: {list(COMPOSITIONS)}")
    print()

    if args.reuse is not None:
        grid = json.loads(args.reuse.read_text("utf-8"))["grid"]
        print(f"grid read from {args.reuse}, {len(grid)} combinations")
    else:
        scan = partial(
            scan_site,
            reference=reference,
            assigned=assigned,
            parameters=parameters,
            structures=args.structures,
            target_surviving=args.target_surviving,
            threshold=args.threshold,
            seed=args.seed,
        )
        grid = []
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for rows in pool.map(scan, sites):
                for row in rows:
                    grid.append(row)
                    if row["buildable"]:
                        print(
                            f"  {row['site']:12s} {row['composition']:12s} {row['length']:3d}  "
                            f"mode {row['mode']:2d}  window {tuple(row['window'])}"
                        )
                    else:
                        print(
                            f"  {row['site']:12s} {row['composition']:12s} {row['length']:3d}  "
                            "no conformation clears the protein"
                        )

    built = [row for row in grid if row["buildable"]]
    flexible = [row for row in built if not assigned.compositions[row["composition"]].rigid]

    reached = defaultdict(list)
    for row in built:
        reached[row["mode"]].append(f"{row['site']}/{row['composition']}-{row['length']}")
    flexible_modes = {row["mode"] for row in flexible}

    indices = list(range(1, reference.spacer_length + 1))
    unreachable_flexible = [index for index in indices if index not in flexible_modes]
    unreachable_any = [index for index in indices if index not in reached]

    # Recovery. Every corpus entry that names a contour length and a window is
    # asked for its own length back, given only the window and its own
    # attachment site.
    #
    # The answer is a set, not a number. Window mode moves slowly with contour
    # length at most attachment sites, so many lengths are consistent with the
    # same target and the inverse problem is not identified by position alone.
    # Reporting a single proposal would hide that behind whichever tie-break was
    # chosen, so both are reported: whether the published length is among the
    # lengths consistent with the target, and how many lengths that leaves.
    #
    # The comparison is against always proposing the commonest published length,
    # which is strong here because published linkers cluster.
    lengths_in_corpus = Counter(entry.linker_residues for entry in corpus)
    modal_length = lengths_in_corpus.most_common(1)[0][0]

    recovery = []
    for entry in corpus:
        if entry.observed is None and entry.observed_window is None:
            continue
        if entry.is_insertion:
            continue
        target = (
            entry.observed.mode
            if entry.observed is not None
            else int(round(0.5 * sum(entry.observed_window)))
        )
        matching = [
            row
            for row in built
            if row["site"] == entry.anchor_site and row["composition"] == entry.linker_composition
        ]
        if not matching:
            continue
        if entry.observed is not None:
            observed_width = entry.observed.width()
        elif entry.observed_width:
            observed_width = entry.observed_width
        elif entry.observed_window:
            observed_width = entry.observed_window[1] - entry.observed_window[0] + 1
        else:
            observed_width = 0
        by_mode = [row for row in matching if abs(row["mode"] - target) <= args.mode_tolerance]
        consistent = sorted(row["length"] for row in by_mode)
        if observed_width:
            consistent_with_width = sorted(
                row["length"]
                for row in by_mode
                if abs(row["width"] - observed_width) <= args.width_tolerance
            )
        else:
            consistent_with_width = list(consistent)
        nearest = min(matching, key=lambda row: (abs(row["mode"] - target), row["length"]))
        recovery.append(
            {
                "entry_id": entry.entry_id,
                "site": entry.anchor_site,
                "composition": entry.linker_composition,
                "published_length": entry.linker_residues,
                "target_mode": target,
                "observed_width": observed_width or None,
                "consistent_lengths": consistent,
                "consistent_lengths_with_width": consistent_with_width,
                "lengths_on_the_grid": len(matching),
                "published_is_consistent": entry.linker_residues in consistent,
                "published_is_consistent_with_width": entry.linker_residues
                in consistent_with_width,
                "nearest_length": nearest["length"],
                "nearest_mode": nearest["mode"],
                "error": abs(nearest["length"] - entry.linker_residues),
                "modal_error": abs(modal_length - entry.linker_residues),
            }
        )

    within = sum(1 for row in recovery if row["error"] <= args.tolerance)
    modal_within = sum(1 for row in recovery if row["modal_error"] <= args.tolerance)
    consistent_count = sum(1 for row in recovery if row["published_is_consistent"])
    identified = sum(
        1 for row in recovery if 0 < len(row["consistent_lengths"]) < row["lengths_on_the_grid"]
    )
    with_width = sum(1 for row in recovery if row["published_is_consistent_with_width"])
    identified_with_width = sum(
        1
        for row in recovery
        if 0 < len(row["consistent_lengths_with_width"]) < row["lengths_on_the_grid"]
    )

    print()
    print(f"buildable combinations: {len(built)} of {len(grid)}")
    print(f"positions reachable by any candidate: {sorted(reached)}")
    print(f"positions no flexible candidate reaches: {unreachable_flexible}")
    print(f"positions no candidate reaches at all: {unreachable_any}")
    print()
    print("mode  candidates that put the peak there")
    for mode in sorted(reached):
        listed = reached[mode]
        print(f"{mode:4d}  {len(listed):3d}  {', '.join(listed[:5])}{' ...' if len(listed) > 5 else ''}")
    print()
    print(f"recovery over {len(recovery)} entries")
    print(
        f"{'entry':32s} {'built':>6s} {'by position':>26s} {'and width':>22s} {'in set':>7s}"
    )
    for row in recovery:
        listed = ",".join(str(value) for value in row["consistent_lengths"]) or "none"
        both = ",".join(str(value) for value in row["consistent_lengths_with_width"]) or "none"
        print(
            f"{row['entry_id']:32s} {row['published_length']:6d} {listed:>26s} {both:>22s} "
            f"{'yes' if row['published_is_consistent_with_width'] else 'no':>7s}"
        )
    print()
    print(f"grid holds up to {max(row['lengths_on_the_grid'] for row in recovery)} lengths per entry")
    print(
        f"position alone narrows the set below the whole grid in {identified} of {len(recovery)}, "
        f"and the published length is in it in {consistent_count}"
    )
    print(
        f"position and width narrow it in {identified_with_width} of {len(recovery)}, "
        f"and the published length is in it in {with_width}"
    )
    print(f"nearest single proposal within {args.tolerance} residues: {within} of {len(recovery)}")
    print(
        f"always proposing {modal_length} residues, within {args.tolerance}: "
        f"{modal_within} of {len(recovery)}"
    )

    report = {
        "threshold": args.threshold,
        "target_surviving": args.target_surviving,
        "seed": args.seed,
        "tolerance": args.tolerance,
        "parameters": parameters.as_dict(),
        "reference_entry": args.reference,
        "sites": sites,
        "lengths": list(LENGTHS),
        "compositions": list(COMPOSITIONS),
        "grid": grid,
        "reachable": {str(mode): reached[mode] for mode in sorted(reached)},
        "unreachable_by_flexible": unreachable_flexible,
        "unreachable_by_any": unreachable_any,
        "modal_length": modal_length,
        "recovery": recovery,
        "within_tolerance": within,
        "modal_within_tolerance": modal_within,
        "published_among_consistent": consistent_count,
        "consistent_set_narrower_than_grid": identified,
        "published_among_consistent_with_width": with_width,
        "consistent_set_narrower_with_width": identified_with_width,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
