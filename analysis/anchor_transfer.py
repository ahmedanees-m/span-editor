"""Score architectures whose window is reported as a span rather than a profile.

Predictions are compared as shifts relative to a reference architecture from the
same publication, which removes the per-source reporting scale.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from span_editor.io import load_assigned_inputs, load_parameters, read_corpus
from span_editor.model import build_context
from span_editor.occupancy import (
    Profile,
    capture_probability,
    saturating_link,
    saturation_index,
)
from span_editor.tether import TetherSampler

THRESHOLD = 0.5
JITTER = (0.0, 5.0, 10.0)


def overlap(first: tuple[int, int], second: tuple[int, int]) -> float:
    """Fraction of the union of two integer spans that both cover."""
    low = max(first[0], second[0])
    high = min(first[1], second[1])
    shared = max(0, high - low + 1)
    union = max(first[1], second[1]) - min(first[0], second[0]) + 1
    return shared / union if union > 0 else 0.0


def centre(span) -> float:
    return 0.5 * (span[0] + span[1])


def span_text(window) -> str:
    return "{}-{}".format(*window) if window else "-"


def score_entry(entry, assigned, sampler, parameters, structures, threshold, radius):
    context = build_context(entry, assigned, structures)
    tether = sampler.sample(
        linker=entry.linker,
        effector=context.effector,
        anchor=context.anchor,
        occupancy=context.occupancy,
        structure_id=entry.structure_id or "none",
        closure=context.closure,
    )
    obstacle = None
    if context.closure is not None and tether.body_centre is not None:
        obstacle = (tether.body_centre, tether.body_radius)

    held = sorted(anchor.index for anchor in context.strand_anchors)
    substrate = context.substrate(parameters.ssdna_persistence, obstacle)
    indices, capture = capture_probability(
        tether, substrate, parameters.capture_radius, context.indices
    )
    _, at_reference = capture_probability(tether, substrate, radius, context.indices)

    indices = np.asarray(indices)
    full = Profile(indices=indices, values=capture).normalised()
    free = ~np.isin(indices, held)
    sampled = Profile(indices=indices[free], values=capture[free]).normalised()
    linked = Profile(
        indices=indices, values=saturating_link(capture, parameters.link_alpha)
    ).normalised()

    # Restricting to the sampled positions removes the advantage a resolved
    # nucleotide gets from carrying no conformational freedom. It only works
    # while enough of the strand is sampled to hold a window. Where a deposition
    # resolves nearly all of it, as 6I1K does, there is no unbiased subset to
    # fall back on and the comparison has to be made on all positions with the
    # bias left in and said out loud.
    if entry.observed_window:
        wanted = range(entry.observed_window[0], entry.observed_window[1] + 1)
        covered = sum(1 for index in wanted if index not in set(held)) / len(list(wanted))
        usable = covered >= 0.5
    else:
        covered = float(np.sum(free)) / indices.shape[0]
        usable = bool(np.any(free)) and float(np.max(capture[free])) > 0
    scored = sampled if usable else full
    window_source = "sampled positions" if usable else "all positions, bias unmitigated"

    sweep = {}
    for level in JITTER:
        if level == 0.0:
            sweep[str(level)] = list(full.window(threshold))
            continue
        jittered = context.substrate(parameters.ssdna_persistence, obstacle, level)
        _, values = capture_probability(
            tether, jittered, parameters.capture_radius, context.indices
        )
        sweep[str(level)] = list(Profile(indices=indices, values=values).window(threshold))

    blocked = None
    if obstacle is not None:
        inside = 0.0
        for index in context.indices:
            cloud, weights = substrate.cloud(index)
            distance = np.linalg.norm(cloud - obstacle[0], axis=1)
            inside += float(np.sum(weights[distance <= obstacle[1]]))
        blocked = round(inside / len(context.indices), 5)

    return {
        "entry_id": entry.entry_id,
        "source_key": entry.source_key,
        "editor": entry.editor,
        "anchor_site": entry.anchor_site,
        "linker_residues": entry.linker_residues,
        "return_site": entry.return_site,
        "return_residues": entry.return_residues,
        "scaffold_variant": entry.scaffold_variant,
        "label": entry.label,
        "readout_class": entry.readout_class,
        "occluding_spheres": len(context.occupancy) if context.occupancy else 0,
        "chains_sampled": tether.sampled,
        "chains_surviving": tether.surviving,
        "effective_conformations": round(tether.effective_size, 1),
        "held_positions": held,
        "substrate_weight_inside_the_inserted_domain": blocked,
        "peak_capture": round(float(np.max(capture)), 6),
        "saturation_index": round(
            saturation_index(parameters.link_alpha, float(np.max(at_reference))), 4
        ),
        "mode": int(scored.mode),
        "window": list(scored.window(threshold)),
        "width": int(scored.width(threshold)),
        "window_source": window_source,
        "positions_sampled": int(np.sum(free)),
        "reported_window_sampled": round(float(covered), 3),
        "window_over_all_positions": list(full.window(threshold)),
        "window_after_link": list(linked.window(threshold)),
        "jitter_sweep": sweep,
        "observed_window": list(entry.observed_window) if entry.observed_window else None,
        "observed_width": entry.observed_width or None,
        "capture_profile": {
            int(index): round(float(value), 4) for index, value in zip(full.indices, full.values)
        },
        "sampled_profile": {
            int(index): round(float(value), 4)
            for index, value in zip(scored.indices, scored.values)
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument(
        "--max-chains",
        type=int,
        default=0,
        help="sampling budget; raising it lowers Monte Carlo noise and changes nothing else",
    )
    parser.add_argument("--out", type=Path, default=Path("results/anchor_transfer.json"))
    args = parser.parse_args(argv)

    fitted = args.params.with_name("fitted.yaml")
    document = yaml.safe_load((fitted if fitted.exists() else args.params).read_text("utf-8"))
    if document.get("status") != "fitted":
        print("no fitted parameters found; run the fit first")
        return 1
    settings = yaml.safe_load(args.params.read_text("utf-8"))
    radius = float(settings.get("link", {}).get("reference_radius", 14.0))

    assigned = load_assigned_inputs(args.assigned)
    parameters = load_parameters(args.params)
    entries = [
        entry
        for entry in read_corpus(args.corpus)
        if entry.observed is None and (entry.observed_window or entry.observed_width)
    ]
    # An entry with no reported window of its own still has to be scored where it
    # is the reference its source compares everything else against.
    wanted = {entry.source_key for entry in entries}
    entries = [
        entry
        for entry in read_corpus(args.corpus)
        if entry.observed is None and entry.source_key in wanted
    ]
    if not entries:
        print("no entries report a window or a width")
        return 1

    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=assigned.sampling.get("tether_chains", 20000),
        seed=assigned.sampling.get("seed", 0),
        target_surviving=assigned.sampling.get("target_surviving", 0),
        max_chains=args.max_chains,
    )

    rows, skipped = [], []
    for entry in entries:
        try:
            rows.append(
                score_entry(
                    entry,
                    assigned,
                    sampler,
                    parameters,
                    args.structures,
                    args.threshold,
                    radius,
                )
            )
        except (ValueError, KeyError, FileNotFoundError, RuntimeError) as error:
            skipped.append({"entry_id": entry.entry_id, "reason": str(error)})

    for row in rows:
        if row["observed_window"]:
            row["window_overlap"] = round(
                overlap(tuple(row["window"]), tuple(row["observed_window"])), 3
            )
            row["centre_offset"] = round(
                centre(row["window"]) - centre(row["observed_window"]), 2
            )
        if row["observed_width"]:
            row["width_error"] = row["width"] - row["observed_width"]

    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source_key"]].append(row)

    shifts = []
    for source, group in by_source.items():
        reference = next((row for row in group if row["label"] == "none"), None)
        if reference is None or not reference["observed_window"]:
            continue
        for row in group:
            if row is reference or not row["observed_window"]:
                continue
            shifts.append(
                {
                    "source_key": source,
                    "reference": reference["entry_id"],
                    "against": row["entry_id"],
                    "predicted_shift": round(
                        centre(row["window"]) - centre(reference["window"]), 2
                    ),
                    "observed_shift": round(
                        centre(row["observed_window"]) - centre(reference["observed_window"]), 2
                    ),
                    "predicted_mode_shift": row["mode"] - reference["mode"],
                }
            )

    print(f"fitted parameters: {parameters.as_dict()}")
    print(f"window threshold: {args.threshold} of the peak, on capture")
    unmitigated = [row for row in rows if row["window_source"] != "sampled positions"]
    if unmitigated:
        print(
            f"{len(unmitigated)} entries have too little of the strand sampled to restrict to it, "
            "and are scored on all positions with the resolved-nucleotide bias left in:"
        )
        for row in unmitigated:
            print(
                f"  {row['entry_id']:32s} {row['positions_sampled']} of "
                f"{len(row['capture_profile'])} positions sampled, covering "
                f"{row['reported_window_sampled']:.0%} of the reported window"
            )
    print()
    print(
        f"{'entry':24s} {'anchor':12s} {'link':>5s} {'return':>7s} {'effective':>10s} "
        f"{'mode':>5s} {'window':>8s} {'all':>8s} {'reported':>9s} {'width':>6s} {'obs':>4s}"
    )
    for source, group in by_source.items():
        print(f"  {source}")
        for row in group:
            print(
                f"{row['entry_id']:24s} {row['anchor_site']:12s} "
                f"{row['linker_residues']:5d} {row['return_residues'] or 0:7d} "
                f"{row['effective_conformations']:10.0f} {row['mode']:5d} "
                f"{span_text(row['window']):>8s} {span_text(row['window_over_all_positions']):>8s} "
                f"{span_text(row['observed_window']):>9s} {row['width']:6d} "
                f"{row['observed_width'] or 0:4d}"
            )
    print()
    for row in rows:
        print(
            f"{row['entry_id']:24s} peak capture {row['peak_capture']:.5f}, "
            f"saturation index {row['saturation_index']:.3f}, "
            f"window after the link {span_text(row['window_after_link'])}"
        )
    print()
    print("window over all positions as the resolved ones are given a spread")
    print(f"{'entry':24s}" + "".join(f"{level:>13.0f} A" for level in JITTER))
    for row in rows:
        print(
            f"{row['entry_id']:24s}"
            + "".join(f"{span_text(row['jitter_sweep'][str(level)]):>15s}" for level in JITTER)
        )
    print()
    for row in rows:
        parts = []
        if "window_overlap" in row:
            parts.append(
                f"overlap {row['window_overlap']:.2f}, centre offset {row['centre_offset']:+.2f}"
            )
        if "width_error" in row:
            parts.append(f"width off by {row['width_error']:+d}")
        if parts:
            print(f"{row['entry_id']:24s} " + "; ".join(parts))
    print()
    for shift in shifts:
        print(
            f"{shift['reference']} to {shift['against']}: predicted shift "
            f"{shift['predicted_shift']:+.2f}, reported {shift['observed_shift']:+.2f}, "
            f"mode moves {shift['predicted_mode_shift']:+d}"
        )
    if skipped:
        print()
        for item in skipped:
            print(f"skipped {item['entry_id']}: {item['reason']}")

    report = {
        "threshold": args.threshold,
        "reference_radius": radius,
        "parameters": parameters.as_dict(),
        "entries": rows,
        "shifts": shifts,
        "skipped": skipped,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
