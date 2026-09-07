"""Signed mode offsets by ortholog family.

The same comparison is also expressed as distance from the fusion junction,
which removes the opposite protospacer numbering conventions of the two
families. Distances come from the deposition each entry is modelled on where it
resolves the relevant positions, and from 6VPC otherwise; the source is recorded
per entry.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from span_editor.io import load_assigned_inputs, read_corpus
from span_editor.metrics import cluster_bootstrap
from span_editor.model import build_context
from span_editor.structures import load_structure

CAS12A = "Cas12a"
RESTING = "resting deposition"
ENGAGED = "engaged complex"

# 6VPC author numbering runs 197 ahead of canonical SpCas9, and chain B is
# modelled from author 201, which is canonical residue 4.
VPC_OFFSET = 197
VPC_FIRST = 201


def family(ortholog: str) -> str:
    return CAS12A if ortholog == CAS12A else "SpCas9 family"


def report(label: str, values: list[float], clusters: list[str], unit: str) -> dict:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {}
    groups = sorted(set(clusters))
    record = {
        "label": label,
        "n": int(array.size),
        "publications": len(groups),
        "mean": round(float(array.mean()), 3),
    }
    if len(groups) < 2:
        # A cluster bootstrap over one cluster resamples nothing and returns the
        # point estimate as its own interval. Reporting that as an interval
        # would be an error, so it is withheld.
        print(
            f"  {label:18s} n={array.size:2d}  mean {array.mean():+6.2f} {unit}  "
            f"one publication, no interval"
        )
        return record
    interval = cluster_bootstrap(array, np.asarray(clusters))
    print(
        f"  {label:18s} n={array.size:2d}  mean {interval.estimate:+6.2f} {unit}  "
        f"[{interval.lower:+6.2f}, {interval.upper:+6.2f}] over {len(groups)} publications  "
        f"{'excludes zero' if interval.excludes_zero else 'includes zero'}"
    )
    record.update(
        lower=round(float(interval.lower), 3),
        upper=round(float(interval.upper), 3),
        excludes_zero=bool(interval.excludes_zero),
    )
    return record


def engaged_distances(structures: Path, mapping_file: Path) -> tuple[dict, dict]:
    """Junction to nucleotide distances in the deposited engaged complex.

    Returns the anchor coordinates by canonical SpCas9 residue and the C1'
    coordinate of each protospacer index that 6VPC resolves.
    """
    structure = load_structure(structures / "6VPC.cif.gz")
    mapping = json.loads(mapping_file.read_text(encoding="utf-8"))["mapping"]
    residue_to_index = {int(k): v for k, v in mapping["strand_residue_to_index"].items()}

    protein = structure[(structure.chain_id == "B") & (structure.atom_name == "CA")]
    anchors = {
        int(protein.res_id[k]) - VPC_OFFSET: protein.coord[k]
        for k in range(protein.array_length())
    }
    strand = structure[(structure.chain_id == "D") & (structure.atom_name == "C1'")]
    positions = {
        residue_to_index[int(strand.res_id[k])]: strand.coord[k]
        for k in range(strand.array_length())
        if int(strand.res_id[k]) in residue_to_index
    }
    return anchors, positions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--mapping", type=Path, default=Path("results/map_6VPC.json"))
    parser.add_argument(
        "--leave-one-out", type=Path, default=Path("results/leave_one_out.json")
    )
    parser.add_argument(
        "--seed-modes", type=Path, default=None,
        help="results/bias_seed_stability.json; when given, the distance arm is "
             "recomputed for every seed in it instead of for the single "
             "leave-one-out run, which is what says whether the angstrom "
             "figure is independent of the modes it re-expresses",
    )
    parser.add_argument("--out", type=Path, default=Path("results/directional_bias.json"))
    args = parser.parse_args(argv)

    if not args.leave_one_out.exists():
        print(f"{args.leave_one_out} not found; run analysis/leave_one_out.py first")
        return 1

    withheld = json.loads(args.leave_one_out.read_text(encoding="utf-8"))["entries"]
    by_id = {entry.entry_id: entry for entry in read_corpus(args.corpus)}
    assigned = load_assigned_inputs(args.assigned)
    anchor_sites = assigned.structure_geometry["5F9R"]["anchor_sites"]

    engaged_anchors, engaged_positions = {}, {}
    if (args.structures / "6VPC.cif.gz").exists() and args.mapping.exists():
        engaged_anchors, engaged_positions = engaged_distances(args.structures, args.mapping)
        print(
            f"engaged complex 6VPC resolves protospacer indices "
            f"{sorted(engaged_positions)}"
        )
        print()

    def measure(entry, predicted: int, observed: int) -> dict:
        """One entry's signed offset and, where coordinates allow, its distance."""
        row = {
            "entry_id": entry.entry_id,
            "source_key": entry.source_key,
            "ortholog": entry.ortholog,
            "family": family(entry.ortholog),
            "anchor_site": entry.anchor_site,
            "effector": entry.effector,
            "observed_mode": observed,
            "predicted_mode": predicted,
            "signed_offset": predicted - observed,
        }

        # First choice is the deposition the entry is actually modelled on, where
        # no effector is bound to the strand.
        try:
            context = build_context(entry, assigned, args.structures)
        except (ValueError, FileNotFoundError) as error:
            print(f"skipped {entry.entry_id}: {error}")
            return None
        coordinates = {anchor.index: anchor.position for anchor in context.strand_anchors}
        coordinates.update(context.observed_bubble)
        source = RESTING

        # 5F9R leaves the SpCas9 window unresolved, so those entries fall back to
        # the engaged complex, with its own junction for the same anchor site.
        if not {predicted, observed} <= set(coordinates):
            canonical = anchor_sites.get(row["anchor_site"])
            junction = None
            if canonical is not None and engaged_positions:
                junction = engaged_anchors.get(max(int(canonical), VPC_FIRST - VPC_OFFSET))
            if junction is None:
                return row
            coordinates = {
                index: position - junction for index, position in engaged_positions.items()
            }
            source = ENGAGED

        if not {predicted, observed} <= set(coordinates):
            return row
        to_predicted = float(np.linalg.norm(coordinates[predicted]))
        to_observed = float(np.linalg.norm(coordinates[observed]))
        # Rank of each mode among the positions carrying a coordinate, as a
        # control on the model simply naming the nearest position.
        ordered = sorted(float(np.linalg.norm(v)) for v in coordinates.values())
        row.update(
            coordinate_source=source,
            reach_to_predicted=round(to_predicted, 2),
            reach_to_observed=round(to_observed, 2),
            further_out=round(to_observed - to_predicted, 2),
            positions_with_a_coordinate=len(ordered),
            rank_of_predicted=ordered.index(to_predicted) + 1,
            rank_of_observed=ordered.index(to_observed) + 1,
        )
        return row

    # Recompute the distance arm from a stored per-seed set of modes.
    if args.seed_modes is not None:
        stability = json.loads(args.seed_modes.read_text(encoding="utf-8"))
        per_seed = {}
        print("distance arm recomputed per seed, from the modes in "
              f"{args.seed_modes.name}\n")
        for seed, entries in stability["entries"].items():
            rows = []
            for record in entries:
                entry = by_id.get(record["entry_id"])
                if entry is None or record.get("degenerate") or record.get("mode") is None:
                    continue
                got = measure(entry, int(record["mode"]), int(record["observed_mode"]))
                if got is not None:
                    rows.append(got)
            reachable = [r for r in rows if "further_out" in r]
            print(f"seed {seed}")
            per_seed[seed] = {}
            for label in ("all entries", "SpCas9 family", CAS12A):
                chosen = [r for r in reachable
                          if label == "all entries" or r["family"] == label]
                rec = report(label, [r["further_out"] for r in chosen],
                             [r["source_key"] for r in chosen], "A ")
                if rec:
                    per_seed[seed][label] = rec
            print()
        print()
        for label in ("all entries", "SpCas9 family"):
            means = [per_seed[s][label]["mean"] for s in per_seed if label in per_seed[s]]
            excl = [per_seed[s][label].get("excludes_zero")
                    for s in per_seed if label in per_seed[s]]
            if means:
                print(f"{label}: spans {min(means):+.3f} to {max(means):+.3f} angstrom "
                      f"(range {max(means) - min(means):.3f}); excludes zero at "
                      f"{sum(1 for e in excl if e)} of {len(excl)} seeds")
        out = args.out.with_name("distance_seed_stability.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            json.dump({"seeds": list(stability["entries"]), "summary": per_seed},
                      handle, indent=2)
            handle.write("\n")
        print(f"\nwritten to {out}")
        return 0

    rows = []
    for record in withheld:
        entry = by_id.get(record["entry_id"])
        if entry is None or record.get("loose_degenerate"):
            continue
        got = measure(entry, int(record["loose_mode"]), int(record["observed_mode"]))
        if got is not None:
            rows.append(got)

    print(f"{len(rows)} entries with a window under leave-one-out scoring")
    groups = defaultdict(list)
    for row in rows:
        groups["all entries"].append(row)
        groups[row["family"]].append(row)

    print()
    print("signed mode offset, predicted minus observed, in nucleotides.")
    print("Sign is not comparable across families: Cas9 is numbered from the")
    print("PAM-distal end and Cas12a from the PAM-proximal end.")
    print()
    summary = {"signed_offset": [], "further_out": []}
    for label in ("all entries", "SpCas9 family", CAS12A):
        chosen = groups.get(label, [])
        record = report(
            label,
            [row["signed_offset"] for row in chosen],
            [row["source_key"] for row in chosen],
            "nt",
        )
        if record:
            summary["signed_offset"].append(record)

    print()
    print("how much further from the fusion junction the measured mode sits than the")
    print("predicted one, in angstrom; positive means the measurement is further out")
    print()
    for label in ("all entries", "SpCas9 family", CAS12A):
        chosen = [row for row in groups.get(label, []) if "further_out" in row]
        record = report(
            label,
            [row["further_out"] for row in chosen],
            [row["source_key"] for row in chosen],
            "A ",
        )
        if record:
            summary["further_out"].append(record)

    reachable = [row for row in rows if "further_out" in row]
    if reachable:
        further = sum(1 for row in reachable if row["further_out"] > 0)
        level = sum(1 for row in reachable if row["further_out"] == 0)
        print()
        print(
            f"  measurement further from the junction than the prediction in "
            f"{further} of {len(reachable)} entries, level in {level}, "
            f"closer in {len(reachable) - further - level}"
        )
        summary["further_out_count"] = [further, level, len(reachable)]
        by_source = defaultdict(list)
        for row in reachable:
            by_source[row["coordinate_source"]].append(row["further_out"])
        for source, values in sorted(by_source.items()):
            print(f"  from the {source:18s} n={len(values):2d}  mean {np.mean(values):+5.2f} A")

        # The control. A model that always named the nearest position would put
        # every predicted rank at one.
        ranks_predicted = [row["rank_of_predicted"] for row in reachable]
        ranks_observed = [row["rank_of_observed"] for row in reachable]
        total = [row["positions_with_a_coordinate"] for row in reachable]
        print()
        print(
            f"  rank of the predicted mode by distance from the junction: "
            f"mean {np.mean(ranks_predicted):.1f} of {np.mean(total):.0f}, "
            f"nearest position named in {sum(1 for r in ranks_predicted if r == 1)} "
            f"of {len(ranks_predicted)} entries"
        )
        print(
            f"  rank of the measured mode:  mean {np.mean(ranks_observed):.1f} "
            f"of {np.mean(total):.0f}"
        )
        summary["rank"] = {
            "predicted_mean": round(float(np.mean(ranks_predicted)), 2),
            "observed_mean": round(float(np.mean(ranks_observed)), 2),
            "positions_mean": round(float(np.mean(total)), 2),
            "predicted_at_nearest": int(sum(1 for r in ranks_predicted if r == 1)),
        }

    print()
    print(f"{'entry':38s} {'obs':>4s} {'pred':>5s} {'nt':>4s} "
          f"{'to obs':>8s} {'to pred':>8s} {'further':>8s}  coordinates")
    for row in sorted(rows, key=lambda item: (item["family"], item["entry_id"])):
        if "further_out" not in row:
            print(f"{row['entry_id']:38s} {row['observed_mode']:4d} "
                  f"{row['predicted_mode']:5d} {row['signed_offset']:+4d}"
                  f"{'':28s}  no coordinate at either mode")
            continue
        print(
            f"{row['entry_id']:38s} {row['observed_mode']:4d} {row['predicted_mode']:5d} "
            f"{row['signed_offset']:+4d} {row['reach_to_observed']:8.1f} "
            f"{row['reach_to_predicted']:8.1f} {row['further_out']:+8.1f}  "
            f"{row['coordinate_source']}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"entries": rows, "summary": summary}, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
