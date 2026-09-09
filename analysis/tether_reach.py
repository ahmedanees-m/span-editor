"""Distance from the fusion junction that a tether can place the active site.

Reports the weighted distribution of active-site positions for each entry, which
is what decides whether a reported window is within reach of its anchor at all.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from span_editor.io import load_assigned_inputs, load_parameters, read_corpus
from span_editor.model import build_context
from span_editor.tether import TetherSampler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument(
        "--entries", nargs="+",
        default=["communbiol-dCas12a-ABE8e", "communbiol-dCas12a-A3A",
                 "communbiol-dCas12a-A3A-Y130F"],
    )
    parser.add_argument(
        "--marks", type=float, nargs="+", default=[35.0, 38.0, 40.0],
        help="distances at which to report the surviving ensemble weight",
    )
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("results/tether_reach.json"))
    args = parser.parse_args(argv)

    assigned = load_assigned_inputs(args.assigned)
    load_parameters(args.params)
    corpus = {e.entry_id: e for e in read_corpus(args.corpus)}
    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=assigned.sampling.get("tether_chains", 20000),
        seed=assigned.sampling.get("seed", 0),
        target_surviving=assigned.sampling.get("target_surviving", 0),
    )

    rows = []
    for entry_id in args.entries:
        entry = corpus.get(entry_id)
        if entry is None:
            print(f"{entry_id} is not in the corpus")
            continue
        context = build_context(entry, assigned, args.structures)
        field = sampler.sample(
            linker=entry.linker,
            effector=context.effector,
            anchor=context.anchor,
            occupancy=context.occupancy,
            structure_id=entry.structure_id or "none",
            closure=context.closure,
        )
        reach = np.linalg.norm(field.positions, axis=1)
        weight = field.weights / field.weights.sum()
        order = np.argsort(reach)
        cumulative = np.cumsum(weight[order])

        def quantile(q: float) -> float:
            return float(reach[order][np.searchsorted(cumulative, q)])

        composition = assigned.compositions[entry.linker_composition]
        counts, edges = np.histogram(reach, bins=args.bins, weights=weight)
        rows.append(
            {
                "entry_id": entry_id,
                "effector": entry.effector,
                "linker_composition": entry.linker_composition,
                "linker_residues": entry.linker_residues,
                "contour_length": round(entry.linker_residues * composition.rise, 1),
                "active_site_offset": round(
                    float(np.linalg.norm(context.effector.active_site_offset)), 1
                ),
                "sampled": int(field.sampled),
                "surviving": int(field.surviving),
                "mean": round(float(np.sum(weight * reach)), 2),
                "rms": round(float(np.sqrt(np.sum(weight * reach**2))), 2),
                "median": round(quantile(0.50), 2),
                "p90": round(quantile(0.90), 2),
                "max": round(float(reach.max()), 2),
                "weight_beyond": {
                    str(m): round(float(weight[reach >= m].sum()), 4) for m in args.marks
                },
                "histogram": {
                    "edges": [round(float(x), 2) for x in edges],
                    "weight": [round(float(x), 6) for x in counts],
                },
            }
        )
        print(
            f"{entry_id:32s} contour {rows[-1]['contour_length']:5.1f} A  "
            f"offset {rows[-1]['active_site_offset']:5.1f} A  "
            f"mean {rows[-1]['mean']:5.1f}  rms {rows[-1]['rms']:5.1f}  "
            f"beyond 40 A {rows[-1]['weight_beyond']['40.0']:.3f}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"marks": args.marks, "entries": rows}, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
