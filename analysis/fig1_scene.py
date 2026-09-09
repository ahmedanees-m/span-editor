"""Emit the structural scene Fig 1A draws, in the anchor frame.

Panel A is the deposited complex rather than a cartoon: the editor, the guide,
both DNA strands and the deaminase are read from the structures the model uses,
and the linker is one conformation from the sampled ensemble. None of it is
drawn by hand, so this writes the coordinates once for the figure script.

Run from the repository root:  python analysis/fig1_scene.py
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from span_editor.io import load_assigned_inputs, load_parameters, read_corpus
from span_editor.model import build_context
from span_editor.polymer import sample_free_chain
from span_editor.structures import load_structure
from span_editor.tether import TetherSampler

ENTRY = "kissling-SpCas9-ABE8e-HEK-Plasmid-5d"
BACKBONE = {"P", "C4'", "CA"}


def chain_trace(structure, chain, atom_names):
    mask = (structure.chain_id == chain) & np.isin(structure.atom_name, list(atom_names))
    sub = structure[mask]
    order = np.argsort(sub.res_id)
    return np.asarray(sub.coord[order], dtype=float), np.asarray(sub.res_id[order])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--entry", default=ENTRY)
    parser.add_argument("--out", type=Path, default=Path("results/fig1_scene.json"))
    args = parser.parse_args(argv)

    assigned = load_assigned_inputs(args.assigned)
    parameters = load_parameters(args.params)
    entry = {e.entry_id: e for e in read_corpus(args.corpus)}[args.entry]
    context = build_context(entry, assigned, args.structures)
    frame = context.anchor

    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=assigned.sampling.get("tether_chains", 20000),
        seed=assigned.sampling.get("seed", 0),
        target_surviving=assigned.sampling.get("target_surviving", 0),
    )
    tether = sampler.sample(
        linker=entry.linker,
        effector=context.effector,
        anchor=context.anchor,
        occupancy=context.occupancy,
        structure_id=entry.structure_id or "none",
        closure=context.closure,
    )

    # --- the deposited complex, moved into the anchor frame ---
    cas9 = load_structure(args.structures / "5F9R.cif.gz")
    protein, _ = chain_trace(cas9, "B", {"CA"})
    guide, _ = chain_trace(cas9, "A", {"P", "C4'"})
    target, _ = chain_trace(cas9, "C", {"P", "C4'"})
    displaced, displaced_ids = chain_trace(cas9, "D", {"P", "C4'"})

    editor = load_structure(args.structures / "6VPC.cif.gz")
    deaminase, deaminase_ids = chain_trace(editor, "F", {"CA"})

    spec = assigned.effectors[entry.effector] if hasattr(assigned, "effectors") else None
    junction = 159
    active = 201

    # Build the drawn conformation the way the model builds one: grow a linker
    # that never enters the excluded volume, put the deaminase's own fusion
    # residue at the far end, and let its active site fall where the deposited
    # domain puts it. A flexible linker leaves the domain rotationally
    # decorrelated, so the orientation is arbitrary and is taken as deposited.
    def residue_point(coords, ids, want):
        hit = np.where(ids == want)[0]
        return coords[hit[0]] if hit.size else coords[len(coords) // 2]

    junction = 159
    active = 201
    junction_local = residue_point(deaminase, deaminase_ids, junction)
    active_local = residue_point(deaminase, deaminase_ids, active)

    composition = assigned.compositions[entry.linker_composition]
    rise = float(getattr(composition, "rise", 3.6))
    persistence = float(getattr(composition, "persistence", 6.4))
    kwargs = dict(
        n_links=int(entry.linker_residues),
        link_length=rise,
        persistence=persistence,
        n_chains=20000,
        rng=np.random.default_rng(7),
    )
    if context.occupancy is not None:
        kwargs["blocked"] = context.occupancy.occluded
    sample = sample_free_chain(**kwargs)
    fields = [a for a in dir(sample) if not a.startswith("_")]
    joints = np.asarray(sample.coords)
    chain_weights = np.asarray(sample.weights, dtype=float)

    reach = np.linalg.norm(tether.positions, axis=1)
    wanted = float(np.median(reach))

    alive = np.where(chain_weights > 0)[0]
    ends = joints[alive, -1, :]
    order = alive[np.argsort(np.abs(np.linalg.norm(ends, axis=1) - wanted))]

    path, body, site = [], deaminase, active_local
    for candidate in order[:400]:
        chain = joints[candidate]
        shifted = deaminase - junction_local + chain[-1]
        if context.occupancy is not None:
            if np.any(context.occupancy.occluded(shifted)):
                continue
            if np.any(context.occupancy.occluded(chain[1:])):
                continue
        path = chain
        body = shifted
        site = active_local - junction_local + chain[-1]
        break
    junction_point = np.asarray(path)[-1] if len(path) else np.zeros(3)
    print("  drawn conformation: end at %.1f " % float(np.linalg.norm(junction_point)),
          "A from the anchor, median tether reach %.1f A" % wanted)

    payload = {
        "entry_id": entry.entry_id,
        "structures": {"editor": "5F9R", "deaminase": "6VPC chain F"},
        "chain_sample_fields": fields,
        "parameters": {
            "capture_radius": parameters.capture_radius,
            "linker_rise": rise,
            "linker_persistence": persistence,
            "linker_residues": int(entry.linker_residues),
        },
        "anchor": [0.0, 0.0, 0.0],
        "protein": [[round(float(v), 2) for v in p] for p in frame.to_local(protein)],
        "guide": [[round(float(v), 2) for v in p] for p in frame.to_local(guide)],
        "target_strand": [[round(float(v), 2) for v in p] for p in frame.to_local(target)],
        "displaced_strand": [[round(float(v), 2) for v in p] for p in frame.to_local(displaced)],
        "displaced_ids": [int(v) for v in displaced_ids],
        "deaminase": [[round(float(v), 2) for v in p] for p in body],
        "active_site": [round(float(v), 2) for v in site],
        "junction": [round(float(v), 2) for v in junction_point],
        "linker_path": [[round(float(v), 2) for v in p] for p in np.asarray(path)],
    }
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("wrote", args.out)
    print("  protein CA", len(payload["protein"]), "guide", len(payload["guide"]),
          "target", len(payload["target_strand"]), "displaced", len(payload["displaced_strand"]))
    print("  deaminase CA", len(payload["deaminase"]), "linker joints", len(payload["linker_path"]))
    print("  ChainSample fields:", fields)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
