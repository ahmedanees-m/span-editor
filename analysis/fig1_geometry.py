"""Emit the sampled geometry and the per-position decomposition behind Fig 1.

Panel A of Fig 1 draws the sampled ensembles rather than a cartoon, and panel B
separates the linker field from the substrate ensemble before their convolution.
Neither quantity is kept by the scoring runs, so this writes them once, for the
entry the parameters were fitted to, at the frozen parameters.

Run from the repository root:  python analysis/fig1_geometry.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from span_editor.io import load_assigned_inputs, load_parameters, read_corpus
from span_editor.model import build_context
from span_editor.occupancy import Profile, capture_probability, saturating_link
from span_editor.tether import TetherSampler

ENTRY = "kissling-SpCas9-ABE8e-HEK-Plasmid-5d"


class RigidSubstrate:
    """The strand held at one point per position, so capture reads the tether alone."""

    def __init__(self, indices, points):
        self.indices = list(indices)
        self._points = {int(i): np.asarray(p, dtype=float) for i, p in zip(indices, points)}

    def cloud(self, index):
        return self._points[int(index)][None, :], np.array([1.0])


def subsample(rng, array, n, weights=None):
    array = np.asarray(array)
    if array.shape[0] <= n:
        keep = np.arange(array.shape[0])
    else:
        keep = rng.choice(array.shape[0], size=n, replace=False)
    if weights is None:
        return array[keep], None
    return array[keep], np.asarray(weights)[keep]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/architectures.tsv"))
    parser.add_argument("--assigned", type=Path, default=Path("configs/assigned_inputs.yaml"))
    parser.add_argument("--params", type=Path, default=Path("configs/params.yaml"))
    parser.add_argument("--structures", type=Path, default=Path("data/structures"))
    parser.add_argument("--entry", default=ENTRY)
    parser.add_argument("--out", type=Path, default=Path("results/fig1_geometry.json"))
    args = parser.parse_args(argv)

    assigned = load_assigned_inputs(args.assigned)
    parameters = load_parameters(args.params)
    settings = yaml.safe_load(args.params.read_text("utf-8"))

    entries = {e.entry_id: e for e in read_corpus(args.corpus)}
    entry = entries[args.entry]

    sampler = TetherSampler(
        compositions=assigned.compositions,
        n_chains=assigned.sampling.get("tether_chains", 20000),
        seed=assigned.sampling.get("seed", 0),
        target_surviving=assigned.sampling.get("target_surviving", 0),
    )
    context = build_context(entry, assigned, args.structures)
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
    substrate = context.substrate(parameters.ssdna_persistence, obstacle)

    indices = list(context.indices)
    _, capture = capture_probability(
        tether, substrate, parameters.capture_radius, indices
    )

    means, spreads, clouds = [], [], {}
    rng = np.random.default_rng(20260101)
    for index in indices:
        cloud, weights = substrate.cloud(index)
        weights = np.asarray(weights, dtype=float)
        total = float(np.sum(weights)) or 1.0
        mean = np.sum(cloud * weights[:, None], axis=0) / total
        rms = float(
            np.sqrt(np.sum(weights * np.sum((cloud - mean) ** 2, axis=1)) / total)
        )
        means.append(mean)
        spreads.append(rms)
        pts, _ = subsample(rng, cloud, 120)
        clouds[int(index)] = [[round(float(v), 2) for v in p] for p in pts]

    _, tether_only = capture_probability(
        tether, RigidSubstrate(indices, means), parameters.capture_radius, indices
    )

    linked = saturating_link(np.asarray(capture), parameters.link_alpha)

    positions, weights = subsample(rng, tether.positions, 4000, tether.weights)
    occ = context.occupancy
    centres, _ = subsample(rng, occ.centres, 2500) if occ is not None else (np.empty((0, 3)), None)
    radius = float(np.median(occ.radii)) if occ is not None else 0.0

    anchors = sorted(a.index for a in context.strand_anchors)
    payload = {
        "entry_id": entry.entry_id,
        "structure": entry.structure_id,
        "effector": context.effector.name if hasattr(context.effector, "name") else str(entry.effector),
        "linker_residues": entry.linker_residues,
        "linker_composition": entry.linker_composition,
        "parameters": {
            "capture_radius": parameters.capture_radius,
            "ssdna_persistence": parameters.ssdna_persistence,
            "link_alpha": parameters.link_alpha,
        },
        "effector_geometry": {
            "active_site_offset": float(context.effector.offset_magnitude)
            if hasattr(context.effector, "offset_magnitude")
            else None,
            "body_radius": float(getattr(context.effector, "body_radius", 0.0)),
        },
        "anchored_positions": anchors,
        "note": (
            "Coordinates are in the anchor frame, so the fusion residue is the origin. "
            "tether_field is capture against the strand held at its ensemble mean, which "
            "isolates the linker; capture is the same convolution against the sampled "
            "ensemble; substrate_rms is the spread of that ensemble."
        ),
        "decomposition": {
            "index": [int(i) for i in indices],
            "tether_field": [round(float(v), 8) for v in tether_only],
            "capture": [round(float(v), 8) for v in capture],
            "linked": [round(float(v), 8) for v in linked],
            "substrate_rms": [round(float(v), 3) for v in spreads],
        },
        "occupancy": {
            "sphere_radius": round(radius, 3),
            "n_spheres": int(occ.centres.shape[0]) if occ is not None else 0,
            "centres": [[round(float(v), 2) for v in c] for c in centres],
        },
        "tether": {
            "n_surviving": int(tether.surviving),
            "n_sampled": int(tether.sampled),
            "positions": [[round(float(v), 2) for v in p] for p in positions],
            "weights": [round(float(w), 6) for w in weights],
        },
        "strand": {
            "index": [int(i) for i in indices],
            "mean": [[round(float(v), 2) for v in m] for m in means],
            "rms": [round(float(s), 2) for s in spreads],
            "cloud": clouds,
        },
    }
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("wrote", args.out)
    print("  surviving chains", tether.surviving, "of", tether.sampled)
    print("  capture peak at index",
          indices[int(np.argmax(capture))], "value", float(np.max(capture)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
