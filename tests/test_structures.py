"""Checks that need a downloaded structure.

The capture volume is not tuned against the effector pose resolved in 6VPC.
Tuning it that way would be circular: an ABE8e-class editor is the case the
model expects to sit outside the freely sampled ensemble, so calibrating the
null to contain it would calibrate on the one architecture declared not to be
null. The pose is used as a necessary condition instead. It must not be
sterically forbidden under the same coarse map the model itself uses.

These skip when the structures have not been fetched.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from span_editor.structures import coarse_occupancy, load_structure, sequence_table
from span_editor.structures import StructureGeometry, read_block

ROOT = Path(__file__).resolve().parents[1]
STRUCTURES = ROOT / "data" / "structures"
LISTING = ROOT / "data" / "structures.txt"
ASSIGNED = ROOT / "configs" / "assigned_inputs.yaml"


def settings():
    return yaml.safe_load(ASSIGNED.read_text(encoding="utf-8"))


def structure_path(accession):
    path = STRUCTURES / f"{accession}.cif.gz"
    if not path.exists():
        pytest.skip(f"{path.name} not fetched; run data/fetch_structures.py")
    return path


def listed_accessions():
    return [
        line.split()[0]
        for line in LISTING.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(("#", " "))
    ]


def test_structure_listing_is_readable():
    accessions = listed_accessions()
    assert accessions
    assert len(accessions) == len(set(accessions))
    assert all(len(accession) == 4 for accession in accessions)


def test_configured_structures_are_all_listed():
    chosen = {value for value in settings()["structures"].values() if value}
    assert chosen <= set(listed_accessions())


def test_every_configured_structure_has_a_geometry_block():
    document = settings()
    chosen = {value for value in document["structures"].values() if value}
    assert chosen <= set(document["structure_geometry"])


def test_spcas9_uses_a_structure_without_a_bound_deaminase():
    # 6VPC resolves the PAM-distal nucleotides only because the deaminase is
    # bound to them. Pinning the substrate ensemble there would place the strand
    # where the effector put it and then ask whether the effector can reach it.
    document = settings()
    assert document["structures"]["spcas9"] != document["falsification_check"]["accession"]
    assert not document["structure_geometry"]["6VPC"]["strand_residue_to_index"]


def test_spcas9_pins_only_the_pam_proximal_nucleotides():
    pins = settings()["structure_geometry"]["5F9R"]["strand_residue_to_index"]
    indices = sorted(int(value) for value in pins.values())
    assert indices == list(range(12, 21))


def test_protospacer_mapping_matches_the_deposited_sequences():
    from analysis.map_protospacer import map_structure

    structure_path("6VPC")
    mapping = map_structure("6VPC", "A", "D", 20, STRUCTURES)
    assert (mapping.first_residue, mapping.last_residue) == (20, 39)
    assert mapping.pam.endswith("GG")
    assert mapping.index_of(26) == 7
    assert mapping.modified_residues.get(26) == "8AZ"

    structure_path("5F9R")
    reference = map_structure("5F9R", "A", "D", 20, STRUCTURES)
    assert (reference.first_residue, reference.last_residue) == (1, 20)
    assert reference.matches == 20
    assert reference.pam.endswith("GG")
    assert reference.ordered_indices == list(range(12, 21))


def test_the_modified_base_falls_inside_the_reported_window():
    from analysis.map_protospacer import map_structure

    structure_path("6VPC")
    mapping = map_structure("6VPC", "A", "D", 20, STRUCTURES)
    index = mapping.index_of(int(settings()["falsification_check"]["target_residue"]))
    assert 4 <= index <= 8


def test_resolved_effector_pose_is_not_sterically_forbidden():
    check = settings()["falsification_check"]
    path = structure_path(check["accession"])
    structure = load_structure(path)

    effector = structure[structure.chain_id == check["effector_chain"]]
    assert effector.array_length() > 0

    site = effector[effector.res_id == int(check["active_site_residue"])]
    assert site.array_length() > 0, "the catalytic centre is absent from the effector chain"

    occupancy = coarse_occupancy(
        structure, settings()["coarse_radii"], chains=tuple(check["occluding_chains"])
    )
    assert not np.any(occupancy.occluded(site.coord))


def test_deposited_titles_match_the_ortholog_assignments():
    # 4UN3 is SpCas9 and 5CZZ is SaCas9. Asserting the titles keeps the
    # ortholog assignments from drifting.
    expected = {
        "5F9R": "pyogenes",
        "5CZZ": "aureus",
        "4UN3": "Cas9",
    }
    for accession, token in expected.items():
        block = read_block(structure_path(accession))
        title = str(block["struct"]["title"].as_array(str)[0])
        assert token.lower() in title.lower(), f"{accession}: {title}"

    document = settings()
    assert document["structures"]["spcas9"] == "5F9R"
    assert document["structures"]["sacas9"] == "5CZZ"


def test_guide_and_strand_chains_carry_the_expected_polymers():
    sequences = sequence_table(read_block(structure_path("5F9R")))
    assert set(sequences["D"]) <= set("ACGTN")
    assert len(sequences["D"]) >= 20


def test_the_rloop_closure_is_recorded_for_spcas9():
    closure = settings()["structure_geometry"]["5F9R"]["distal_closure"]
    assert closure["chain"] == "C"
    assert closure["index"] == 1
    assert closure["residue"] == 30


def test_reach_from_the_n_terminal_anchor_lands_in_the_reported_window():
    """The geometry alone, with nothing fitted, must put the accessible band
    where adenine base editors are reported to edit.

    A mapping running the other way would place the band at 13 to 17, and
    losing the R-loop closure constraint moves it onto position 1.
    """
    import numpy as np

    from analysis.substrate_check import reference_entry
    from span_editor.io import load_assigned_inputs
    from span_editor.model import build_context
    from span_editor.tether import TetherSampler

    structure_path("5F9R")
    assigned = load_assigned_inputs(ASSIGNED)
    assigned.sampling["substrate_samples"] = 400
    entry = reference_entry("spcas9", "5F9R", 20, "tada8e")
    context = build_context(entry, assigned, STRUCTURES)

    sampler = TetherSampler(compositions=assigned.compositions, n_chains=6000, seed=7)
    field = sampler.sample(
        entry.linker, context.effector, anchor=context.anchor, occupancy=context.occupancy
    )
    live = field.positions[field.weights > 0]
    assert live.shape[0] > 100, "excluded volume left too few conformations to judge"

    substrate = context.substrate(15.0)
    approach = {}
    for index in context.indices:
        cloud, weights = substrate.cloud(index)
        kept = cloud[weights > 0][:300]
        approach[index] = float(
            np.linalg.norm(live[:, None, :] - kept[None, :, :], axis=2).min()
        )

    closest = min(approach, key=approach.get)
    assert 4 <= closest <= 8, f"accessible band centred on index {closest}: {approach}"
    assert approach[closest] < approach[1]
    assert approach[closest] < approach[20]


def test_attachment_sites_are_named_and_a_missing_one_is_an_error():
    geometry = StructureGeometry.from_config(
        "TEST",
        {
            "anchor_chain": "B",
            "anchor_residue": 3,
            "strand_chain": "D",
            "strand_residue_to_index": {12: 12},
            "anchor_sites": {"n_terminal": 3, "hnh_amino": 793},
        },
    )
    assert geometry.residue_for("n_terminal") == 3
    assert geometry.residue_for("hnh_amino") == 793
    with pytest.raises(ValueError):
        geometry.residue_for("nowhere")

    without = StructureGeometry.from_config(
        "TEST",
        {
            "anchor_chain": "B",
            "anchor_residue": 7,
            "strand_chain": "D",
            "strand_residue_to_index": {12: 12},
        },
    )
    assert without.residue_for("anything") == 7
