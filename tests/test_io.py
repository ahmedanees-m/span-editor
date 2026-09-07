import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from span_editor.io import (
    CorpusEntry,
    format_profile,
    load_assigned_inputs,
    load_parameters,
    parse_profile,
    parse_window,
    read_corpus,
    read_sources,
    sha256_of,
    write_fit_record,
)

ROOT = Path(__file__).resolve().parents[1]


def test_source_register_separates_its_roles():
    sources = read_sources(ROOT / "data" / "sources.tsv")
    assert sources
    roles = {source.role for source in sources}
    assert {"architecture", "profile", "external"} <= roles

    architecture = [source for source in sources if source.is_architecture_source]
    assert len(architecture) >= 12
    assert all(source.key for source in architecture)

    resolved = [source for source in sources if source.has_resolved_record]
    assert resolved
    assert all(source.doi.startswith("10.") for source in resolved)


def test_corpus_header_matches_the_data_dictionary():
    header = (ROOT / "data" / "architectures.tsv").read_text(encoding="utf-8").splitlines()[0]
    columns = header.split("\t")
    required = {
        "entry_id",
        "source_key",
        "linker_residues",
        "linker_composition",
        "label",
        "readout_class",
        "split",
        "numbering",
        "panel",
        "observed_profile",
    }
    assert required <= set(columns)


def test_example_corpus_parses():
    entries = read_corpus(ROOT / "data" / "architectures.example.tsv")
    assert len(entries) == 8
    assert {entry.split for entry in entries} == {"fit", "held_out"}
    assert all(entry.observed is not None for entry in entries)

    fitting = [entry for entry in entries if entry.fits_parameters]
    assert len(fitting) == 4


def test_profile_round_trip():
    profile = parse_profile("5:0.5;3:1.0;7:0.25")
    assert list(profile.indices) == [3, 5, 7]
    assert profile.mode == 3
    assert parse_profile(format_profile(profile)).values.tolist() == profile.values.tolist()
    assert parse_profile("") is None
    assert parse_profile("pending") is None


def test_entry_rejects_an_unknown_label():
    with pytest.raises(ValueError):
        CorpusEntry(
            entry_id="bad",
            source_key="demo",
            editor="demo",
            ortholog="SpCas9",
            anchor_site="n_terminal",
            linker_residues=16,
            linker_composition="xten",
            effector="rapobec1",
            label="tether-varied",
            readout_class="amplicon_sequencing",
            tier="A",
            spacer_length=20,
        )


def test_non_sequencing_entries_never_fit():
    entry = CorpusEntry(
        entry_id="yeast",
        source_key="tan2019",
        editor="demo",
        ortholog="SpCas9",
        anchor_site="n_terminal",
        linker_residues=16,
        linker_composition="xten",
        effector="rapobec1",
        label="tether_varied",
        readout_class="functional_selection",
        tier="A",
        spacer_length=20,
        split="fit",
    )
    assert not entry.fits_parameters


def test_configuration_files_load():
    parameters = load_parameters(ROOT / "configs" / "params.yaml")
    assert parameters.capture_radius > 0
    assert parameters.ssdna_persistence > 0
    assert parameters.link_alpha > 0

    assigned = load_assigned_inputs(ROOT / "configs" / "assigned_inputs.yaml")
    assert "xten" in assigned.compositions
    assert assigned.compositions["helical"].rigid
    assert assigned.compositions["xten"].persistence < assigned.compositions["helical"].persistence
    assert np.asarray(assigned.effectors["apobec1"].active_site_offset).shape == (3,)


def test_every_composition_carries_a_published_range():
    assigned = load_assigned_inputs(ROOT / "configs" / "assigned_inputs.yaml")
    for name, composition in assigned.compositions.items():
        assert composition.persistence_range, name
        low, high = composition.persistence_range
        assert low <= composition.persistence <= high, name


def test_the_ordered_classes_are_the_uncertain_ones():
    # Disordered chains are pinned to within a factor of one and a half.
    # Polyproline spans nearly an order of magnitude across methods, so a single
    # value would misstate what is known.
    assigned = load_assigned_inputs(ROOT / "configs" / "assigned_inputs.yaml")
    assert assigned.compositions["xten"].spread_in_persistence < 2
    assert assigned.compositions["polyproline"].spread_in_persistence > 5


def test_measured_effectors_carry_a_structure_and_a_body():
    import yaml

    document = yaml.safe_load(
        (ROOT / "configs" / "assigned_inputs.yaml").read_text(encoding="utf-8")
    )
    measured = [
        name
        for name, values in document["effectors"].items()
        if values.get("status") == "measured"
    ]
    assert {"tada8e", "abemax", "apobec1", "ha3a"} <= set(measured)
    for name in measured:
        values = document["effectors"][name]
        assert values["structure"] and values["chain"]
        assert len(values["body_residues"]) == 2
        assert values["body_radius"] > 0


def test_fit_record_hashes_both_configuration_files(tmp_path):
    record = write_fit_record(
        [ROOT / "configs" / "params.yaml", ROOT / "configs" / "assigned_inputs.yaml"],
        tmp_path / "fit_record.json",
    )
    assert len(record["files"]) == 2
    assert all(len(digest) == 64 for digest in record["files"].values())

    reloaded = json.loads((tmp_path / "fit_record.json").read_text(encoding="utf-8"))
    assert reloaded["files"] == record["files"]


def test_corpus_carries_curated_profiles():
    entries = read_corpus(ROOT / "data" / "architectures.tsv")
    assert len(entries) >= 26
    profiles = [entry for entry in entries if entry.observed is not None]
    assert len(profiles) >= 26
    assert {entry.ortholog for entry in entries} >= {"SpCas9", "SpG", "SpRY"}
    assert len({entry.context for entry in entries}) >= 9
    assert all(entry.panel for entry in entries)
    assert all(
        entry.observed is not None or entry.observed_window is not None
        for entry in entries
        if entry.tier == "A"
    )


def test_window_entries_parse_as_spans():
    entries = read_corpus(ROOT / "data" / "architectures.tsv")
    windows = [entry for entry in entries if entry.observed_window is not None]
    assert windows
    for entry in windows:
        first, last = entry.observed_window
        assert 1 <= first < last <= entry.spacer_length


def test_inserted_effectors_record_both_attachments():
    entries = read_corpus(ROOT / "data" / "architectures.tsv")
    inserted = [entry for entry in entries if entry.is_insertion]
    assert inserted
    for entry in inserted:
        assert entry.anchor_site != entry.return_site
        assert entry.return_residues > 0
        assert entry.effector.endswith("_inserted")


def test_window_parsing_round_trips():
    assert parse_window("4-8") == (4, 8)
    assert parse_window("") is None
    assert parse_window("pending") is None


def test_the_fitting_subset_is_one_architecture():
    entries = read_corpus(ROOT / "data" / "architectures.tsv")
    fitting = [entry for entry in entries if entry.fits_parameters]
    assert fitting
    assert {entry.editor for entry in fitting} == {"SpCas9-ABE8e"}
    assert all(entry.ortholog == "SpCas9" for entry in fitting)


def test_named_termini_agree_with_the_single_anchor_residue():
    """Adding named attachment sites must not move an entry that was already scored."""
    assigned = load_assigned_inputs(ROOT / "configs" / "assigned_inputs.yaml")
    for accession, settings in assigned.structure_geometry.items():
        sites = settings.get("anchor_sites") or {}
        if not sites or not settings.get("anchor_residue"):
            continue
        assert sites.get("n_terminal") == settings["anchor_residue"], accession


def test_the_fitted_parameters_match_the_recorded_fit():
    record = json.loads((ROOT / "results" / "fit_record.json").read_text(encoding="utf-8"))
    for name in ("configs/fitted.yaml", "configs/params.yaml"):
        current = sha256_of(ROOT / name)
        assert current == record["files"][name], name
