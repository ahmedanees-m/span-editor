from pathlib import Path

import numpy as np
import pytest

from span_editor.io import CorpusEntry, read_corpus
from span_editor.model import build_context
from span_editor.occupancy import ModelParameters
from span_editor.structures import SIZE_CLASS, StructureGeometry

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ModelParameters(capture_radius=9.0, ssdna_persistence=14.0, link_alpha=40.0)


def test_context_caches_its_ensemble_per_persistence(small_context):
    first = small_context.substrate(14.0)
    assert small_context.substrate(14.0) is first
    assert small_context.substrate(20.0) is not first


def test_prediction_covers_the_requested_indices(model, small_context):
    profile = model.predict(small_context, PARAMETERS)
    assert list(profile.indices) == small_context.indices
    assert np.all(np.isfinite(profile.values))
    assert profile.values.max() > 0


def test_prediction_moves_when_the_tether_lengthens(model, small_context, effector):
    from dataclasses import replace

    short = model.predict(small_context, PARAMETERS)
    longer = replace(small_context.entry, linker_residues=40)
    stretched = replace(small_context, entry=longer, _ensembles=small_context._ensembles)
    assert not np.allclose(short.values, model.predict(stretched, PARAMETERS).values)


def test_build_context_reports_a_missing_assignment(assigned):
    from dataclasses import replace

    entry = replace(
        read_corpus(ROOT / "data" / "architectures.example.tsv")[0], structure_id="5CZZ"
    )
    with pytest.raises(ValueError, match="assignments still to be made"):
        build_context(entry, assigned, ROOT / "data" / "structures")


def test_build_context_reports_a_missing_structure_id(assigned):
    entry = CorpusEntry(
        entry_id="no-structure",
        source_key="unit",
        editor="unit",
        ortholog="SpCas9",
        anchor_site="n_terminal",
        linker_residues=16,
        linker_composition="xten",
        effector="rapobec1",
        label="tether_varied",
        readout_class="amplicon_sequencing",
        tier="A",
        spacer_length=20,
    )
    with pytest.raises(ValueError, match="no structure recorded"):
        build_context(entry, assigned, ROOT / "data" / "structures")


def test_structure_geometry_lists_every_missing_field():
    with pytest.raises(ValueError) as raised:
        StructureGeometry.from_config("6VPC", {"anchor_chain": "A"})
    message = str(raised.value)
    for field in ("anchor_residue", "strand_chain", "strand_residue_to_index"):
        assert field in message


def test_size_classes_cover_the_standard_residues():
    assert len(SIZE_CLASS) == 20
    assert SIZE_CLASS["GLY"] == "glycine"
    assert SIZE_CLASS["TRP"] == "large"
