from pathlib import Path

import numpy as np
import pytest

from span_editor.io import CorpusEntry, load_assigned_inputs, read_corpus
from span_editor.model import SpanModel
from span_editor.synthetic import SyntheticGeometry, synthetic_context
from span_editor.tether import TetherSampler

ROOT = Path(__file__).resolve().parents[1]

# Sample counts are kept small here. The tests check behaviour, not the
# precision of the estimates, and the analysis scripts carry their own settings.
TETHER_CHAINS = 1200
SUBSTRATE_SAMPLES = 200


@pytest.fixture(scope="session")
def assigned():
    return load_assigned_inputs(ROOT / "configs" / "assigned_inputs.yaml")


@pytest.fixture(scope="session")
def effector(assigned):
    return assigned.effectors["apobec1"]


@pytest.fixture(scope="session")
def model(assigned):
    return SpanModel(
        TetherSampler(compositions=assigned.compositions, n_chains=TETHER_CHAINS, seed=5)
    )


@pytest.fixture(scope="session")
def example_contexts(effector):
    entries = read_corpus(ROOT / "data" / "architectures.example.tsv")
    return [
        synthetic_context(
            entry,
            effector,
            geometry=SyntheticGeometry(standoff=22.0),
            n_samples=SUBSTRATE_SAMPLES,
            seed=index,
        )
        for index, entry in enumerate(entries)
    ]


@pytest.fixture(scope="session")
def small_context(effector):
    entry = CorpusEntry(
        entry_id="unit-01",
        source_key="unit",
        editor="unit",
        ortholog="SpCas9",
        anchor_site="n_terminal",
        linker_residues=16,
        linker_composition="xten",
        effector=effector.name,
        label="tether_varied",
        readout_class="amplicon_sequencing",
        tier="A",
        spacer_length=12,
        split="fit",
    )
    return synthetic_context(
        entry, effector, n_samples=SUBSTRATE_SAMPLES, seed=3
    )


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20260101)
