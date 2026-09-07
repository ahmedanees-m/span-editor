import numpy as np

from span_editor.baselines import (
    DistanceStericsBaseline,
    LinearLinkerBaseline,
    MarginalBaseline,
    ModalLinkerBaseline,
    SubstrateOnlyBaseline,
    default_baselines,
)
from span_editor.occupancy import ModelParameters


def fitting_subset(contexts):
    return [context for context in contexts if context.entry.fits_parameters]


def test_default_set_has_the_four_profile_baselines():
    parameters = ModelParameters(capture_radius=9.0, ssdna_persistence=15.0, link_alpha=3.0)
    names = [baseline.name for baseline in default_baselines(parameters)]
    assert names == ["marginal", "substrate_only", "distance_sterics", "linear_in_length"]


def test_marginal_ignores_the_architecture(example_contexts):
    baseline = MarginalBaseline()
    baseline.fit(fitting_subset(example_contexts))

    first, second = example_contexts[0], example_contexts[3]
    assert first.entry.linker_residues != second.entry.linker_residues
    assert baseline.predict(first).mode == baseline.predict(second).mode


def test_substrate_only_returns_a_profile_over_the_context_indices(example_contexts):
    baseline = SubstrateOnlyBaseline(persistence=15.0)
    baseline.fit(fitting_subset(example_contexts))
    profile = baseline.predict(example_contexts[0])
    assert list(profile.indices) == example_contexts[0].indices
    assert np.all(profile.values >= 0)


def test_distance_sterics_reach_grows_with_linker_length(example_contexts):
    baseline = DistanceStericsBaseline(persistence=15.0)
    short = min(example_contexts, key=lambda c: c.entry.linker_residues)
    long = max(example_contexts, key=lambda c: c.entry.linker_residues)
    assert baseline._reach(short) < baseline._reach(long)


def test_distance_sterics_fit_selects_a_reach_and_a_link(example_contexts):
    baseline = DistanceStericsBaseline(persistence=15.0)
    baseline.fit(fitting_subset(example_contexts))
    assert 0.2 <= baseline.reach_factor <= 1.0
    assert baseline.alpha > 0


def test_linear_baseline_moves_its_mode_with_length(example_contexts):
    baseline = LinearLinkerBaseline()
    baseline.fit(fitting_subset(example_contexts))
    short = min(example_contexts, key=lambda c: c.entry.linker_residues)
    long = max(example_contexts, key=lambda c: c.entry.linker_residues)
    if abs(baseline.slope) > 1e-6:
        assert baseline.predict(short).mode != baseline.predict(long).mode


def test_modal_linker_picks_the_most_common_construct(example_contexts):
    baseline = ModalLinkerBaseline()
    baseline.fit([context.entry for context in example_contexts])
    assert baseline.linker.n_residues == 16
    assert baseline.linker.composition in {"xten", "helical"}
    assert baseline.propose(target_mode=6) == baseline.linker
