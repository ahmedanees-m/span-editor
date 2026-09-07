import numpy as np
import pytest

from span_editor.geometry import StrandAnchor, SubstrateEnsemble
from span_editor.occupancy import (
    ModelParameters,
    MotifPreference,
    Profile,
    capture_probability,
    capture_probability_curve,
    predict_profile,
    saturating_link,
)
from span_editor.tether import EffectorSpec, LinkerSpec, TetherField


def build_tether(seed=0, n=800):
    rng = np.random.default_rng(seed)
    positions = rng.normal(loc=[18.0, 0.0, 0.0], scale=9.0, size=(n, 3))
    return TetherField(
        positions=positions,
        weights=np.full(n, 1.0 / n),
        linker=LinkerSpec(16, "xten"),
        effector="demo",
    )


def build_substrate(n=300):
    anchors = [
        StrandAnchor(index=1, position=np.array([20.0, -30.0, 0.0])),
        StrandAnchor(index=12, position=np.array([20.0, 30.0, 0.0])),
    ]
    return SubstrateEnsemble(
        anchors=anchors, spacer_length=12, rise=6.3, persistence=15.0, n_samples=n, seed=4
    )


def test_profile_mode_width_and_normalisation():
    profile = Profile(
        indices=np.arange(1, 8), values=np.array([0.1, 0.4, 0.9, 1.8, 0.9, 0.3, 0.05])
    )
    assert profile.mode == 4
    assert profile.width(0.5) == 3
    assert profile.centre_of_mass == pytest.approx(4.0, abs=0.2)
    assert profile.normalised().values.max() == pytest.approx(1.0)


def test_profile_mode_on_a_plateau_sits_in_the_middle():
    profile = Profile(indices=np.arange(1, 6), values=np.array([0.2, 1.0, 1.0, 1.0, 0.2]))
    assert profile.mode == 3


def test_link_is_monotone_and_saturates():
    values = np.linspace(0.0, 1.0, 25)
    linked = saturating_link(values, 3.0)
    assert np.all(np.diff(linked) > 0)
    assert linked[0] == pytest.approx(0.0)
    assert saturating_link(np.array([1.0]), 50.0)[0] == pytest.approx(1.0, abs=1e-6)


def test_capture_curve_agrees_with_the_direct_calculation():
    tether = build_tether()
    substrate = build_substrate()
    radii = np.array([4.0, 8.0, 12.0, 16.0])

    _, curve = capture_probability_curve(tether, substrate, radii)
    for column, radius in enumerate(radii):
        _, direct = capture_probability(tether, substrate, float(radius))
        assert np.allclose(curve[:, column], direct, atol=1e-9)


def test_capture_probability_grows_with_radius():
    tether = build_tether()
    substrate = build_substrate()
    _, curve = capture_probability_curve(tether, substrate, np.array([3.0, 6.0, 9.0, 12.0]))
    assert np.all(np.diff(curve, axis=1) >= -1e-12)


def test_motif_term_is_flat_when_targets_are_aggregated():
    motif = MotifPreference(name="apobec", base_weights={"T": 1.0, "C": 0.6})
    indices = np.arange(1, 6)
    assert np.allclose(motif.per_index(indices, sequence=None), 1.0)

    weights = motif.per_index(np.array([3, 4]), sequence="ATCGT")
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.6)


def test_forward_model_returns_a_profile_over_the_requested_indices():
    parameters = ModelParameters(capture_radius=9.0, ssdna_persistence=15.0, link_alpha=4.0)
    profile = predict_profile(
        build_tether(), build_substrate(), parameters, indices=list(range(2, 12))
    )
    assert list(profile.indices) == list(range(2, 12))
    assert np.all(profile.values >= 0)
    assert profile.values.max() > 0


def test_effector_spec_carries_its_offset():
    effector = EffectorSpec(
        name="demo", active_site_offset=np.array([16.0, 0.0, 0.0]), body_radius=14.0
    )
    assert np.linalg.norm(effector.active_site_offset) == pytest.approx(16.0)


def test_window_brackets_the_run_that_width_measures():
    profile = Profile(
        indices=np.arange(1, 9),
        values=np.array([0.1, 0.2, 0.6, 0.9, 1.0, 0.7, 0.3, 0.1]),
    )
    assert profile.window(0.5) == (3, 6)
    assert profile.width(0.5) == 4
    assert profile.window(0.25) == (3, 7)
    assert profile.window(0.15) == (2, 7)
