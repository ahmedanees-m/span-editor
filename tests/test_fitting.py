import numpy as np
import pytest

from span_editor.fitting import fit_parameters, profile_loss
from span_editor.occupancy import ModelParameters
from span_editor.synthetic import synthetic_corpus

TRUTH = ModelParameters(capture_radius=9.0, ssdna_persistence=14.0, link_alpha=60.0)


@pytest.fixture(scope="module")
def generated(model, effector):
    contexts = synthetic_corpus(
        effector=effector,
        linker_lengths=(10, 26),
        standoffs=(22.0,),
        curvatures=(0.0,),
        n_samples=250,
        seed=41,
    )
    for context in contexts:
        context.entry.observed = model.predict(context, TRUTH).normalised()
    return contexts


def test_fit_recovers_the_generating_geometry(model, generated):
    result = fit_parameters(
        model,
        generated,
        radii=np.array([7.0, 9.0, 11.0]),
        persistences=np.array([10.0, 14.0, 20.0]),
        alphas=np.array([20.0, 60.0, 180.0]),
    )
    assert result.parameters.capture_radius == pytest.approx(TRUTH.capture_radius)
    assert result.parameters.ssdna_persistence == pytest.approx(TRUTH.ssdna_persistence)
    assert result.n_entries == len(generated)
    assert result.grid_shape == (3, 3, 3)


def test_the_generating_parameters_sit_at_the_minimum_of_the_loss(model, generated):
    at_truth = profile_loss(model, generated, TRUTH)
    for wrong in (
        ModelParameters(4.0, 14.0, TRUTH.link_alpha),
        ModelParameters(15.0, 14.0, TRUTH.link_alpha),
        ModelParameters(9.0, 30.0, TRUTH.link_alpha),
    ):
        assert profile_loss(model, generated, wrong) > at_truth


def test_fit_refuses_a_subset_with_no_observations(model, effector):
    contexts = synthetic_corpus(
        effector=effector,
        linker_lengths=(16,),
        standoffs=(22.0,),
        curvatures=(0.0,),
        n_samples=100,
    )
    with pytest.raises(ValueError):
        fit_parameters(
            model,
            contexts,
            radii=np.array([9.0]),
            persistences=np.array([14.0]),
            alphas=np.array([1.0]),
        )
