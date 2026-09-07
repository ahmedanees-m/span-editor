import numpy as np
import pytest

from span_editor.polymer import (
    bending_concentration,
    log_von_mises_fisher,
    mean_square_end_to_end,
    sample_bridge,
    sample_free_chain,
    sample_von_mises_fisher,
)


def test_end_to_end_limits():
    # A chain far stiffer than one link reaches its contour length. One far more
    # flexible than a link becomes freely jointed and reaches n times b squared.
    stiff = mean_square_end_to_end(20, 3.6, 1e5)
    assert stiff == pytest.approx((20 * 3.6) ** 2, rel=1e-3)

    jointed = mean_square_end_to_end(400, 3.6, 0.05)
    assert jointed == pytest.approx(400 * 3.6**2, rel=1e-6)


def test_long_chain_approaches_the_flory_ratio():
    link, persistence, n_links = 3.6, 6.4, 4000
    correlation = np.exp(-link / persistence)
    expected = n_links * link**2 * (1 + correlation) / (1 - correlation)
    assert mean_square_end_to_end(n_links, link, persistence) == pytest.approx(
        expected, rel=1e-3
    )


def test_bending_kernel_reproduces_the_requested_persistence():
    for link, persistence in [(3.6, 6.4), (6.3, 15.0), (1.5, 800.0)]:
        kappa = bending_concentration(link, persistence)
        mean_cosine = 1.0 / np.tanh(kappa) - 1.0 / kappa if kappa < 1e4 else 1.0 - 1.0 / kappa
        assert -link / np.log(mean_cosine) == pytest.approx(persistence, rel=1e-3)


def test_free_chain_matches_the_discrete_theory():
    rng = np.random.default_rng(11)
    sample = sample_free_chain(20, 3.6, 6.4, 40000, rng)
    observed = float(np.mean(np.sum(sample.endpoints**2, axis=1)))
    assert observed == pytest.approx(mean_square_end_to_end(20, 3.6, 6.4), rel=0.03)


def test_von_mises_fisher_mean_resultant():
    rng = np.random.default_rng(3)
    kappa = 5.0
    mean = np.tile(np.array([0.0, 0.0, 1.0]), (40000, 1))
    drawn = sample_von_mises_fisher(mean, np.full(40000, kappa), rng)
    expected = 1.0 / np.tanh(kappa) - 1.0 / kappa
    assert float(np.mean(drawn[:, 2])) == pytest.approx(expected, abs=0.01)


def test_von_mises_fisher_density_integrates_to_one():
    # Integrate the density over the sphere by uniform Monte Carlo.
    rng = np.random.default_rng(5)
    points = rng.normal(size=(200000, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    mean = np.tile(np.array([1.0, 0.0, 0.0]), (points.shape[0], 1))
    density = np.exp(log_von_mises_fisher(points, mean, np.full(points.shape[0], 2.0)))
    assert float(np.mean(density) * 4.0 * np.pi) == pytest.approx(1.0, abs=0.02)


def test_bridge_reaches_its_target():
    rng = np.random.default_rng(7)
    target = np.array([40.0, 10.0, -5.0])
    sample = sample_bridge(16, 6.3, 15.0, target, 4000, rng)
    landing = np.average(sample.endpoints, axis=0, weights=sample.weights)
    assert np.linalg.norm(landing - target) < 3.0
    assert sample.effective_size > 50


def test_bridge_rejects_an_unreachable_target():
    rng = np.random.default_rng(9)
    with pytest.raises(ValueError):
        sample_bridge(4, 3.6, 6.4, np.array([100.0, 0.0, 0.0]), 100, rng)
