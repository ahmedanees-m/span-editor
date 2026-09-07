import numpy as np
import pytest

from span_editor.power import (
    COMONOTONIC,
    INDEPENDENT,
    PERCENTILE,
    WILD,
    Design,
    bootstrap_excludes_zero,
    draw_trial,
    minimum_detectable,
    power_at,
    rejects_null,
    wild_cluster_rejects,
)


def design(**overrides):
    settings = dict(n_clusters=9, entries_per_cluster=5.0, base_rate=0.45, icc=0.3)
    settings.update(overrides)
    return Design(**settings)


def test_cluster_spread_follows_the_requested_correlation():
    assert design(icc=0.0).cluster_sigma() == 0.0
    sigma = design(icc=0.3).cluster_sigma()
    assert sigma == pytest.approx(np.sqrt(0.3 * (np.pi**2 / 3) / 0.7))
    assert design(icc=0.6).cluster_sigma() > sigma


def test_design_rejects_unusable_settings():
    with pytest.raises(ValueError):
        design(pairing="antithetic")
    with pytest.raises(ValueError):
        design(icc=1.0)


def test_comonotonic_null_leaves_no_discordant_pair():
    # One shared draw per entry with equal rates makes the two predictors agree
    # everywhere, so the paired difference is identically zero and no interval
    # can reject. The zero rejection rate is arithmetic, not coverage.
    rng = np.random.default_rng(1)
    for _ in range(50):
        sums, _ = draw_trial(design(pairing=COMONOTONIC), 0.0, rng)
        assert not np.any(sums)


def test_comonotonic_differences_are_never_negative():
    rng = np.random.default_rng(2)
    for _ in range(50):
        sums, _ = draw_trial(design(pairing=COMONOTONIC), 0.15, rng)
        assert np.all(sums >= 0)


def test_independent_null_produces_discordant_pairs_in_both_directions():
    rng = np.random.default_rng(3)
    sums = np.concatenate(
        [draw_trial(design(pairing=INDEPENDENT), 0.0, rng)[0] for _ in range(60)]
    )
    assert np.any(sums > 0)
    assert np.any(sums < 0)
    assert abs(sums.mean()) < 0.5


def test_bootstrap_rejects_a_clear_separation_and_not_a_null():
    rng = np.random.default_rng(4)
    sizes = np.full(12, 5.0)
    assert bootstrap_excludes_zero(np.full(12, 4.0), sizes, 500, rng)
    assert not bootstrap_excludes_zero(np.zeros(12), sizes, 500, rng)


def test_independent_null_rejects_near_the_nominal_rate():
    rng = np.random.default_rng(5)
    rate = power_at(design(n_clusters=40, pairing=INDEPENDENT), 0.0, 200, 600, rng)
    assert 0.01 <= rate <= 0.12


def test_power_rises_with_the_margin():
    rng = np.random.default_rng(6)
    small = power_at(design(), 0.05, 120, 500, rng)
    large = power_at(design(), 0.30, 120, 500, rng)
    assert large > small


def test_minimum_detectable_interpolates_between_grid_points():
    curve = [
        {"delta": 0.0, "power": 0.0},
        {"delta": 0.1, "power": 0.6},
        {"delta": 0.2, "power": 1.0},
    ]
    assert minimum_detectable(curve) == pytest.approx(0.15)
    assert minimum_detectable([{"delta": 0.0, "power": 0.1}]) is None


def test_wild_bootstrap_rejects_a_clear_separation_and_not_a_null():
    rng = np.random.default_rng(7)
    sizes = np.full(12, 5.0)
    assert wild_cluster_rejects(np.full(12, 4.0) + rng.normal(0, 0.3, 12), sizes, 999, rng)
    assert not wild_cluster_rejects(np.zeros(12), sizes, 999, rng)


def test_wild_bootstrap_holds_its_size_where_the_percentile_one_does_not():
    # Few clusters is the regime the corpus sits in, and it is where the
    # percentile interval rejects too often.
    small = design(n_clusters=6, pairing=INDEPENDENT)
    percentile = power_at(small, 0.0, 300, 800, np.random.default_rng(8), method=PERCENTILE)
    wild = power_at(small, 0.0, 300, 800, np.random.default_rng(8), method=WILD)
    assert percentile > 0.08
    assert wild < percentile


def test_unknown_interval_method_is_rejected():
    rng = np.random.default_rng(9)
    with pytest.raises(ValueError):
        rejects_null(np.zeros(5), np.full(5, 3.0), 100, rng, method="jackknife")
