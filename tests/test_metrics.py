import numpy as np
import pytest

from span_editor.geometry import NUMBERING_FROM_PAM_DISTAL, NUMBERING_FROM_PAM_PROXIMAL
from span_editor.metrics import (
    PERCENTILE,
    WILD,
    cluster_bootstrap,
    cluster_totals,
    mode_hit,
    mode_offset,
    paired_margin,
    report_in_both_conventions,
    saturation_statistic,
    wild_cluster_interval,
)
from span_editor.occupancy import Profile


def profile_at(mode, span=9):
    indices = np.arange(1, span + 1)
    values = np.exp(-0.5 * ((indices - mode) / 1.5) ** 2)
    return Profile(indices=indices, values=values)


def test_mode_offset_is_signed():
    assert mode_offset(profile_at(7), profile_at(5)) == 2
    assert mode_offset(profile_at(4), profile_at(6)) == -2
    assert mode_hit(profile_at(6), profile_at(5), tolerance=1)
    assert not mode_hit(profile_at(8), profile_at(5), tolerance=1)


def test_cluster_bootstrap_widens_as_clusters_are_pooled():
    rng = np.random.default_rng(0)
    values = rng.normal(0.4, 0.5, size=60)

    many = cluster_bootstrap(values, np.arange(60), seed=1)
    few = cluster_bootstrap(values, np.repeat(np.arange(6), 10), seed=1)

    assert many.n_clusters == 60
    assert few.n_clusters == 6
    assert (few.upper - few.lower) > (many.upper - many.lower)


def test_cluster_totals_preserves_the_sum_and_the_count():
    values = np.array([1.0, 0.0, 1.0, 1.0, 0.0])
    clusters = np.array(["a", "a", "b", "b", "b"])
    labels, sums, sizes = cluster_totals(values, clusters)
    assert list(labels) == ["a", "b"]
    assert sums.tolist() == [1.0, 2.0]
    assert sizes.tolist() == [2.0, 3.0]
    assert sums.sum() == values.sum()


def test_studentised_statistic_saturates_at_the_root_cluster_count():
    # This is why the null-imposed test cannot be inverted for this estimand:
    # a distant candidate inflates the standard error as fast as the numerator.
    assert saturation_statistic(np.full(9, 5.0)) == pytest.approx(3.0)
    assert saturation_statistic(np.full(16, 4.0)) == pytest.approx(4.0)


def test_wild_interval_is_finite_and_brackets_the_estimate():
    rng = np.random.default_rng(11)
    for n_clusters in (6, 9, 16):
        sums = rng.integers(-3, 4, size=n_clusters).astype(float)
        sizes = np.full(n_clusters, 5.0)
        lower, upper = wild_cluster_interval(sums, sizes, n_bootstrap=999, seed=2)
        estimate = sums.sum() / sizes.sum()
        assert np.isfinite(lower) and np.isfinite(upper)
        assert lower <= estimate <= upper
        assert (upper - lower) < 2.0


def test_wild_interval_is_wider_than_the_percentile_one():
    values = np.concatenate([np.ones(20), np.zeros(25)])
    clusters = np.repeat(np.arange(9), 5)
    wild = cluster_bootstrap(values, clusters, method=WILD, seed=3)
    percentile = cluster_bootstrap(values, clusters, method=PERCENTILE, seed=3)
    assert (wild.upper - wild.lower) > (percentile.upper - percentile.lower)
    assert wild.method == WILD and percentile.method == PERCENTILE


def test_unknown_interval_method_is_rejected():
    with pytest.raises(ValueError):
        cluster_bootstrap(np.zeros(10), np.repeat(np.arange(5), 2), method="jackknife")


def test_bootstrap_interval_covers_a_null_difference():
    rng = np.random.default_rng(2)
    difference = rng.normal(0.0, 0.3, size=40)
    interval = cluster_bootstrap(difference, np.repeat(np.arange(8), 5), seed=3)
    assert not interval.excludes_zero


def test_paired_margin_reports_the_clusters_it_used():
    hits = np.array([1, 1, 1, 0, 1, 1, 1, 1, 0, 1], dtype=float)
    reference = np.zeros(10)
    clusters = np.repeat(np.arange(5), 2)

    margin = paired_margin(hits, reference, clusters, "span", "marginal", seed=4)
    assert margin.n_entries == 10
    assert margin.interval.n_clusters == 5
    assert margin.interval.estimate == pytest.approx(0.8)
    assert margin.as_dict()["reference"] == "marginal"


def test_a_window_is_reported_in_both_conventions():
    cas9 = report_in_both_conventions(6, 20, NUMBERING_FROM_PAM_DISTAL)
    assert cas9 == {"protospacer_index": 6, "nucleotides_from_pam": 15}

    cas12a = report_in_both_conventions(10, 23, NUMBERING_FROM_PAM_PROXIMAL)
    assert cas12a == {"protospacer_index": 10, "nucleotides_from_pam": 10}
