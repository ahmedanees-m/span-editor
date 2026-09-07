import numpy as np
import pytest

from span_editor.geometry import (
    NUMBERING_FROM_PAM_DISTAL,
    NUMBERING_FROM_PAM_PROXIMAL,
    Frame,
    OccupancyMap,
    StrandAnchor,
    SubstrateEnsemble,
    frame_from_backbone,
    pam_distance,
)


def test_frame_is_orthonormal_and_round_trips():
    frame = frame_from_backbone(
        nitrogen=np.array([-1.2, 1.4, 0.0]),
        alpha_carbon=np.array([0.0, 0.0, 0.0]),
        carbon=np.array([1.5, 0.3, 0.2]),
    )
    assert np.allclose(frame.basis @ frame.basis.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(frame.basis) == pytest.approx(1.0, abs=1e-9)

    points = np.array([[3.0, -2.0, 5.0], [0.0, 0.0, 0.0]])
    assert np.allclose(frame.to_global(frame.to_local(points)), points, atol=1e-9)


def test_frame_preserves_hand_computed_distances():
    frame = frame_from_backbone(
        nitrogen=np.array([0.0, 1.458, 0.0]),
        alpha_carbon=np.array([2.0, 2.0, 2.0]),
        carbon=np.array([3.525, 2.0, 2.0]),
    )
    points = np.array([[5.0, 2.0, 2.0], [2.0, 6.0, 2.0]])
    local = frame.to_local(points)
    assert np.linalg.norm(local[0]) == pytest.approx(3.0, abs=1e-9)
    assert np.linalg.norm(local[1]) == pytest.approx(4.0, abs=1e-9)
    assert np.linalg.norm(local[0] - local[1]) == pytest.approx(5.0, abs=1e-9)


def test_pam_distance_conventions():
    # Cas9 numbering runs from the PAM-distal end, so position 20 sits next to
    # the PAM. Cas12a numbering runs the other way.
    assert pam_distance(20, 20, NUMBERING_FROM_PAM_DISTAL) == 1
    assert pam_distance(1, 20, NUMBERING_FROM_PAM_DISTAL) == 20
    assert pam_distance(8, 23, NUMBERING_FROM_PAM_PROXIMAL) == 8
    with pytest.raises(ValueError):
        pam_distance(1, 20, "somewhere")


def test_occupancy_map_matches_direct_distances():
    rng = np.random.default_rng(0)
    centres = rng.uniform(-30, 30, size=(400, 3))
    radii = rng.uniform(2.5, 4.0, size=400)
    occupancy = OccupancyMap(centres, radii)

    query = rng.uniform(-35, 35, size=(500, 3))
    expected = np.any(
        np.linalg.norm(query[:, None, :] - centres[None, :, :], axis=2) < radii[None, :],
        axis=1,
    )
    assert np.array_equal(occupancy.occluded(query), expected)


def test_occupancy_probe_radius_widens_the_exclusion():
    occupancy = OccupancyMap(np.zeros((1, 3)), np.array([4.0]))
    probed = OccupancyMap(np.zeros((1, 3)), np.array([4.0]), probe_radius=3.0)
    point = np.array([[6.0, 0.0, 0.0]])
    assert not occupancy.occluded(point)[0]
    assert probed.occluded(point)[0]


def test_substrate_ensemble_pins_its_ordered_nucleotides():
    anchors = [
        StrandAnchor(index=1, position=np.array([20.0, -40.0, 0.0])),
        StrandAnchor(index=20, position=np.array([20.0, 40.0, 0.0])),
    ]
    ensemble = SubstrateEnsemble(
        anchors=anchors, spacer_length=20, rise=6.3, persistence=15.0, n_samples=400, seed=1
    )
    for anchor in anchors:
        cloud, weights = ensemble.cloud(anchor.index)
        assert np.allclose(cloud, anchor.position)
        assert weights.sum() == pytest.approx(1.0)

    assert ensemble.indices == list(range(1, 21))
    middle = ensemble.mean_position(10)
    assert np.linalg.norm(middle - np.array([20.0, 0.0, 0.0])) < 25.0


def test_substrate_ensemble_needs_an_anchor():
    with pytest.raises(ValueError):
        SubstrateEnsemble(anchors=[], spacer_length=20)


def test_frame_rotation_ignores_translation():
    frame = Frame(origin=np.array([5.0, 5.0, 5.0]), basis=np.eye(3))
    vector = np.array([[1.0, 0.0, 0.0]])
    assert np.allclose(frame.rotate_to_local(vector), vector)


def test_a_resolved_nucleotide_is_carried_as_one_point():
    anchors = [
        StrandAnchor(index=1, position=np.array([0.0, 0.0, 0.0])),
        StrandAnchor(index=10, position=np.array([20.0, 0.0, 0.0])),
    ]
    ensemble = SubstrateEnsemble(
        anchors=anchors, spacer_length=20, rise=6.3, persistence=15.0, n_samples=400, seed=1
    )
    for anchor in anchors:
        cloud, weights = ensemble.cloud(anchor.index)
        assert cloud.shape == (1, 3)
        assert weights.sum() == pytest.approx(1.0)

    jittered = SubstrateEnsemble(
        anchors=anchors,
        spacer_length=20,
        rise=6.3,
        persistence=15.0,
        n_samples=400,
        seed=1,
        anchor_jitter=4.0,
    )
    cloud, weights = jittered.cloud(1)
    assert cloud.shape == (400, 3)
    spread = np.sqrt(np.average(np.sum((cloud - cloud.mean(axis=0)) ** 2, axis=1), weights=weights))
    assert 4.0 < spread < 12.0
