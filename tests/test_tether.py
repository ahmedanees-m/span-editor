import numpy as np
import pytest

from span_editor.geometry import OccupancyMap
from span_editor.tether import (
    CompositionClass,
    EffectorSpec,
    LinkerSpec,
    TetherClosure,
    TetherSampler,
    random_rotations,
)

XTEN = CompositionClass(name="xten", persistence=6.4, rise=3.6)
HELICAL = CompositionClass(name="helical", persistence=800.0, rise=1.5, rigid=True)
EFFECTOR = EffectorSpec(
    name="demo", active_site_offset=np.array([16.0, 0.0, 0.0]), body_radius=14.0
)


def sampler(cache_dir=None, n_chains=4000):
    return TetherSampler(
        compositions={"xten": XTEN, "helical": HELICAL},
        n_chains=n_chains,
        seed=17,
        cache_dir=cache_dir,
    )


def test_random_rotations_are_proper():
    rng = np.random.default_rng(2)
    rotations = random_rotations(500, rng)
    products = np.einsum("nij,nkj->nik", rotations, rotations)
    assert np.allclose(products, np.eye(3), atol=1e-9)
    assert np.allclose(np.linalg.det(rotations), 1.0, atol=1e-9)


def test_longer_linkers_reach_further():
    engine = sampler()
    spreads = [
        engine.sample(LinkerSpec(length, "xten"), EFFECTOR).radius_of_gyration
        for length in (8, 16, 32)
    ]
    assert spreads[0] < spreads[1] < spreads[2]


def test_rigid_linker_extends_further_at_the_same_contour_length():
    # Twenty XTEN residues and forty-eight helical residues both span 72 A of
    # contour. The flexible chain coils back on itself and the helix does not.
    engine = sampler()
    flexible = engine.sample(LinkerSpec(20, "xten"), EFFECTOR)
    rigid = engine.sample(LinkerSpec(48, "helical"), EFFECTOR)

    assert rigid.directional_uncertainty
    assert not flexible.directional_uncertainty

    flexible_reach = np.linalg.norm(flexible.positions, axis=1)
    rigid_reach = np.linalg.norm(rigid.positions, axis=1)
    assert rigid_reach.mean() > 2 * flexible_reach.mean()
    assert rigid_reach.std() < flexible_reach.std()


def test_cache_returns_the_same_field(tmp_path):
    engine = sampler(cache_dir=tmp_path)
    first = engine.sample(LinkerSpec(16, "xten"), EFFECTOR)
    assert len(list(tmp_path.glob("tether-*.npz"))) == 1

    reloaded = sampler(cache_dir=tmp_path).sample(LinkerSpec(16, "xten"), EFFECTOR)
    assert np.array_equal(first.positions, reloaded.positions)
    assert np.array_equal(first.weights, reloaded.weights)


def test_sampling_is_reproducible_without_a_cache():
    first = sampler().sample(LinkerSpec(12, "xten"), EFFECTOR)
    second = sampler().sample(LinkerSpec(12, "xten"), EFFECTOR)
    assert np.array_equal(first.positions, second.positions)


def test_excluded_volume_removes_conformations():
    engine = sampler()
    free = engine.sample(LinkerSpec(16, "xten"), EFFECTOR)

    # A slab of spheres behind the anchor, which the chain would otherwise enter.
    axis = np.arange(-40.0, 40.1, 6.0)
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    centres = np.column_stack([np.full(grid.shape[0], -12.0), grid])
    occupancy = OccupancyMap(centres, 4.0)

    blocked = engine.sample(LinkerSpec(16, "xten"), EFFECTOR, occupancy=occupancy)
    assert blocked.surviving < blocked.sampled
    assert 0.0 < blocked.survival < 1.0
    assert blocked.positions.shape[0] == blocked.surviving
    assert blocked.weights.sum() == pytest.approx(1.0)
    assert blocked.centroid[0] > free.centroid[0]


def test_sampling_continues_until_the_survivor_target_is_met():
    axis = np.arange(-40.0, 40.1, 6.0)
    grid = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
    centres = np.column_stack([np.full(grid.shape[0], -12.0), grid])
    occupancy = OccupancyMap(centres, 4.0)

    engine = TetherSampler(
        compositions={"xten": XTEN, "helical": HELICAL},
        n_chains=500,
        seed=3,
        target_surviving=1200,
    )
    field = engine.sample(LinkerSpec(16, "xten"), EFFECTOR, occupancy=occupancy)
    assert field.surviving >= 1200
    assert field.sampled > 500
    assert field.effective_size == pytest.approx(field.positions.shape[0])


def test_unknown_composition_is_rejected():
    with pytest.raises(KeyError):
        sampler().sample(LinkerSpec(16, "polyproline"), EFFECTOR)


INSERTED = EffectorSpec(
    name="inserted",
    active_site_offset=np.array([16.0, 0.0, 0.0]),
    body_radius=14.0,
    exit_offset=np.array([0.0, 16.0, 0.0]),
    exit_tail=2,
)


def test_a_closure_pulls_the_field_towards_the_return_point():
    engine = sampler(n_chains=8000)
    linker = LinkerSpec(10, "xten")
    free = engine.sample(linker, INSERTED)
    near = TetherClosure(point=np.array([20.0, 0.0, 0.0]), n_links=4, composition="xten")
    held = engine.sample(linker, INSERTED, closure=near)

    assert held.positions.shape[0] < free.positions.shape[0]
    assert np.linalg.norm(held.centroid - near.point) < np.linalg.norm(free.centroid - near.point)
    assert held.radius_of_gyration < free.radius_of_gyration


def test_a_return_point_out_of_reach_leaves_nothing():
    engine = sampler(n_chains=4000)
    far = TetherClosure(point=np.array([400.0, 0.0, 0.0]), n_links=3, composition="xten")
    with pytest.raises(RuntimeError):
        engine.sample(LinkerSpec(6, "xten"), INSERTED, closure=far)


def test_an_effector_without_an_exit_cannot_be_inserted():
    engine = sampler(n_chains=1000)
    closure = TetherClosure(point=np.array([10.0, 0.0, 0.0]), n_links=4, composition="xten")
    with pytest.raises(ValueError):
        engine.sample(LinkerSpec(6, "xten"), EFFECTOR, closure=closure)


def test_closure_weights_sum_to_one():
    engine = sampler(n_chains=6000)
    closure = TetherClosure(point=np.array([25.0, 5.0, 0.0]), n_links=6, composition="xten")
    field = engine.sample(LinkerSpec(12, "xten"), INSERTED, closure=closure)
    assert np.isclose(float(np.sum(field.weights)), 1.0)
    assert field.effective_size <= field.positions.shape[0]
