import numpy as np

from span_editor.invert import candidate_linkers, invert_window, reachable_positions
from span_editor.occupancy import ModelParameters

PARAMETERS = ModelParameters(capture_radius=9.0, ssdna_persistence=14.0, link_alpha=3.0)


def test_candidate_grid_covers_every_combination():
    candidates = candidate_linkers(["xten", "helical"], [8, 16, 24])
    assert len(candidates) == 6
    assert {linker.composition for linker in candidates} == {"xten", "helical"}


def test_inversion_recovers_the_linker_that_generated_the_window(model, small_context):
    target = model.predict(small_context, PARAMETERS)
    candidates = candidate_linkers(["xten"], [4, 10, 16, 22, 28])

    ranked = invert_window(model, small_context, PARAMETERS, target, candidates)
    assert ranked[0].mode_error == 0

    lengths = [result.linker.n_residues for result in ranked[:3]]
    assert small_context.entry.linker_residues in lengths


def test_inversion_accepts_a_bare_mode(model, small_context):
    target = model.predict(small_context, PARAMETERS)
    candidates = candidate_linkers(["xten"], [10, 16, 22])
    ranked = invert_window(model, small_context, PARAMETERS, target.mode, candidates)
    assert ranked[0].mode_error <= 1
    assert np.isnan(ranked[0].shape_error)


def test_inversion_leaves_the_original_entry_alone(model, small_context):
    before = small_context.entry.linker_residues
    invert_window(
        model, small_context, PARAMETERS, 6, candidate_linkers(["xten"], [8, 20])
    )
    assert small_context.entry.linker_residues == before


def test_reachable_map_lists_linkers_by_position(model, small_context):
    candidates = candidate_linkers(["xten"], [8, 16, 24])
    reachable = reachable_positions(model, small_context, PARAMETERS, candidates)
    assert reachable
    assert all(1 <= index <= small_context.spacer_length for index in reachable)
    assert all(entries for entries in reachable.values())
