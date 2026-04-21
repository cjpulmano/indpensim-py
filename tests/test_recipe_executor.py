"""Unit tests for RecipeExecutor: transitions, phase advance, log, lookup semantics.

The legacy-parity test (``test_recipe_legacy_parity.py``) is the
end-to-end gate. These tests pin down the smaller mechanisms in
isolation so failures have a short explanation.
"""
from __future__ import annotations

import pytest

from indpensim.control.history import BatchHistory
from indpensim.recipe.executor import (
    PhaseTransitionLog,
    RecipeExecutor,
    _lookup,
    _trigger_fires,
)
from indpensim.recipe.legacy import _slice_for_range
from indpensim.recipe.types import (
    Phase,
    PhaseState,
    Recipe,
    SetpointProfile,
    TransitionTrigger,
)


# ---------------------------------------------------------------------------
# helpers: build small, hand-specified recipes for deterministic tests
# ---------------------------------------------------------------------------

def _make_recipe(phases: list[Phase]) -> Recipe:
    return Recipe(name="test", phases=tuple(phases))


def _empty_history(N: int = 100) -> BatchHistory:
    return BatchHistory.empty(N)


# ---------------------------------------------------------------------------
# _lookup semantics — must match controller._recipe_lookup exactly
# ---------------------------------------------------------------------------

def test_lookup_first_match():
    schedule = ((10.0, 1.0), (20.0, 2.0), (30.0, 3.0))
    assert _lookup(5, schedule) == 1.0
    assert _lookup(10, schedule) == 1.0
    assert _lookup(11, schedule) == 2.0
    assert _lookup(20, schedule) == 2.0
    assert _lookup(25, schedule) == 3.0


def test_lookup_fall_through_to_last():
    schedule = ((10.0, 1.0), (20.0, 2.0))
    assert _lookup(999, schedule) == 2.0


def test_lookup_empty_returns_default():
    assert _lookup(42, (), default=7.0) == 7.0


# ---------------------------------------------------------------------------
# _slice_for_range (legacy helper)
# ---------------------------------------------------------------------------

def test_slice_includes_first_bp_at_or_above_k_hi():
    bps = (40, 100, 200, 450, 1000, 1250, 1750)
    sps = (30, 42, 55, 60, 75, 65, 60)
    # PRODUCTION range (201, 1400] — naive slice of bps strictly inside
    # would miss 1750; must include it so fall-through picks SP=60, not 65.
    sliced = _slice_for_range(bps, sps, 201, 1400)
    # Ensure at k=1300 the sliced table returns the same SP as the full table.
    from indpensim.recipe.executor import _lookup
    from indpensim.control.controller import _recipe_lookup
    for k in (201, 400, 900, 1250, 1300, 1400):
        assert _lookup(k, sliced) == _recipe_lookup(k, bps, sps), f"mismatch at k={k}"


def test_slice_inoculate_phase():
    bps = (15, 60, 80, 100)
    sps = (8, 15, 30, 75)
    sliced = _slice_for_range(bps, sps, 1, 20)
    # Must include bp=60 because it's the first ≥ 20; bp=15 is the first ≥ 1.
    assert sliced == ((15.0, 8.0), (60.0, 15.0))


def test_slice_past_all_breakpoints_uses_last_sp():
    bps = (15, 60)
    sps = (8, 15)
    sliced = _slice_for_range(bps, sps, 100, 200)
    assert sliced == ((60.0, 15.0),)


# ---------------------------------------------------------------------------
# TransitionTrigger validation
# ---------------------------------------------------------------------------

def test_trigger_requires_at_least_one_field():
    with pytest.raises(ValueError, match="max_hours or a full state triplet"):
        TransitionTrigger()


def test_trigger_rejects_unknown_op():
    with pytest.raises(ValueError, match="unsupported state_op"):
        TransitionTrigger(state_var="X", state_op="!=", state_value=1.0)


def test_trigger_accepts_partial_state_if_time_set():
    # Time-only is legal even though state fields are None.
    TransitionTrigger(max_hours=1.0)


# ---------------------------------------------------------------------------
# _trigger_fires dispatch
# ---------------------------------------------------------------------------

def test_trigger_fires_on_time_threshold():
    trig = TransitionTrigger(max_hours=4.0)
    h = _empty_history()
    assert not _trigger_fires(trig, time_in_phase_h=3.9, history=h, k=10)
    assert _trigger_fires(trig, time_in_phase_h=4.0, history=h, k=10)


def test_trigger_fires_on_state_ge():
    trig = TransitionTrigger(state_var="X", state_op=">=", state_value=5.0)
    h = _empty_history()
    h.set("X", 9, 4.0)
    assert not _trigger_fires(trig, time_in_phase_h=0.0, history=h, k=10)
    h.set("X", 9, 5.0)
    assert _trigger_fires(trig, time_in_phase_h=0.0, history=h, k=10)


def test_trigger_fires_on_state_le():
    trig = TransitionTrigger(state_var="DO2", state_op="<=", state_value=10.0)
    h = _empty_history()
    h.set("DO2", 9, 15.0)
    assert not _trigger_fires(trig, time_in_phase_h=0.0, history=h, k=10)
    h.set("DO2", 9, 9.5)
    assert _trigger_fires(trig, time_in_phase_h=0.0, history=h, k=10)


def test_trigger_hybrid_time_fires_first():
    trig = TransitionTrigger(
        max_hours=1.0, state_var="X", state_op=">=", state_value=999.0,
    )
    h = _empty_history()  # X stays at zero
    assert _trigger_fires(trig, time_in_phase_h=1.0, history=h, k=10)


def test_trigger_hybrid_state_fires_first():
    trig = TransitionTrigger(
        max_hours=100.0, state_var="X", state_op=">=", state_value=5.0,
    )
    h = _empty_history()
    h.set("X", 9, 5.0)
    assert _trigger_fires(trig, time_in_phase_h=0.1, history=h, k=10)


# ---------------------------------------------------------------------------
# RecipeExecutor end-to-end (small, hand-built recipes)
# ---------------------------------------------------------------------------

def _two_phase_time_recipe() -> Recipe:
    return _make_recipe([
        Phase(
            name="P1",
            setpoints=SetpointProfile(Fs=((100.0, 10.0),)),
            transition=TransitionTrigger(max_hours=2.0),
        ),
        Phase(
            name="P2",
            setpoints=SetpointProfile(Fs=((200.0, 20.0),)),
            transition=TransitionTrigger(max_hours=10.0),
        ),
    ])


def test_executor_starts_on_first_phase():
    ex = RecipeExecutor(recipe=_two_phase_time_recipe(), h=0.2)
    assert ex.current_phase.name == "P1"
    assert ex.phase_state == PhaseState.RUNNING


def test_executor_time_based_advance():
    ex = RecipeExecutor(recipe=_two_phase_time_recipe(), h=0.2)
    h = _empty_history()
    # At k=1..10 the phase_start_k=1, so time_in_phase = (k-1)*0.2.
    # Trigger fires when ≥ 2.0 → (k-1)*0.2 ≥ 2.0 → k ≥ 11.
    for k in range(1, 11):
        out = ex.step(k, h)
        assert ex.current_phase.name == "P1"
        assert out.Fs == 10.0
    out = ex.step(11, h)
    assert ex.current_phase.name == "P2"
    assert out.Fs == 20.0


def test_executor_logs_transitions():
    ex = RecipeExecutor(recipe=_two_phase_time_recipe(), h=0.2)
    h = _empty_history()
    for k in range(1, 12):
        ex.step(k, h)
    log = ex.transitions
    assert len(log) == 1
    assert log[0] == PhaseTransitionLog(
        from_phase="P1", to_phase="P2",
        at_k=11, at_time_h=11 * 0.2, reason="trigger",
    )


def test_executor_drain_new_transitions_is_incremental():
    ex = RecipeExecutor(recipe=_two_phase_time_recipe(), h=0.2)
    h = _empty_history()
    for k in range(1, 12):
        ex.step(k, h)
    first = ex.drain_new_transitions()
    assert len(first) == 1
    # A second drain returns nothing (no new transitions since).
    assert ex.drain_new_transitions() == []


def test_executor_marks_complete_after_last_phase():
    # Single-phase recipe that fires immediately — should flip to COMPLETE
    # and stay on the same phase (setpoints continue to resolve).
    recipe = _make_recipe([
        Phase(
            name="ONLY",
            setpoints=SetpointProfile(Fs=((1000.0, 42.0),)),
            transition=TransitionTrigger(max_hours=0.0),
        ),
    ])
    ex = RecipeExecutor(recipe=recipe, h=0.2)
    h = _empty_history()
    out = ex.step(1, h)    # time_in_phase=0.0 ≥ 0.0 → fires
    assert ex.phase_state == PhaseState.COMPLETE
    assert ex.current_phase.name == "ONLY"
    assert out.Fs == 42.0
    # Further steps still work (setpoints resolved from the same phase).
    out = ex.step(2, h)
    assert out.Fs == 42.0


def test_executor_state_based_advance():
    recipe = _make_recipe([
        Phase(
            name="P1",
            setpoints=SetpointProfile(Fs=((100.0, 1.0),)),
            transition=TransitionTrigger(
                state_var="X", state_op=">=", state_value=5.0,
            ),
        ),
        Phase(
            name="P2",
            setpoints=SetpointProfile(Fs=((100.0, 2.0),)),
            transition=TransitionTrigger(max_hours=100.0),
        ),
    ])
    ex = RecipeExecutor(recipe=recipe, h=0.2)
    h = _empty_history()
    # X stays below threshold — no advance.
    h.set("X", 1, 1.0); ex.step(1, h)
    h.set("X", 2, 2.0); ex.step(2, h)
    assert ex.current_phase.name == "P1"
    # Push X over threshold for step 3 (reads history[k-1]=history[2]).
    h.set("X", 2, 5.5)
    out = ex.step(3, h)
    assert ex.current_phase.name == "P2"
    assert out.Fs == 2.0


def test_executor_mutator_hooks_raise():
    ex = RecipeExecutor(recipe=_two_phase_time_recipe(), h=0.2)
    with pytest.raises(NotImplementedError):
        ex.pause()
    with pytest.raises(NotImplementedError):
        ex.resume()
    with pytest.raises(NotImplementedError):
        ex.advance_phase()
    with pytest.raises(NotImplementedError):
        ex.abort()


def test_recipe_rejects_duplicate_phase_names():
    with pytest.raises(ValueError, match="duplicate phase names"):
        _make_recipe([
            Phase(name="A", setpoints=SetpointProfile(),
                  transition=TransitionTrigger(max_hours=1.0)),
            Phase(name="A", setpoints=SetpointProfile(),
                  transition=TransitionTrigger(max_hours=1.0)),
        ])


def test_recipe_rejects_empty_phases():
    with pytest.raises(ValueError, match="at least one phase"):
        Recipe(name="empty", phases=())
