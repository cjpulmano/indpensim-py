"""JSON round-trip tests for Recipe."""
from __future__ import annotations

import json

from indpensim.recipe import (
    Phase,
    Recipe,
    SetpointProfile,
    TransitionTrigger,
    from_dict,
    legacy_sbc_recipe,
    to_dict,
)


def test_round_trip_simple_recipe():
    original = Recipe(name="toy", phases=(
        Phase(
            name="P1",
            setpoints=SetpointProfile(
                Fs=((10.0, 1.0), (20.0, 2.0)),
                pressure=((100.0, 0.9),),
                T_sp=300.0,
            ),
            transition=TransitionTrigger(max_hours=4.0),
        ),
        Phase(
            name="P2",
            setpoints=SetpointProfile(Fpaa=((50.0, 5.0),)),
            transition=TransitionTrigger(
                state_var="X", state_op=">=", state_value=1.2,
            ),
        ),
    ))
    restored = from_dict(to_dict(original))
    assert restored == original


def test_round_trip_through_json_string():
    original = legacy_sbc_recipe()
    s = json.dumps(to_dict(original))
    restored = from_dict(json.loads(s))
    assert restored == original


def test_legacy_recipe_structure():
    r = legacy_sbc_recipe()
    names = [p.name for p in r.phases]
    assert names == ["INOCULATE", "GROWTH", "PRODUCTION", "HARVEST"]
    # All transitions are pure-time for legacy.
    for p in r.phases:
        assert p.transition.max_hours is not None
        assert p.transition.state_var is None
        # T_sp/pH_sp must stay None so the controller keeps reading ControlFlags.
        assert p.setpoints.T_sp is None
        assert p.setpoints.pH_sp is None


def test_legacy_recipe_h_scales_boundaries():
    # Halve h → boundaries in hours should also halve.
    r_base = legacy_sbc_recipe(h=0.2)
    r_half = legacy_sbc_recipe(h=0.1)
    for p_base, p_half in zip(r_base.phases, r_half.phases):
        assert p_base.transition.max_hours == 2 * p_half.transition.max_hours
