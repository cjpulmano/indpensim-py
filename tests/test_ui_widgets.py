"""Unit tests for the streamlit UI's pure-function helpers.

The streamlit-bound widgets in ``indpensim.ui.widgets`` are not unit
tested (they have no value outside a Streamlit runtime). What's tested
here is the dataclass ↔ form-state conversion in ``state.py`` and the
plotly figure builder in ``rendering.py`` — both runtime-independent.

Skips cleanly if ``streamlit`` or ``plotly`` are not installed.
"""
from __future__ import annotations

import pytest

streamlit = pytest.importorskip("streamlit")
plotly = pytest.importorskip("plotly")

import pandas as pd                                  # noqa: E402
import plotly.graph_objects as go                    # noqa: E402

from indpensim.recipe import (                        # noqa: E402
    Phase,
    Recipe,
    SetpointProfile,
    TransitionTrigger,
    from_dict,
    legacy_sbc_recipe,
    to_dict,
)
from indpensim.ui.rendering import (                  # noqa: E402
    build_recipe_timeline_figure,
    phase_summary_rows,
)
from indpensim.ui.sfc import (                         # noqa: E402
    build_sfc_dot,
    sfc_state_from_active,
)
from indpensim.ui.units import c_to_k, format_temp_c, k_to_c   # noqa: E402
from indpensim.ui.state import (                      # noqa: E402
    SETPOINT_CHANNELS,
    empty_phase,
    rows_to_schedule,
    schedule_to_rows,
)


# ---------------------------------------------------------------------------
# state.py — pure helpers
# ---------------------------------------------------------------------------

def test_empty_phase_has_all_channels():
    p = empty_phase("FOO")
    assert p["name"] == "FOO"
    for ch in SETPOINT_CHANNELS:
        assert p["setpoints"][ch] == []
    assert p["setpoints"]["T_sp"] is None
    assert p["setpoints"]["pH_sp"] is None
    assert p["transition"]["max_hours"] == 1.0


def test_empty_phase_round_trips_when_attached_to_recipe():
    """An empty_phase appended to a recipe must validate (Recipe(...) doesn't
    raise) — i.e. the trigger is well-formed by default."""
    p_dict = empty_phase("STAGE_A")
    recipe_dict = {"name": "tmp", "phases": [p_dict]}
    recipe = from_dict(recipe_dict)
    assert recipe.phases[0].name == "STAGE_A"


def test_schedule_to_rows_round_trip_via_list():
    schedule = [[10.0, 1.5], [20.0, 3.0], [30.0, 7.5]]
    rows = schedule_to_rows(schedule)
    assert rows == [
        {"breakpoint_k": 10.0, "value": 1.5},
        {"breakpoint_k": 20.0, "value": 3.0},
        {"breakpoint_k": 30.0, "value": 7.5},
    ]
    assert rows_to_schedule(rows) == schedule


def test_schedule_to_rows_round_trip_via_dataframe():
    schedule = [[15.0, 2.0], [60.0, 8.5]]
    rows = schedule_to_rows(schedule)
    df = pd.DataFrame(rows)
    assert rows_to_schedule(df) == schedule


def test_rows_to_schedule_drops_partial_rows():
    rows = [
        {"breakpoint_k": 10.0, "value": 1.0},
        {"breakpoint_k": None, "value": 5.0},   # in-progress edit, drop
        {"breakpoint_k": 20.0, "value": None},  # in-progress edit, drop
        {"breakpoint_k": 30.0, "value": 9.0},
    ]
    assert rows_to_schedule(rows) == [[10.0, 1.0], [30.0, 9.0]]


def test_legacy_recipe_dict_round_trip_via_state():
    d = to_dict(legacy_sbc_recipe())
    restored = from_dict(d)
    assert restored == legacy_sbc_recipe()


# ---------------------------------------------------------------------------
# rendering.py — plotly figure builder
# ---------------------------------------------------------------------------

def test_build_figure_for_legacy_recipe():
    fig = build_recipe_timeline_figure(legacy_sbc_recipe(), h=0.2)
    assert isinstance(fig, go.Figure)
    # All 7 channels are populated in legacy → expect 7 subplots (rows).
    # Plotly stores each row's title in fig.layout.annotations (subplot titles).
    titles = [a.text for a in fig.layout.annotations if a.text]
    # Legacy populates Fs, Foil, Fg, pressure, Fpaa, Fwater, Fdischarge.
    expected = {"Fs (L/h)", "Foil (L/h)", "Fg (L/h)", "pressure (bar)",
                 "Fpaa (L/h)", "Fwater (L/h)", "Fdischarge (L/h)"}
    title_strs = set(titles)
    # Subplot titles are present (other annotations are phase band labels).
    assert expected.issubset(title_strs), title_strs


def test_build_figure_handles_empty_recipe_gracefully():
    """A recipe with all-empty schedules should still produce a Figure."""
    recipe = Recipe(name="empty", phases=(
        Phase(
            name="ONLY",
            setpoints=SetpointProfile(),
            transition=TransitionTrigger(max_hours=10.0),
        ),
    ))
    fig = build_recipe_timeline_figure(recipe, h=0.2)
    assert isinstance(fig, go.Figure)


def test_phase_summary_rows_for_legacy_recipe():
    rows = phase_summary_rows(legacy_sbc_recipe(), h=0.2)
    names = [r["Phase"] for r in rows]
    assert names == ["INOCULATE", "GROWTH", "PRODUCTION", "HARVEST"]
    # Triggers should all be time-based for legacy.
    for r in rows:
        assert "t≥" in r["Trigger"]
        assert "h" in r["Duration"]


def test_phase_summary_handles_state_only_trigger():
    recipe = Recipe(name="hybrid", phases=(
        Phase(name="P1",
              setpoints=SetpointProfile(),
              transition=TransitionTrigger(state_var="X", state_op=">=", state_value=5.0)),
        Phase(name="P2",
              setpoints=SetpointProfile(),
              transition=TransitionTrigger(max_hours=10.0)),
    ))
    rows = phase_summary_rows(recipe, h=0.2)
    assert "state-trigger" in rows[0]["Duration"]
    assert "X>=" in rows[0]["Trigger"]


# ---------------------------------------------------------------------------
# sfc.py — SFC DOT builder
# ---------------------------------------------------------------------------

def test_sfc_dot_config_view_for_legacy_recipe():
    dot = build_sfc_dot(legacy_sbc_recipe())
    # Structural sanity
    assert dot.startswith("digraph SFC")
    assert dot.rstrip().endswith("}")
    # All phase names appear as labels.
    for name in ("INOCULATE", "GROWTH", "PRODUCTION", "HARVEST"):
        assert name in dot
    # Initial/final bars present.
    assert "_start" in dot
    assert "_end" in dot
    # One transition pill per phase.
    assert dot.count("t0 [") == 1
    assert dot.count("t1 [") == 1
    assert dot.count("t2 [") == 1
    assert dot.count("t3 [") == 1


def test_sfc_dot_config_has_no_active_highlight():
    """Pure config view: no phase should be colored with the active-green."""
    dot = build_sfc_dot(legacy_sbc_recipe())
    assert "#6DBE72" not in dot  # active green
    assert "#BAD5BA" not in dot  # completed green


def test_sfc_dot_execution_view_colors_active_and_completed():
    recipe = legacy_sbc_recipe()
    completed, active = sfc_state_from_active(recipe, "PRODUCTION")
    assert completed == ("INOCULATE", "GROWTH")
    assert active == "PRODUCTION"
    dot = build_sfc_dot(recipe, active_phase=active, completed_phases=completed)
    # Active green + completed muted-green should both appear.
    assert "#6DBE72" in dot
    assert "#BAD5BA" in dot


def test_sfc_dot_held_and_aborted_colors():
    recipe = legacy_sbc_recipe()
    dot_held = build_sfc_dot(recipe, held_phase="GROWTH")
    dot_abort = build_sfc_dot(recipe, aborted_phase="HARVEST")
    assert "#F5D76E" in dot_held    # amber
    assert "#E57A7A" in dot_abort   # red


def test_sfc_trigger_text_for_hybrid_trigger():
    recipe = Recipe(name="hybrid", phases=(
        Phase(name="P1",
              setpoints=SetpointProfile(),
              transition=TransitionTrigger(
                  max_hours=12.0,
                  state_var="X", state_op=">=", state_value=4.5,
              )),
    ))
    dot = build_sfc_dot(recipe)
    # Both components present, joined.
    assert "t≥12h" in dot
    assert "X>=4.5" in dot
    assert " or " in dot


def test_sfc_state_from_active_handles_unknown_name():
    recipe = legacy_sbc_recipe()
    completed, active = sfc_state_from_active(recipe, "NOT_A_PHASE")
    assert completed == ()
    assert active is None


def test_sfc_state_from_active_first_phase_has_no_completed():
    recipe = legacy_sbc_recipe()
    completed, active = sfc_state_from_active(recipe, "INOCULATE")
    assert completed == ()
    assert active == "INOCULATE"


# ---------------------------------------------------------------------------
# units.py — Kelvin ↔ Celsius
# ---------------------------------------------------------------------------

def test_k_c_round_trip():
    # Typical fermentation setpoint: 298.15 K = 25.0 °C
    assert k_to_c(298.15) == pytest.approx(25.0)
    assert c_to_k(25.0) == pytest.approx(298.15)
    # Round-trip for a few arbitrary values
    for c in (0.0, 18.5, 37.0, 100.0):
        assert k_to_c(c_to_k(c)) == pytest.approx(c)


def test_format_temp_c_rendering():
    assert format_temp_c(298.15) == "25.0°C"
    assert format_temp_c(298.15, decimals=2) == "25.00°C"


def test_sfc_step_box_renders_temp_in_celsius():
    """If a phase has T_sp set (in Kelvin), the SFC step-box summary
    should display it in Celsius — the UI boundary contract."""
    recipe = Recipe(name="warm", phases=(
        Phase(
            name="BAKE",
            setpoints=SetpointProfile(T_sp=298.15),
            transition=TransitionTrigger(max_hours=5.0),
        ),
    ))
    dot = build_sfc_dot(recipe)
    assert "T=25.0°C" in dot
    # The Kelvin value shouldn't leak into the display.
    assert "T=298.15" not in dot
    assert "T=298" not in dot


# ---------------------------------------------------------------------------
# Smoke imports for pages — they must import without crashing
# ---------------------------------------------------------------------------

def test_streamlit_app_imports():
    # Streamlit's `st.set_page_config` will raise if called outside a
    # ScriptRunContext, but only as a warning under newer versions.
    # The import itself must succeed.
    import importlib
    importlib.import_module("indpensim.ui.streamlit_app")


def test_authoring_page_imports():
    import importlib
    importlib.import_module("indpensim.ui.pages.01_authoring")


def test_visualize_page_imports():
    import importlib
    importlib.import_module("indpensim.ui.pages.02_visualize")
