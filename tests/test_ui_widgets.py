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
