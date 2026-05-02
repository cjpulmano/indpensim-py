"""Recipe visualization page.

Two input modes:
  1. Use the in-progress recipe from session state (default).
  2. Upload a Recipe JSON to inspect (does not overwrite session).

Renders a stacked Plotly subplot — one row per channel that carries
data — with phase boundaries drawn as colored vertical bands. A
summary table below lists each phase's k-range, duration, and
trigger.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from indpensim.recipe import from_dict
from indpensim.ui.glossary import render_glossary_expander
from indpensim.ui.rendering import build_recipe_timeline_figure, phase_summary_rows
from indpensim.ui.sfc import build_sfc_dot, sfc_state_from_active
from indpensim.ui.state import current_recipe, init_session

st.set_page_config(page_title="Visualize | indpensim", layout="wide")
init_session()
render_glossary_expander()

st.title("Recipe visualization")

source = st.radio(
    "Source", options=("Session recipe", "Upload JSON"),
    horizontal=True,
)

recipe = None
if source == "Session recipe":
    try:
        recipe = current_recipe()
    except Exception as e:
        st.error(f"Session recipe is invalid: {e}")
else:
    uploaded = st.file_uploader("Upload Recipe JSON", type=["json"])
    if uploaded is not None:
        try:
            recipe = from_dict(json.loads(uploaded.read()))
        except Exception as e:
            st.error(f"Could not parse: {e}")

st.divider()

if recipe is None:
    st.info("No recipe to visualize yet.")
    st.stop()

st.subheader(f"Recipe: {recipe.name}")

tab_sfc, tab_timeline, tab_summary = st.tabs(
    ["SFC", "Setpoint timeline", "Phase summary"]
)

with tab_sfc:
    st.caption(
        "Sequential Function Chart. Steps (boxes) run top-to-bottom, "
        "separated by transitions (dark pills) whose condition must be "
        "true to advance."
    )
    mode = st.radio(
        "View mode",
        options=("Config", "Execution preview"),
        horizontal=True,
        help="Config is the static chart. Execution preview simulates "
             "which step is active — prior steps flip to completed.",
    )
    active_phase = None
    completed: tuple[str, ...] = ()
    held_phase: str | None = None
    aborted_phase: str | None = None

    if mode == "Execution preview":
        col_a, col_b = st.columns([2, 1])
        with col_a:
            phase_names = [p.name for p in recipe.phases]
            active_choice = st.selectbox(
                "Active phase", options=phase_names, index=0,
                help="The SFC will show this phase as RUNNING, prior "
                     "phases as COMPLETE, later phases as pending.",
            )
        with col_b:
            exec_state = st.selectbox(
                "Phase state",
                options=("RUNNING", "HELD", "ABORTED"),
                index=0,
            )
        completed, active = sfc_state_from_active(recipe, active_choice)
        if exec_state == "RUNNING":
            active_phase = active
        elif exec_state == "HELD":
            held_phase = active
        elif exec_state == "ABORTED":
            aborted_phase = active

    dot = build_sfc_dot(
        recipe,
        active_phase=active_phase,
        completed_phases=completed,
        held_phase=held_phase,
        aborted_phase=aborted_phase,
    )
    st.graphviz_chart(dot, use_container_width=True)

    # Legend
    st.caption(
        "Legend: "
        ":green[■] active (RUNNING) · "
        "■ completed · "
        ":orange[■] held · "
        ":red[■] aborted · "
        ":violet[■] pending"
    )

with tab_timeline:
    h = st.number_input(
        "Sample period h (hours)",
        value=0.2, min_value=0.01, step=0.05,
        help="Used to compute phase boundaries from time-in-phase triggers. "
             "0.2h matches the simulator default.",
    )
    fig = build_recipe_timeline_figure(recipe, h=h)
    st.plotly_chart(fig, use_container_width=True)

with tab_summary:
    rows = phase_summary_rows(recipe, h=0.2)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
