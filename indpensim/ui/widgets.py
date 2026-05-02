"""Reusable Streamlit widgets — phase editor, transition editor.

Each widget renders into the current Streamlit page and mutates the
phase/recipe dict in place. They return nothing; the caller writes
back to ``st.session_state["recipe_dict"]`` when committing edits.

These are coupled to streamlit at import time on purpose — they have
no value outside that runtime. Pure-function helpers (dict ↔ form
state conversion) live in ``state.py`` so they can be unit-tested
without spinning up a Streamlit session.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from indpensim.ui.state import (
    SETPOINT_CHANNELS,
    rows_to_schedule,
    schedule_to_rows,
)
from indpensim.ui.units import c_to_k, k_to_c


_STATE_VAR_OPTIONS = (
    "(none)", "X", "P", "S", "DO2", "PAA", "NH3", "T", "pH", "Culture_age",
)


def setpoint_schedule_editor(label: str, schedule: list, key: str) -> list[list[float]]:
    """Render an editable [breakpoint_k, value] table. Returns the new schedule."""
    initial = pd.DataFrame(schedule_to_rows(schedule))
    if initial.empty:
        initial = pd.DataFrame(columns=["breakpoint_k", "value"])
    edited = st.data_editor(
        initial,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "breakpoint_k": st.column_config.NumberColumn(
                "Breakpoint (k)", min_value=1, step=1,
                help="Sample index (1-based). At sim time t = k * h.",
            ),
            "value": st.column_config.NumberColumn(
                "Setpoint", help=f"{label} value held until next breakpoint",
            ),
        },
        key=key,
    )
    return rows_to_schedule(edited)


def transition_trigger_editor(trigger: dict[str, Any], key_prefix: str) -> dict[str, Any]:
    """Render the time + state-threshold trigger inputs."""
    col1, col2, col3, col4 = st.columns([2, 2, 1, 2])
    max_hours = col1.number_input(
        "max_hours",
        value=float(trigger.get("max_hours") or 0.0),
        min_value=0.0, step=0.5,
        help="Time-in-phase trigger. 0 = no time trigger.",
        key=f"{key_prefix}_max_hours",
    )

    state_var_current = trigger.get("state_var") or "(none)"
    state_var = col2.selectbox(
        "state_var",
        options=_STATE_VAR_OPTIONS,
        index=_STATE_VAR_OPTIONS.index(state_var_current)
              if state_var_current in _STATE_VAR_OPTIONS else 0,
        key=f"{key_prefix}_state_var",
    )
    state_op = col3.selectbox(
        "op", options=(">=", "<="),
        index=0 if (trigger.get("state_op") or ">=") == ">=" else 1,
        disabled=(state_var == "(none)"),
        key=f"{key_prefix}_state_op",
    )
    state_value = col4.number_input(
        "state_value",
        value=float(trigger.get("state_value") or 0.0),
        disabled=(state_var == "(none)"),
        key=f"{key_prefix}_state_value",
    )

    return {
        "max_hours": max_hours if max_hours > 0 else None,
        "state_var": state_var if state_var != "(none)" else None,
        "state_op":  state_op if state_var != "(none)" else None,
        "state_value": state_value if state_var != "(none)" else None,
    }


def phase_editor(phase: dict[str, Any], idx: int) -> dict[str, Any]:
    """Render a complete phase editor block. Returns the edited phase dict."""
    new_name = st.text_input(
        "Phase name", value=phase["name"], key=f"phase_{idx}_name",
    )

    st.markdown("**Transition trigger** — whichever fires first advances the phase")
    new_trigger = transition_trigger_editor(phase["transition"], key_prefix=f"phase_{idx}_trig")

    st.markdown("**Setpoint schedules**")
    new_setpoints: dict[str, Any] = dict(phase["setpoints"])  # copy
    tab_objs = st.tabs(list(SETPOINT_CHANNELS))
    for tab, channel in zip(tab_objs, SETPOINT_CHANNELS):
        with tab:
            new_setpoints[channel] = setpoint_schedule_editor(
                label=channel,
                schedule=phase["setpoints"].get(channel, []),
                key=f"phase_{idx}_sp_{channel}",
            )

    with st.expander("Per-phase T_sp / pH_sp overrides (advanced)"):
        st.caption(
            "When set, these override `ControlFlags.T_sp` / `pH_sp` "
            "for every sample the phase is active; leave at 0 "
            "(displayed as None on the JSON side) to fall back to the "
            "campaign-level setpoint. Temperature is stored in Kelvin "
            "internally but edited here in Celsius."
        )
        col_t, col_ph = st.columns(2)
        stored_k = phase["setpoints"].get("T_sp")
        displayed_c = k_to_c(stored_k) if stored_k is not None else 0.0
        t_sp_c_in = col_t.number_input(
            "T_sp (°C) — leave 0 for None",
            value=float(displayed_c),
            step=0.5, key=f"phase_{idx}_T_sp",
        )
        ph_sp_in = col_ph.number_input(
            "pH_sp — leave 0 for None",
            value=float(phase["setpoints"].get("pH_sp") or 0.0),
            step=0.1, key=f"phase_{idx}_pH_sp",
        )
        new_setpoints["T_sp"] = c_to_k(t_sp_c_in) if t_sp_c_in != 0 else None
        new_setpoints["pH_sp"] = ph_sp_in if ph_sp_in > 0 else None

    return {
        "name": new_name,
        "setpoints": new_setpoints,
        "transition": new_trigger,
    }
