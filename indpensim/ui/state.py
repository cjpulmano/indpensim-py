"""Session-state helpers for the Streamlit recipe UI.

The single source of truth in session is ``recipe_dict`` — the same
JSON-shape produced by ``indpensim.recipe.io.to_dict``. Every page
reads/writes that dict; conversion to a validated ``Recipe`` happens
on demand via ``current_recipe()`` (raises if the in-progress dict is
malformed, which is what we want — the UI surfaces the validation
error rather than silently accepting bad input).

Why a dict and not a Recipe? Two reasons:
  1. Streamlit re-runs the script top-to-bottom on every interaction.
     Mutable nested dicts survive ``st.data_editor`` edits cleanly;
     frozen dataclasses don't.
  2. Authoring an in-progress recipe naturally goes through invalid
     intermediate states (empty schedule, half-typed name, no trigger
     yet). Storing as dict postpones validation to the moment of use.
"""
from __future__ import annotations

from typing import Any

import streamlit as st

from indpensim.recipe import (
    Recipe,
    SetpointProfile,
    from_dict,
    legacy_sbc_recipe,
    to_dict,
)


_STATE_KEY = "recipe_dict"


def init_session() -> None:
    """Idempotent — call at the top of every page."""
    if _STATE_KEY not in st.session_state:
        st.session_state[_STATE_KEY] = to_dict(legacy_sbc_recipe())


def get_recipe_dict() -> dict[str, Any]:
    init_session()
    return st.session_state[_STATE_KEY]


def set_recipe_dict(d: dict[str, Any]) -> None:
    st.session_state[_STATE_KEY] = d


def reset_to_legacy() -> None:
    set_recipe_dict(to_dict(legacy_sbc_recipe()))


def current_recipe() -> Recipe:
    """Validate-and-return the in-progress recipe. Raises on malformed."""
    return from_dict(get_recipe_dict())


# ---------------------------------------------------------------------------
# Pure helpers (no streamlit dependency at call time — testable)
# ---------------------------------------------------------------------------

# Channel names exposed in the UI; order matters (display order).
SETPOINT_CHANNELS: tuple[str, ...] = (
    "Fs", "Foil", "Fg", "pressure", "Fpaa", "Fwater", "Fdischarge",
)


def empty_phase(name: str = "NEW_PHASE") -> dict[str, Any]:
    """Build a blank phase dict suitable for appending to the recipe."""
    return {
        "name": name,
        "setpoints": {ch: [] for ch in SETPOINT_CHANNELS} | {
            "T_sp": None, "pH_sp": None,
        },
        "transition": {
            "max_hours": 1.0,
            "state_var": None,
            "state_op": None,
            "state_value": None,
        },
    }


def schedule_to_rows(schedule: list) -> list[dict[str, float]]:
    """Convert [[bp, sp], ...] to st.data_editor row dicts."""
    return [{"breakpoint_k": float(bp), "value": float(sp)} for bp, sp in schedule]


def rows_to_schedule(rows) -> list[list[float]]:
    """Inverse of schedule_to_rows. Tolerates pandas DataFrame or list of dicts."""
    if hasattr(rows, "iterrows"):
        return [
            [float(r["breakpoint_k"]), float(r["value"])]
            for _, r in rows.iterrows()
            if r["breakpoint_k"] is not None and r["value"] is not None
        ]
    return [
        [float(r["breakpoint_k"]), float(r["value"])]
        for r in rows
        if r.get("breakpoint_k") is not None and r.get("value") is not None
    ]
