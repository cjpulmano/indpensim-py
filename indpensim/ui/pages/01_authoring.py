"""Recipe authoring page.

Form-based CRUD for the in-progress Recipe in session state. Each
phase is editable inline; Add / Remove / Move-up / Move-down operate
on the recipe's phase list. Save downloads JSON; Load accepts an
uploaded JSON file.

The page validates on every interaction by attempting
``current_recipe()``. Errors render inline so authoring can proceed
through invalid intermediate states (empty schedule, missing trigger,
etc.) without losing work.
"""
from __future__ import annotations

import json

import streamlit as st

from indpensim.recipe import from_dict
from indpensim.ui.state import (
    current_recipe,
    empty_phase,
    get_recipe_dict,
    init_session,
    reset_to_legacy,
    set_recipe_dict,
)
from indpensim.ui.widgets import phase_editor

st.set_page_config(page_title="Authoring | indpensim", layout="wide")
init_session()

st.title("Recipe authoring")

# ---------------------------------------------------------------------------
# Top toolbar — load / save / reset
# ---------------------------------------------------------------------------

with st.container():
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Reset to legacy SBC", use_container_width=True):
            reset_to_legacy()
            st.rerun()

    with col2:
        st.download_button(
            "Download as JSON",
            data=json.dumps(get_recipe_dict(), indent=2).encode("utf-8"),
            file_name=f"{get_recipe_dict().get('name', 'recipe')}.json",
            mime="application/json",
            use_container_width=True,
        )

    with col3:
        uploaded = st.file_uploader("Load from JSON", type=["json"])
        if uploaded is not None:
            try:
                loaded = json.loads(uploaded.read())
                from_dict(loaded)            # validate
                set_recipe_dict(loaded)
                st.success(f"Loaded: {loaded.get('name', '?')}")
            except Exception as e:
                st.error(f"Could not load: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Recipe-level editing
# ---------------------------------------------------------------------------

recipe_dict = get_recipe_dict()

new_name = st.text_input("Recipe name", value=recipe_dict.get("name", "untitled"))
recipe_dict["name"] = new_name

# ---------------------------------------------------------------------------
# Validation banner
# ---------------------------------------------------------------------------

try:
    recipe = current_recipe()
    st.success(f"Valid recipe — {len(recipe.phases)} phases")
except Exception as e:
    st.error(f"Recipe is currently invalid: {e}")

st.divider()

# ---------------------------------------------------------------------------
# Per-phase editing
# ---------------------------------------------------------------------------

st.subheader("Phases")

phases = recipe_dict.get("phases", [])
if not phases:
    st.info("This recipe has no phases. Add one below.")
else:
    for idx, phase in enumerate(phases):
        with st.expander(f"Phase {idx + 1}: {phase['name']}", expanded=(idx == 0)):
            edited = phase_editor(phase, idx)
            phases[idx] = edited

            ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1, 1, 1, 4])
            if ctrl1.button("↑ Up", key=f"up_{idx}", disabled=(idx == 0)):
                phases[idx - 1], phases[idx] = phases[idx], phases[idx - 1]
                st.rerun()
            if ctrl2.button("↓ Down", key=f"dn_{idx}", disabled=(idx == len(phases) - 1)):
                phases[idx], phases[idx + 1] = phases[idx + 1], phases[idx]
                st.rerun()
            if ctrl3.button("✕ Remove", key=f"rm_{idx}"):
                phases.pop(idx)
                st.rerun()

recipe_dict["phases"] = phases
set_recipe_dict(recipe_dict)

st.divider()

if st.button("➕ Add new phase", type="primary"):
    next_idx = len(phases) + 1
    phases.append(empty_phase(name=f"PHASE_{next_idx}"))
    recipe_dict["phases"] = phases
    set_recipe_dict(recipe_dict)
    st.rerun()
