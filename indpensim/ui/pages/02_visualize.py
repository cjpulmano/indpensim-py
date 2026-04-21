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
from indpensim.ui.rendering import build_recipe_timeline_figure, phase_summary_rows
from indpensim.ui.state import current_recipe, init_session

st.set_page_config(page_title="Visualize | indpensim", layout="wide")
init_session()

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

h = st.number_input(
    "Sample period h (hours)",
    value=0.2, min_value=0.01, step=0.05,
    help="Used to compute phase boundaries from time-in-phase triggers. "
         "0.2h matches the simulator default.",
)

fig = build_recipe_timeline_figure(recipe, h=h)
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Phase summary")
rows = phase_summary_rows(recipe, h=h)
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
