"""Run a single batch from the studio's authored recipe.

Inline-blocking simulate() with a spinner. Two IC sources:
  - Captured (bundled MATLAB reference batch — deterministic)
  - Random (fresh init from numpy with a user-chosen seed)

The completed SimulationResult is parked in session state so the user
can rerun, change settings, and compare without losing the previous
trajectory.
"""
from __future__ import annotations

import streamlit as st

from indpensim.simulation import simulate
from indpensim.ui.glossary import render_glossary_expander
from indpensim.ui.run import (
    RunSpec,
    captured_seed_choices,
    build_spec,
    summarize,
    trajectory_dataframe,
)
from indpensim.ui.state import current_recipe, init_session


st.set_page_config(page_title="Run | indpensim", layout="wide")
init_session()
render_glossary_expander()

st.title("Run a batch")
st.caption(
    "Runs the authored recipe through the full simulator (controller "
    "+ ODE) and shows the resulting trajectory. Inline blocking — the "
    "page freezes for ~2-3 seconds while the batch integrates."
)

try:
    recipe = current_recipe()
except Exception as e:                        # noqa: BLE001
    st.error(f"Recipe is invalid — fix it on the Authoring page first.\n\n{e}")
    st.stop()

st.subheader(f"Recipe: {recipe.name}  ({len(recipe.phases)} phase(s))")

# ---------------------------------------------------------------------------
# IC source picker
# ---------------------------------------------------------------------------
col_src, col_seed = st.columns([2, 2])
with col_src:
    ic_source_label = st.radio(
        "Initial conditions",
        options=("Captured reference", "Random (Python RNG)"),
        horizontal=False,
        help=(
            "Captured: load a bundled MATLAB reference batch's initial "
            "state. Deterministic, validation-flavored.\n\n"
            "Random: draw a fresh initial state from numpy. "
            "Production-flavored — different starting biomass, "
            "substrate, pH each seed."
        ),
    )
    ic_source = "captured" if ic_source_label.startswith("Captured") else "python_rng"

with col_seed:
    if ic_source == "captured":
        seeds = captured_seed_choices() or [42]
        captured_seed = st.selectbox("Captured seed", options=seeds, index=0)
        captured_batch = st.number_input("Batch index", min_value=1, max_value=2,
                                         value=1, step=1)
        python_seed = 42
    else:
        captured_seed = 42
        captured_batch = 1
        python_seed = st.number_input("Random seed", min_value=0, value=42, step=1,
                                       help="Same seed → same initial state.")

st.divider()

# ---------------------------------------------------------------------------
# Run button
# ---------------------------------------------------------------------------
if st.button("Run batch", type="primary"):
    run_spec = RunSpec(
        recipe=recipe,
        ic_source=ic_source,
        captured_seed=int(captured_seed),
        captured_batch=int(captured_batch),
        python_seed=int(python_seed),
    )
    try:
        cap = build_spec(run_spec)
    except FileNotFoundError as e:
        st.error(f"Could not load capture: {e}")
        st.stop()

    with st.spinner(f"Simulating {cap.T} h batch …"):
        result = simulate(cap)

    # Stash for the post-run views; re-running just overwrites.
    st.session_state["last_run_result"] = result
    st.session_state["last_run_spec"] = run_spec
    st.session_state["last_run_n_transitions"] = 0  # phase log is not yet
                                                     # surfaced by simulate()

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
result = st.session_state.get("last_run_result")
last_spec: RunSpec | None = st.session_state.get("last_run_spec")

if result is None or last_spec is None:
    st.info("Click **Run batch** to simulate the recipe.")
else:
    st.subheader("Results")
    st.caption(
        f"IC source: **{last_spec.ic_source}** · "
        + (f"captured seed {last_spec.captured_seed}, batch {last_spec.captured_batch}"
           if last_spec.ic_source == "captured"
           else f"random seed {last_spec.python_seed}")
    )

    summary = summarize(result, st.session_state.get("last_run_n_transitions", 0))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Duration",   f"{summary.duration_h:.1f} h")
    m2.metric("Final P",    f"{summary.P_final_g_per_L:.2f} g/L")
    m3.metric("Final X",    f"{summary.X_final_g_per_L:.2f} g/L")
    m4.metric("Final V",    f"{summary.V_final_L:,.0f} L")

    # Channel plots — small hand-picked set covering biology + control.
    DEFAULT_CHANNELS = ("P", "X", "S", "DO2", "pH", "T")
    chosen = st.multiselect(
        "Channels to plot",
        options=("P", "X", "S", "DO2", "pH", "T", "PAA", "NH3", "V", "Wt"),
        default=list(DEFAULT_CHANNELS),
    )
    if chosen:
        df = trajectory_dataframe(result, tuple(chosen))
        df = df.set_index("time_h")
        st.line_chart(df, height=420, use_container_width=True)
