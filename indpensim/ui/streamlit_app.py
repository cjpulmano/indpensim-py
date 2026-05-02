"""Landing page for the indpensim Streamlit UI.

Run with::

    streamlit run indpensim/ui/streamlit_app.py

Streamlit auto-discovers any module under ``indpensim/ui/pages/`` and
adds it to the sidebar. The numeric prefix (``01_``, ``02_``)
controls ordering.
"""
from __future__ import annotations

import json

import streamlit as st

from indpensim.recipe import legacy_sbc_recipe, to_dict
from indpensim.ui.glossary import render_glossary_expander
from indpensim.ui.state import current_recipe, get_recipe_dict, init_session, reset_to_legacy

st.set_page_config(
    page_title="indpensim Recipe Studio",
    page_icon="⚗",
    layout="wide",
)

init_session()
render_glossary_expander()

st.title("indpensim Recipe Studio")
st.markdown(
    """
A Streamlit UI for authoring and visualizing **Recipe** objects that
drive `indpensim`'s industrial-penicillin fermentation simulator.

**Pages** (sidebar):
- **Authoring** — build a Recipe by editing phases, transition triggers,
  and per-channel setpoint schedules. Save / load as JSON.
- **Visualize** — load a Recipe JSON and inspect its setpoint timeline
  with phase boundaries.

The current in-progress Recipe lives in this Streamlit session. The
default seed is `legacy_sbc_recipe()` — the 4-phase Recipe that
reconstitutes the original hardcoded SBC tables bit-for-bit.
"""
)

st.divider()

with st.container():
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.subheader("Current session recipe")
        try:
            recipe = current_recipe()
            st.success(f"Valid: **{recipe.name}** with {len(recipe.phases)} phases — "
                       + " → ".join(p.name for p in recipe.phases))
        except Exception as e:
            st.error(f"Recipe in session is currently invalid: {e}")

        if st.button("Reset to legacy_sbc_recipe()"):
            reset_to_legacy()
            st.rerun()

    with col_r:
        st.subheader("Quick download")
        st.download_button(
            "Download current recipe as JSON",
            data=json.dumps(get_recipe_dict(), indent=2).encode("utf-8"),
            file_name="recipe.json",
            mime="application/json",
        )

st.divider()

with st.expander("How does this connect to the simulator?"):
    st.markdown(
        """
A `Recipe` is consumed by `simulate()` via `BatchConfig.recipe` or
`CapturedBatch.recipe`. When attached, the controller's
`_recipe_lookup` calls are replaced with the executor's per-phase
schedule resolution. Phase context flows into the streaming layer:
each `Sample` carries `phase` / `phase_state`, and the MQTT runner
emits `_phase_start` events at every transition.

```python
from indpensim.driver import BatchConfig, CampaignConfig, batch_spec_from_python_rng
from indpensim.recipe import from_dict
from indpensim.simulation import simulate
import json, numpy as np

with open("recipe.json") as f:
    my_recipe = from_dict(json.load(f))

rng = np.random.default_rng(42)
spec = batch_spec_from_python_rng(
    rng, batch_no=1,
    campaign=CampaignConfig(),
    batch=BatchConfig(recipe=my_recipe),
)
result = simulate(spec)
```
"""
    )
