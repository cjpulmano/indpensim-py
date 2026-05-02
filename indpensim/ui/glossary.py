"""Variable-name glossary for the recipe UI.

Setpoint channel names mirror the MATLAB simulator's `inp1(...)` slots
(see ``docs/state_vector.md``). Not all are self-explanatory — Fpaa,
Foil, Fg, Fdischarge read as jargon unless you know the process. This
module exposes a small reference table that each page can drop into a
sidebar expander.
"""
from __future__ import annotations

import streamlit as st


SETPOINT_GLOSSARY: tuple[tuple[str, str, str], ...] = (
    ("Fs",         "Sugar (substrate) feed rate",                    "L/h"),
    ("Foil",       "Soybean oil feed rate (secondary C source)",     "L/h"),
    ("Fg",         "Gas (air) aeration flow rate",                   "m³/h"),
    ("pressure",   "Vessel head pressure",                           "bar"),
    ("Fpaa",       "Phenoxyacetic acid feed (penicillin precursor)", "L/h"),
    ("Fwater",     "Water-for-injection feed rate",                  "L/h"),
    ("Fdischarge", "Discharge / withdrawal flow (sign applied by "
                   "controller)",                                    "L/h"),
    ("T_sp",       "Temperature setpoint (per-phase override; "
                   "None = keep controller default). Edited in °C in "
                   "the UI; stored as Kelvin internally.",           "°C (display)"),
    ("pH_sp",      "pH setpoint (per-phase override; None = keep "
                   "controller default)",                            "—"),
)


def render_glossary_expander(*, in_sidebar: bool = True, expanded: bool = False) -> None:
    """Render the glossary table inside a Streamlit expander.

    Call at the top of any page. Default places it in the sidebar so it
    stays available across pages.
    """
    target = st.sidebar if in_sidebar else st
    with target.expander("Variable glossary", expanded=expanded):
        st.caption(
            "Setpoint-channel names mirror the MATLAB simulator. Source: "
            "`docs/state_vector.md`."
        )
        # `st.table` renders as static HTML with native text wrapping —
        # `st.dataframe` truncates with ellipsis, which cuts off longer
        # descriptions in the narrow sidebar column.
        rows = [
            {"Name": name, "Description": desc, "Units": units}
            for name, desc, units in SETPOINT_GLOSSARY
        ]
        st.table(rows)
