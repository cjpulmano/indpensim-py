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


# Process-state channels — the things the simulator observes. Shown on
# the Run page's trajectory plots; named here so engineers can decode
# the legend.
STATE_GLOSSARY: tuple[tuple[str, str, str], ...] = (
    ("P",   "Penicillin concentration — the product. Legacy recipe "
            "reaches ~10-15 g/L by 230 h.",                          "g/L"),
    ("X",   "Biomass concentration (P. chrysogenum cells). Grows "
            "fast in GROWTH, levels off in PRODUCTION when cells "
            "spend energy on P instead of replicating. Typical end: "
            "~30-40 g/L.",                                           "g/L"),
    ("S",   "Substrate (sugar) concentration in the broth. Driven "
            "down by uptake, replenished by Fs feed.",               "g/L"),
    ("DO2", "Dissolved oxygen. Closed-loop with Fg/RPM; falls "
            "during high biomass / high feed.",                      "mg/L"),
    ("pH",  "Broth pH. Controlled to ~6.5 by acid (Fa) and base "
            "(Fb) flows.",                                           "—"),
    ("T",   "Broth temperature. Controlled to ~298 K (25 °C) by "
            "cooling (Fc) and heating (Fh) flows.",                  "K"),
    ("V",   "Vessel liquid volume. Grows from feeds, drops on "
            "discharge events.",                                     "L"),
    ("PAA", "Phenoxyacetic acid concentration in the broth — the "
            "penicillin precursor. Topped up by Fpaa.",              "mg/L"),
    ("NH3", "Ammonia / nitrogen source concentration.",              "mg/L"),
    ("Wt",  "Total broth mass.",                                     "kg"),
)


def render_glossary_expander(*, in_sidebar: bool = True, expanded: bool = False) -> None:
    """Render the glossary tables inside a Streamlit expander.

    Call at the top of any page. Default places it in the sidebar so it
    stays available across pages.
    """
    target = st.sidebar if in_sidebar else st
    with target.expander("Variable glossary", expanded=expanded):
        st.caption(
            "Names mirror the MATLAB simulator. Source: "
            "`docs/state_vector.md`."
        )
        # `st.table` renders as static HTML with native text wrapping —
        # `st.dataframe` truncates with ellipsis, which cuts off longer
        # descriptions in the narrow sidebar column.
        st.markdown("**Setpoints** (what the controller drives)")
        st.table([
            {"Name": name, "Description": desc, "Units": units}
            for name, desc, units in SETPOINT_GLOSSARY
        ])
        st.markdown("**State variables** (what the simulator observes)")
        st.table([
            {"Name": name, "Description": desc, "Units": units}
            for name, desc, units in STATE_GLOSSARY
        ])
