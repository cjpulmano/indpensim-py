"""33-element state-vector schema for the IndPenSim ODE.

See docs/state_vector.md for the canonical glossary including units and
porting gotchas.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np


class S(IntEnum):
    """0-based indices into the 33-element state vector."""
    SUBSTRATE = 0          # g/L     was MATLAB y(1)  / X.S
    DO2 = 1                # mg/L                y(2)  / X.DO2
    O2_OFF = 2             # mole frac           y(3)  / X.O2
    PENICILLIN = 3         # g/L                 y(4)  / X.P
    VOLUME = 4             # L                   y(5)  / X.V
    WEIGHT = 5             # kg                  y(6)  / X.Wt
    H_PLUS = 6             # mol/L (NOT pH)      y(7)  / X.pH (post-ODE converted to pH)
    TEMPERATURE = 7        # K                   y(8)  / X.T
    Q_HEAT = 8             # kcal cumulative     y(9)  / X.Q
    VISCOSITY = 9          # cP                  y(10) / X.Viscosity
    INTEGRAL_X = 10        # g·h/L               y(11) / X.Culture_age
    A0 = 11                # g/L  growing        y(12) / X.a0
    A1 = 12                # g/L  non-growing    y(13) / X.a1
    A3 = 13                # g/L  degenerated    y(14) / X.a3
    A4 = 14                # g/L  autolysed      y(15) / X.a4
    # Vacuole interior bins n_0..n_9 → indices 15..24 (use VAC_BIN_SLICE)
    N_VAC_MAX = 25         #                     y(26) / X.nm
    PHI_0 = 26             #                     y(27) / X.phi0
    CO2_OFF = 27           # %                   y(28) / X.CO2outgas
    CO2_DISSOLVED = 28     # g/L                 y(29) / X.CO2_d
    PAA = 29               # mg/L                y(30) / X.PAA
    NH3 = 30               # mg/L                y(31) / X.NH3
    INTEGRAL_MU_P = 31     # ∫mu_p dt            y(32) / X.mu_P_calc
    INTEGRAL_MU_E = 32     # ∫mu_e dt            y(33) / X.mu_X_calc


N_STATES: int = 33
N_VAC_BINS: int = 10
VAC_BIN_SLICE: slice = slice(15, 25)   # indices 15..24 inclusive


def initial_state() -> np.ndarray:
    """Return a 33-element zero-initialised state vector (float64)."""
    return np.zeros(N_STATES, dtype=np.float64)
