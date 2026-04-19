"""Pre-allocated trajectory history container — Python analog of MATLAB's
``X`` struct used by ``fctrl_indpensim`` and ``indpensim``.

Each named channel holds a length-(N+1) ``np.ndarray`` indexed 1-based to
mirror MATLAB exactly. Slot 0 is unused (or holds the initial condition
where MATLAB's ``X.<field>.y(1)`` is initialized from ``x0`` before the
first ``fctrl`` call). Index ``k`` corresponds to MATLAB sample ``k``.

The controller treats state channels (S, DO2, X, P, V, ..., pH, T, PAA,
PAA_pred) as read-only and writes to its own output channels (Fa, Fb, Fc,
Fh, Fs, Fpaa, Fg, Foil, RPM, Fw, pressure, Fremoved, Fault_ref) plus a
single dedicated side-effect channel (``PRBS_noise_addition``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# All 33 ODE state channels + the derived `X` total-biomass channel.
# Order matches indpensim.m:248-323 (X.<field>.y(k) = y_sol(end, n)).
_STATE_CHANNELS: tuple[str, ...] = (
    "S", "DO2", "O2", "P", "V", "Wt", "pH", "T",
    "Q", "Viscosity", "Culture_age",
    "a0", "a1", "a3", "a4",
    "n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9",
    "nm", "phi0",
    "CO2outgas", "CO2_d",
    "PAA", "NH3",
    "mu_P_calc", "mu_X_calc",
    "X",                              # derived: a0+a1+a3+a4
    "PAA_pred",                       # written by Substrate_prediction (Raman_spec=2)
    "OUR", "CER",                     # derived per indpensim.m:335-339
)

# Channels the controller WRITES to (its own outputs, used as u_{k-1} on the next call).
_CONTROL_CHANNELS: tuple[str, ...] = (
    "Fa", "Fb", "Fc", "Fh", "Fs", "Fpaa", "Fg", "Foil",
    "RPM", "Fw", "pressure", "Fremoved", "Fault_ref",
    "viscosity",
)


@dataclass
class BatchHistory:
    """Mutable per-channel arrays for the simulation horizon."""
    N: int                                       # number of samples (T/h, integer)
    channels: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def empty(cls, N: int) -> "BatchHistory":
        h = cls(N=N)
        for name in _STATE_CHANNELS + _CONTROL_CHANNELS:
            h.channels[name] = np.zeros(N + 1)
        # PRBS noise is the controller's only mutable side-effect channel
        h.channels["PRBS_noise_addition"] = np.zeros(N + 1)
        return h

    def y(self, name: str, k: int) -> float:
        """1-based MATLAB-style read: ``history.y('pH', k)`` ↔ ``X.pH.y(k)``."""
        return float(self.channels[name][k])

    def set(self, name: str, k: int, value: float) -> None:
        """1-based MATLAB-style write."""
        self.channels[name][k] = value


@dataclass(frozen=True)
class IndustrialData:
    """Stub for ``Xd`` — only ``NH3_shots`` is read (and only when always-zero).

    Full SBC=1 support would carry per-sample arrays for Foil, F_discharge_cal,
    pressure, Fpaa, Fw, viscosity, Fg, Fs. Captured batch 1 uses SBC=0, so
    we defer that until needed. Returns 0 for any field at any sample.
    """

    def NH3_shots(self, k: int) -> float:  # noqa: N802 — match MATLAB field name
        return 0.0
