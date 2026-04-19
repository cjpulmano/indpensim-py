"""End-to-end batch simulation — port of indpensim.m main loop.

Drives the ODE integration sample-by-sample using the Python controller.
Mirrors `indpensim.m:107-379` exactly (per-step floors, DO2 stability rule,
inhibition mu_X tweak, OUR/CER derived calcs, post-loop pH and Q
conversions).

Differences from playback.py:
  - playback feeds MATLAB's controller outputs (CSV) at each step;
  - simulate calls the Python controller and uses ITS outputs.
  - Otherwise: same solver (BDF), same per-step floors, same conversions.

What's NOT yet implemented:
  - Off-line measurements: skipped (recording-only, no feedback).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from indpensim.control.controller import controller_step
from indpensim.control.history import BatchHistory, IndustrialData
from indpensim.io.initial_conditions import CapturedBatch, ControlFlags
from indpensim.io.parameters import Parameters
from indpensim.ode.rhs import N_STATES, rhs
from indpensim.pat.pls_model import PAAPLSModel
from indpensim.pat.raman import build_reference, simulate_spectrum
from indpensim.pat.substrate import predict_and_store

_REFERENCE_SPECTRA_PATH = Path(__file__).resolve().parents[1] / "reference_Specra.txt"


# Map state index → BatchHistory channel name (for post-step storage).
# Mirrors indpensim.m:248-320.
_STATE_INDEX_TO_CHANNEL: tuple[str, ...] = (
    "S", "DO2", "O2", "P", "V", "Wt", "pH", "T",            # 0-7
    "Q", "Viscosity", "Culture_age",                          # 8-10
    "a0", "a1", "a3", "a4",                                   # 11-14
    "n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9",  # 15-24
    "nm", "phi0",                                             # 25-26
    "CO2outgas", "CO2_d",                                     # 27-28
    "PAA", "NH3",                                             # 29-30
    "mu_P_calc", "mu_X_calc",                                 # 31-32
)


@dataclass
class SimulationResult:
    t: np.ndarray                  # shape (N+1,)
    history: BatchHistory          # all channels, MATLAB 1-based indexing
    states: np.ndarray             # shape (N+1, 33), endpoint state per sample
    pH_trajectory: np.ndarray      # shape (N+1,), -log10(y[6])
    raman_spectra: np.ndarray | None = None
    """Raman intensities, shape (N+1, 2200), MATLAB 1-based; None if Raman_spec=0
    or k <= 10 (MATLAB only computes spectra for k > 10)."""


def _build_initial_state(cap: CapturedBatch) -> np.ndarray:
    """Mirror of indpensim.m:131 (the k==1 x00 vector)."""
    ic = cap.initial_conditions
    y = np.zeros(N_STATES)
    y[0] = ic.S
    y[1] = ic.DO2
    y[2] = ic.O2
    y[3] = ic.P
    y[4] = ic.V
    y[5] = ic.Wt
    y[6] = 10.0 ** (-ic.pH)        # pH → [H+] (mirror of indpensim.m:105)
    y[7] = ic.T
    y[8] = 0.0                      # Q
    y[9] = 4.0                      # Viscosity init literal
    y[10] = ic.Culture_age
    y[11] = ic.a0
    y[12] = ic.a1
    y[13] = ic.a3
    y[14] = ic.a4
    # vacuoles n0..n9 + nm + phi0 = zeros(1,12)
    y[27] = ic.CO2outgas
    y[28] = 0.0                     # CO2_d
    y[29] = ic.PAA
    y[30] = ic.NH3
    y[31] = 0.0                     # mu_P_calc
    y[32] = 0.0                     # mu_X_calc
    return y


def _apply_matlab_floors(y: np.ndarray) -> None:
    """Mirror of indpensim.m:216-220 and :252-257."""
    mask = y[:31] <= 0.0
    y[:31][mask] = 0.001
    if y[1] < 2.0:
        y[1] = 1.0


def _build_inputs(u, cap: CapturedBatch, k: int, h_ode: float) -> np.ndarray:
    """Build the 26-element controller-input vector for `rhs`.

    Mirror of indpensim.m:178. Pulls disturbances from the captured trajectory.
    """
    dist = cap.disturbances.at(k - 1)    # disturbances are 0-indexed by sample
    inp = np.zeros(26)
    inp[0]  = cap.control_flags.Inhib
    inp[1]  = u.Fs
    inp[2]  = u.Fg
    inp[3]  = u.RPM
    inp[4]  = u.Fc
    inp[5]  = u.Fh
    inp[6]  = u.Fb
    inp[7]  = u.Fa
    inp[8]  = h_ode
    inp[9]  = u.Fw
    inp[10] = u.pressure
    inp[11] = u.viscosity
    inp[12] = u.Fremoved
    inp[13] = u.Fpaa
    inp[14] = u.Foil
    inp[15] = u.NH3_shots
    inp[16] = cap.control_flags.Dis
    inp[17:25] = dist
    inp[25] = cap.control_flags.Vis
    return inp


def _calc_OUR_CER(history: BatchHistory, k: int) -> tuple[float, float]:
    """Mirror of indpensim.m:335-339."""
    O2_in = 0.204
    Fg = history.y("Fg", k)
    O2 = history.y("O2", k)
    CO2outgas = history.y("CO2outgas", k)
    OUR = (32 * Fg / 22.4) * (O2_in - O2 * (0.7902 / (1 - O2 - CO2outgas / 100)))
    CER = (44 * Fg / 22.4) * ((0.65 * CO2outgas / 100)
                              * (0.7902 / (1 - O2_in - CO2outgas / 100) - 0.0330))
    return float(OUR), float(CER)


def simulate(
    cap: CapturedBatch,
    *,
    ctrl_flags: ControlFlags | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    raman_rng: np.random.Generator | None = None,
    raman_noise_traj: np.ndarray | None = None,
) -> SimulationResult:
    """Run a full batch through the Python controller + ODE.

    Args:
        cap: CapturedBatch from `load_captured_batch(...)` — supplies x0,
            randomized parameters, disturbance trajectories, and (by default)
            control flags.
        ctrl_flags: override `cap.control_flags` (e.g. to disable PRBS for
            reproducibility). Defaults to the captured flags.
        rtol/atol: solver tolerances (defaults match playback).
        raman_rng: rng for stochastic Raman noise. Default
            ``np.random.default_rng(0)`` if needed (Raman_spec >= 1).
        raman_noise_traj: optional (N+1, 2200) array of pre-captured noise
            vectors (one per sample). Overrides ``raman_rng`` when set.
    """
    if ctrl_flags is None:
        ctrl_flags = cap.control_flags

    h = cap.h
    T = cap.T
    N = int(round(T / h))
    h_ode = h / 20.0
    t_grid = np.linspace(0.0, N * h, N + 1)

    # Parameter vector (mutable: inhibition logic at indpensim.m:182-201
    # rewrites par[1] in place).
    par = Parameters.default(
        mu_p=cap.initial_conditions.mup,
        mux_max=cap.initial_conditions.mux,
        alpha_kla=cap.initial_conditions.alpha_kla,
        N_conc_paa=cap.initial_conditions.N_conc_paa,
        PAA_c=cap.initial_conditions.PAA_c,
    ).to_legacy_par_vector().copy()

    history = BatchHistory.empty(N)
    states = np.zeros((N + 1, N_STATES))

    # Initial state (slot 0 = t=0, mirroring our Python convention)
    y_curr = _build_initial_state(cap)
    states[0] = y_curr.copy()

    # Pre-fill k=1 history slots that fctrl reads at first call (indpensim.m:113-122).
    # These will be OVERWRITTEN with integrated values after the first ODE solve.
    history.set("S", 1, cap.initial_conditions.S)
    history.set("DO2", 1, cap.initial_conditions.DO2)
    history.set("X", 1, cap.initial_conditions.X)
    history.set("P", 1, cap.initial_conditions.P)
    history.set("V", 1, cap.initial_conditions.V)
    history.set("CO2outgas", 1, cap.initial_conditions.CO2outgas)
    history.set("pH", 1, 10.0 ** (-cap.initial_conditions.pH))
    history.set("T", 1, cap.initial_conditions.T)

    Xd = IndustrialData()

    # ---- Raman pipeline setup (only if enabled)
    raman_active = ctrl_flags.Raman_spec >= 1
    if raman_active:
        raman_ref = build_reference(_REFERENCE_SPECTRA_PATH)
        raman_spectra = np.zeros((N + 1, 2200))
        if raman_rng is None:
            raman_rng = np.random.default_rng(0)
    else:
        raman_ref = None
        raman_spectra = None
    if ctrl_flags.Raman_spec == 2:
        pls = PAAPLSModel.load()
    else:
        pls = None

    for k in range(1, N + 1):
        # ---- 1. Inhibition: degrade mu_X if 63+ of last 64 diffs are negative
        # (indpensim.m:192-198). Faithful port — including the *5 factor that
        # contradicts the comment ("current minimum value").
        if ctrl_flags.Inhib in (1, 2) and k > 65:
            mu_X_window = history.channels["mu_X_calc"][k - 65 : k]   # 65 elements
            d = np.diff(mu_X_window)                                   # 64 elements
            if int(np.sum(d < 0)) >= 63:
                par[1] = history.y("mu_X_calc", k - 1) * 5.0

        # ---- 2. Controller call
        u = controller_step(history, Xd, k, h, T, ctrl_flags)

        # Reset per-sample mu_P_calc / mu_X_calc accumulators
        # (indpensim.m treats them as per-step increments by including 0,0
        # as the last two entries in x00 every iteration; line 131 + 164-165).
        y_curr[31] = 0.0
        y_curr[32] = 0.0

        # ---- 3. Build inputs and integrate
        inp = _build_inputs(u, cap, k, h_ode)
        sol = solve_ivp(
            rhs, [t_grid[k - 1], t_grid[k]], y_curr,
            args=(inp, par), method="BDF",
            rtol=rtol, atol=atol,
            t_eval=[t_grid[k]],
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed at sample {k}: {sol.message}")
        y_curr = sol.y[:, -1]
        _apply_matlab_floors(y_curr)
        states[k] = y_curr

        # ---- 4. Save controller outputs to history (so next iteration's
        # u_{k-1} args are correct).
        history.set("Fa", k, u.Fa)
        history.set("Fb", k, u.Fb)
        history.set("Fc", k, u.Fc)
        history.set("Fh", k, u.Fh)
        history.set("Fs", k, u.Fs)
        history.set("Fpaa", k, u.Fpaa)
        history.set("Fg", k, u.Fg)
        history.set("Foil", k, u.Foil)
        history.set("RPM", k, u.RPM)
        history.set("Fw", k, u.Fw)
        history.set("pressure", k, u.pressure)
        history.set("Fremoved", k, u.Fremoved)
        history.set("Fault_ref", k, u.Fault_ref)
        history.set("viscosity", k, u.viscosity)

        # ---- 5. Save all 33 state values to history (indpensim.m:248-320)
        for idx, channel in enumerate(_STATE_INDEX_TO_CHANNEL):
            history.set(channel, k, float(y_curr[idx]))

        # Total biomass X = a0 + a1 + a3 + a4 (line 322)
        history.set("X", k,
                    history.y("a0", k) + history.y("a1", k)
                    + history.y("a3", k) + history.y("a4", k))

        # ---- 6. OUR / CER (lines 335-339)
        OUR, CER = _calc_OUR_CER(history, k)
        history.set("OUR", k, OUR)
        history.set("CER", k, CER)

        # ---- 7. Raman (lines 341-348). MATLAB starts spectra at k > 10.
        if raman_active and k > 10:
            noise_k = (raman_noise_traj[k] if raman_noise_traj is not None else None)
            spec = simulate_spectrum(
                reference=raman_ref,
                P=history.y("P", k),
                X=history.y("X", k),
                viscosity=history.y("Viscosity", k),
                S=history.y("S", k),
                PAA=history.y("PAA", k),
                k=k, N_samples=N,
                rng=raman_rng, noise=noise_k,
            )
            raman_spectra[k] = spec

            # Raman_spec=2 closes the PAA control loop via Substrate_prediction.
            # Note j = k - 1 lag (Substrate_prediction.m:4) — predict from
            # PREVIOUS sample's spectrum and store at slot j = k - 1.
            if ctrl_flags.Raman_spec == 2 and k >= 2:
                j = k - 1
                spec_at_j = raman_spectra[j]
                # spec_at_j may be all zeros if j <= 10 — predict_and_store
                # would then produce a meaningless prediction. Guard:
                if j > 10:
                    predict_and_store(
                        pls=pls, spectrum_at_j=spec_at_j,
                        paa_pred_history=history.channels["PAA_pred"], j=j,
                    )

        # ---- 8. Off-line measurements (lines 352-375): skipped (recording only)

    # ---- Post-loop unit conversions (indpensim.m:382-383)
    pH_traj = -np.log10(states[:, 6])
    states[:, 8] = states[:, 8] / 1000.0
    history.channels["pH"][:] = -np.log10(np.maximum(history.channels["pH"][:], 1e-30))
    history.channels["Q"][:] = history.channels["Q"][:] / 1000.0

    return SimulationResult(t=t_grid, history=history, states=states,
                            pH_trajectory=pH_traj, raman_spectra=raman_spectra)
