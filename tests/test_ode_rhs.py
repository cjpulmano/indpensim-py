"""Smoke + sanity tests for the ODE right-hand side.

We can't bit-compare against MATLAB yet because x0/alpha_kla/PAA_c/N_conc_paa
are randomized inside indpensim_run.m and not exposed by the dump. Strong
trajectory validation arrives in Phase 3 once the full simulation pipeline
runs in Python with the same random sequence (or by extending the dump
script to capture x0).

What these tests CAN catch:
  - Crashes / NaN propagation
  - The y[6] mutation gotcha (verified by calling rhs twice with same y)
  - Wrong array length (33 vs 31)
  - Inhibition flag dispatch errors
  - Sign errors on slow-varying states (FD comparison vs MATLAB trajectory)
  - Order-of-magnitude errors
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from indpensim.io.parameters import Parameters
from indpensim.ode.rhs import N_STATES, rhs


REF_DIR = Path(__file__).resolve().parents[1] / "data" / "matlab_reference"
STATES_CSV = REF_DIR / "batch_seed42_b01_states.csv"
INITCONDS_MAT = REF_DIR / "batch_seed42_b01_initconds.mat"


def _sane_state(*, pH=6.5, pen=0.5, temp=298.0, vol=58000.0):
    """Build a 33-element y vector with mid-batch-ish values that won't divide-by-zero."""
    y = np.zeros(N_STATES)
    y[0]  = 5.0      # S g/L
    y[1]  = 12.0     # DO2 mg/L
    y[2]  = 0.18     # O2 off-gas frac
    y[3]  = pen      # P g/L
    y[4]  = vol      # V L
    y[5]  = vol * 1.06    # Wt kg
    y[6]  = 10.0 ** -pH   # [H+] mol/L  (NOT pH directly — see gotcha #2)
    y[7]  = temp     # T K
    y[8]  = 1000.0   # Q kJ
    y[9]  = 30.0     # viscosity cP
    y[10] = 100.0    # ∫X dt
    y[11] = 1.5      # a0 g/L
    y[12] = 8.0      # a1 g/L
    y[13] = 0.5      # a3 g/L
    y[14] = 0.05     # a4 g/L
    # Vacuole bins n_0..n_9 and n_max — small positive
    y[15:25] = np.array([1.0e6, 5.0e4, 8.0e2, 8.0, 0.05,
                          5e-4, 5e-6, 5e-8, 5e-10, 5e-12])
    y[25] = 1e-13    # n_max
    y[26] = 1e-5     # phi0
    y[27] = 0.05     # CO2 off-gas %
    y[28] = 0.2      # CO2_d g/L
    y[29] = 1500.0   # PAA mg/L
    y[30] = 2000.0   # NH3 mg/L
    y[31] = 0.0      # ∫mu_p dt (irrelevant for rhs output)
    y[32] = 0.0      # ∫mu_e dt
    return y


def _sane_inputs(inhib=2, dist_flag=0, vis_flag=0):
    """26-element inp1 vector — mid-batch typical values, no disturbances by default.

    NOTE: Fh and Fc must be strictly positive. The temperature ODE divides by
    `Fh/1000 + alpha_1*(Fh/1000)^beta_T/2*pho_b*C_ps`, which is 0 at Fh=0.
    The MATLAB controller never sends exactly zero (deadband + pump physics);
    typical values are 1e-4 .. 30 L/h. We mirror that.
    """
    inp = np.zeros(26)
    inp[0]  = inhib
    inp[1]  = 8.0      # Fs L/h
    inp[2]  = 30.0     # Fg m^3/h (gets /60 inside rhs)
    inp[3]  = 100.0    # RPM
    inp[4]  = 16.0     # Fc L/h
    inp[5]  = 1e-4     # Fh L/h (controller never sends exactly zero — see docstring)
    inp[6]  = 0.5      # Fb
    inp[7]  = 1e-4     # Fa
    inp[8]  = 0.01     # h_ode step
    inp[9]  = 22.0     # Fw L/h
    inp[10] = 0.6      # head pressure bar
    inp[11] = 30.0     # recorded viscosity (used only when vis_flag=1)
    inp[12] = 0.0      # F_discharge (Fremoved); negative = harvest, 0 = no discharge
    inp[13] = 5.0      # Fpaa L/h
    inp[14] = 1e-4     # Foil L/h
    inp[15] = 0.0      # NH3 shots
    inp[16] = dist_flag
    # disturbance values 17..24 = 0 by default
    inp[25] = vis_flag
    return inp


def _params():
    return Parameters.default(
        mu_p=0.041, mux_max=0.41, alpha_kla=85.0,
        N_conc_paa=150000.0, PAA_c=530000.0,
    ).to_legacy_par_vector()


# =============================================================================
# Smoke tests
# =============================================================================

@pytest.mark.parametrize("inhib_flag", [0, 1, 2])
def test_rhs_returns_33_finite_floats(inhib_flag):
    """Every inhibition mode produces a 33-element finite dy vector."""
    y = _sane_state()
    inp = _sane_inputs(inhib=inhib_flag)
    par = _params()
    dy = rhs(t=10.0, y=y, inp1=inp, par=par)
    assert dy.shape == (33,)
    assert np.all(np.isfinite(dy)), f"non-finite dy in inhib={inhib_flag}: {dy}"


def test_rhs_does_not_mutate_input_state():
    """The y[6] mutation in the basic-pH branch must not leak back to caller."""
    y = _sane_state(pH=8.0)   # basic regime
    y_before = y.copy()
    inp = _sane_inputs(inhib=2, dist_flag=0)
    par = _params()
    rhs(t=10.0, y=y, inp1=inp, par=par)
    np.testing.assert_array_equal(y, y_before, err_msg="rhs mutated input y!")


def test_rhs_is_deterministic():
    y = _sane_state()
    inp = _sane_inputs()
    par = _params()
    a = rhs(t=10.0, y=y, inp1=inp, par=par)
    b = rhs(t=10.0, y=y, inp1=inp, par=par)
    np.testing.assert_array_equal(a, b)


def test_basic_ph_branch_runs_without_nan():
    """pH > 7 path mutates y[6] internally — exercise it explicitly."""
    y = _sane_state(pH=8.5)
    inp = _sane_inputs(inhib=2)
    par = _params()
    dy = rhs(t=10.0, y=y, inp1=inp, par=par)
    assert np.all(np.isfinite(dy))


def test_acidic_ph_branch_runs_without_nan():
    y = _sane_state(pH=6.0)
    inp = _sane_inputs(inhib=2)
    par = _params()
    dy = rhs(t=10.0, y=y, inp1=inp, par=par)
    assert np.all(np.isfinite(dy))


def test_dy_for_constant_growth_rate_param_equals_mu_p():
    """dy[31] = mu_p (par[0]). If disturbance is on, dy[31] = mu_p + distMuP."""
    y = _sane_state()
    par = _params()
    inp = _sane_inputs(dist_flag=0)
    dy = rhs(t=10.0, y=y, inp1=inp, par=par)
    assert dy[31] == pytest.approx(par[0])  # mu_p


def test_disturbance_flag_shifts_growth_rate_integrals():
    """With dist_flag=1, dy[31] should be mu_p + distMuP."""
    y = _sane_state()
    par = _params()
    inp = _sane_inputs(dist_flag=1)
    inp[17] = 0.005   # distMuP
    inp[18] = 0.05    # distMuX
    dy = rhs(t=10.0, y=y, inp1=inp, par=par)
    assert dy[31] == pytest.approx(par[0] + 0.005)


def test_invalid_inhib_flag_raises():
    y = _sane_state()
    inp = _sane_inputs(inhib=3)
    par = _params()
    with pytest.raises(ValueError, match="inhib_flag"):
        rhs(t=10.0, y=y, inp1=inp, par=par)


def test_volume_and_weight_derivatives_have_consistent_signs():
    """With positive Fs/Fb/Fa/Fw and zero discharge/evap, V and Wt must rise."""
    y = _sane_state()
    inp = _sane_inputs()
    par = _params()
    dy = rhs(t=10.0, y=y, inp1=inp, par=par)
    # No discharge, modest evaporation, positive feed → V and Wt should increase
    assert dy[4] > 0, f"expected dy[V] > 0, got {dy[4]}"
    assert dy[5] > 0, f"expected dy[Wt] > 0, got {dy[5]}"


# =============================================================================
# Order-of-magnitude vs MATLAB trajectory (loose; bit-exact comes in Phase 3)
# =============================================================================

def _y_from_csv_row(row) -> np.ndarray:
    """Reconstruct the 33-element ODE state from a dump CSV row.

    pH column in the CSV is in pH units (post-ODE conversion); ODE expects [H+].
    """
    y = np.zeros(N_STATES)
    y[0] = row["S"]; y[1] = row["DO2"]; y[2] = row["O2"]
    y[3] = row["P"]; y[4] = row["V"]; y[5] = row["Wt"]
    y[6] = 10.0 ** (-row["pH"])
    y[7] = row["T"]; y[8] = row["Q"]; y[9] = row["Viscosity"]
    y[10] = row["Culture_age"]
    y[11] = row["a0"]; y[12] = row["a1"]; y[13] = row["a3"]; y[14] = row["a4"]
    for i, name in enumerate(["n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9"]):
        y[15 + i] = row[name]
    y[25] = row["nm"]; y[26] = row["phi0"]
    y[27] = row["CO2outgas"]; y[28] = row["CO2_d"]
    y[29] = row["PAA"]; y[30] = row["NH3"]
    y[31] = row["mu_P_calc"]; y[32] = row["mu_X_calc"]
    return y


def _inputs_from_csv_row(row, *, inhib=2, dist_flag=0, vis_flag=0) -> np.ndarray:
    """Reconstruct the 26-element controller-input vector from a dump CSV row."""
    inp = _sane_inputs(inhib=inhib, dist_flag=dist_flag, vis_flag=vis_flag)
    inp[1]  = float(row["Fs"])
    inp[2]  = float(row["Fg"])
    inp[3]  = float(row["RPM"])
    inp[4]  = float(row["Fc"])
    inp[5]  = max(float(row["Fh"]), 1e-9)   # protect against exact-zero blowup
    inp[6]  = float(row["Fb"])
    inp[7]  = float(row["Fa"])
    inp[9]  = float(row["Fw"])
    inp[10] = float(row["pressure"])
    inp[12] = float(row["Fremoved"])
    inp[13] = float(row["Fpaa"])
    inp[14] = float(row["Foil"])
    return inp


@pytest.mark.skipif(not STATES_CSV.exists(), reason="MATLAB reference states CSV missing")
@pytest.mark.parametrize("k", [200, 700])     # mid-batch growth, late-batch
def test_volume_derivative_matches_matlab_finite_diff(k):
    """Volume is barely affected by the unknown random disturbances, so its
    instantaneous Python derivative should match a central FD on the MATLAB
    trajectory tightly even though we can't reconstruct the exact disturbance
    history. <5% relative error is the bar.

    Skip samples adjacent to harvest-state transitions (Fremoved jumps).
    """
    df = pd.read_csv(STATES_CSV, header=[0, 1])
    df.columns = df.columns.get_level_values(0)
    h = float(df["time_h"].iloc[1] - df["time_h"].iloc[0])

    # Skip transition samples — discharge ramping up/down between rows
    # would put the central FD estimate between two regimes.
    if df["Fremoved"].iloc[k - 1] != df["Fremoved"].iloc[k + 1]:
        pytest.skip(f"sample {k} sits across a Fremoved transition; FD invalid")

    row = df.iloc[k]
    y = _y_from_csv_row(row)
    inp = _inputs_from_csv_row(row)
    par = _params()
    dy_py = rhs(t=float(df["time_h"].iloc[k]), y=y, inp1=inp, par=par)
    assert np.all(np.isfinite(dy_py))

    fd_V = (df["V"].iloc[k + 1] - df["V"].iloc[k - 1]) / (2.0 * h)
    rel_err = abs(dy_py[4] - fd_V) / max(abs(fd_V), 1.0)
    assert rel_err < 0.05, (
        f"sample {k}: dy[V] py={dy_py[4]:.4g} vs MATLAB FD={fd_V:.4g} "
        f"(rel_err={rel_err*100:.2f}%) — exceeds 5% bar"
    )


@pytest.mark.skipif(not STATES_CSV.exists(), reason="MATLAB reference states CSV missing")
@pytest.mark.parametrize("col,state_idx,tol", [
    ("Wt",        5,  0.05),  # weight                — 0.05% in practice
    ("P",         3,  0.05),  # penicillin            — 0.27%
    ("Viscosity", 9,  0.01),  # viscosity             — 0.00% (deterministic ODE)
    ("a3",       13,  0.05),  # degenerated biomass   — 0.10%
    ("a4",       14,  0.05),  # autolysed biomass     — 0.14%
    ("PAA",      29,  0.05),  # phenylacetic acid     — 1.16%
    ("NH3",      30,  0.05),  # nitrogen              — 0.86%
])
def test_disturbance_insensitive_states_match_matlab_fd(col, state_idx, tol):
    """States whose dynamics are dominated by ODE structure (not control loops
    or stochastic disturbances) should match MATLAB's central FD tightly even
    without reconstructing the disturbance trajectory.

    Sample 200 = ~40h, mid-batch growth phase, no harvest active.
    """
    df = pd.read_csv(STATES_CSV, header=[0, 1])
    df.columns = df.columns.get_level_values(0)
    h = float(df["time_h"].iloc[1] - df["time_h"].iloc[0])

    k = 200
    row = df.iloc[k]
    y = _y_from_csv_row(row)
    inp = _inputs_from_csv_row(row)
    par = _params()
    dy_py = rhs(t=float(df["time_h"].iloc[k]), y=y, inp1=inp, par=par)

    fd = (df[col].iloc[k + 1] - df[col].iloc[k - 1]) / (2.0 * h)
    rel_err = abs(dy_py[state_idx] - fd) / max(abs(fd), 1e-9)
    assert rel_err < tol, (
        f"dy[{col}] py={dy_py[state_idx]:.5g} vs MATLAB FD={fd:.5g} "
        f"(rel_err={rel_err*100:.3f}%) — exceeds {tol*100:.1f}% bar"
    )


# =============================================================================
# Tight bit-ish-exact validation using the captured x0 + disturbances
# =============================================================================

@pytest.mark.skipif(
    not (STATES_CSV.exists() and INITCONDS_MAT.exists()),
    reason="MATLAB reference (states CSV or initconds .mat) missing",
)
@pytest.mark.parametrize("col,state_idx,tol", [
    ("P",         3,  0.005),   # penicillin           — should be ~0.2%
    ("V",         4,  0.01),    # volume               — ~0.8% (V_evp etc. are FD-window-averaged)
    ("Wt",        5,  0.01),    # weight               — ~0.5%
    ("Viscosity", 9,  0.001),   # viscosity            — ~0.001%
    ("a0",       11,  0.01),    # growing biomass      — ~0.1% with disturbances
    ("a1",       12,  0.01),    # non-growing biomass  — ~0.1% with disturbances
    ("a3",       13,  0.005),
    ("a4",       14,  0.005),
    ("PAA",      29,  0.005),
    ("NH3",      30,  0.005),
])
def test_states_match_matlab_with_actual_params_and_disturbances(col, state_idx, tol):
    """Strong validation: same params + same disturbances → match MATLAB FD
    within tight tolerances.

    Remaining error (a few tenths of a percent on most states) is from:
      - FD truncation (O(h²) ≈ 0.04 for our h=0.2)
      - Disturbance index alignment (k vs k+1 — they're nearly equal under IIR(0.995))
      - MATLAB taking sub-steps within an ode_solver call vs our single rhs evaluation

    Excluded states (T, Q, S, DO2, O2, CO2_d, CO2outgas, pH): these are
    fast-control-loop or near-equilibrium states where the central FD over
    0.4h doesn't capture the instantaneous derivative meaningfully.
    """
    from indpensim.io.initial_conditions import load_captured_batch

    cap = load_captured_batch(seed=42, batch_index=1)
    ic = cap.initial_conditions

    df = pd.read_csv(STATES_CSV, header=[0, 1])
    df.columns = df.columns.get_level_values(0)
    h = cap.h
    k = 200
    row = df.iloc[k]

    y = _y_from_csv_row(row)
    inp = _inputs_from_csv_row(row, inhib=cap.control_flags.Inhib,
                                dist_flag=cap.control_flags.Dis,
                                vis_flag=cap.control_flags.Vis)
    inp[17:25] = cap.disturbances.at(k)   # CSV row k uses Python index k

    par = Parameters.default(
        mu_p=ic.mup, mux_max=ic.mux, alpha_kla=ic.alpha_kla,
        N_conc_paa=ic.N_conc_paa, PAA_c=ic.PAA_c,
    ).to_legacy_par_vector()

    dy_py = rhs(t=float(df["time_h"].iloc[k]), y=y, inp1=inp, par=par)
    fd = (df[col].iloc[k + 1] - df[col].iloc[k - 1]) / (2.0 * h)
    rel_err = abs(dy_py[state_idx] - fd) / max(abs(fd), 1e-9)

    assert rel_err < tol, (
        f"{col}: py={dy_py[state_idx]:.6g} vs MATLAB FD={fd:.6g} "
        f"(rel_err={rel_err*100:.3f}%) — exceeds {tol*100:.2f}% bar"
    )
