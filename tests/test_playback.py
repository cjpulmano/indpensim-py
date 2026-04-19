"""End-to-end ODE playback validation.

Replays a captured MATLAB batch using MATLAB's actual controller outputs
as inputs, and asserts the full Python trajectory matches the dumped MATLAB
trajectory within tight per-state tolerances.

This is the strongest single test in the suite — it validates the ODE
right-hand side AND the integration loop AND the per-step floors AND the
post-loop unit conversions. If this passes, the only remaining piece is
the controller (fctrl_indpensim.m → control/controller.py). Until that's
ported, end-to-end runs need MATLAB-captured inputs to play back against.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REF_DIR = Path(__file__).resolve().parents[1] / "data" / "matlab_reference"
STATES_CSV = REF_DIR / "batch_seed42_b01_states.csv"
INITCONDS_MAT = REF_DIR / "batch_seed42_b01_initconds.mat"


@pytest.fixture(scope="module")
def playback_result():
    """Run the full 217h batch playback once; share across tests."""
    from indpensim.io.initial_conditions import load_captured_batch
    from indpensim.validation.playback import playback, compare_to_matlab

    cap = load_captured_batch(seed=42, batch_index=1)
    result = playback(cap, STATES_CSV)
    summary = compare_to_matlab(result, STATES_CSV)
    return result, summary


@pytest.mark.skipif(
    not (STATES_CSV.exists() and INITCONDS_MAT.exists()),
    reason="MATLAB reference (states CSV or initconds .mat) missing",
)
@pytest.mark.parametrize("col,max_mean_rel_err_pct", [
    # Every state, with the empirical mean-rel-err observed during initial
    # validation, padded by ~1.5x for safety. Ordered by MATLAB y-index.
    ("S",            0.20),
    ("DO2",          0.02),
    ("O2",           0.01),
    ("P",            0.05),
    ("V",            0.001),
    ("Wt",           0.001),
    ("pH",           0.01),
    ("T",            0.005),
    ("Q",            0.50),
    ("Viscosity",    0.05),
    ("Culture_age",  0.10),
    ("a0",           0.10),
    ("a1",           0.10),
    ("a3",           1.00),
    ("a4",           1.00),
    ("n0",           0.20),
    ("n1",           0.20),
    ("n2",           0.20),
    ("n3",           0.30),
    ("n4",           0.30),
    ("n5",           0.40),
    ("n6",           0.50),
    ("n7",           0.60),
    ("n8",           0.60),
    ("n9",           0.80),
    ("nm",           0.80),
    ("phi0",         0.20),
    ("CO2outgas",    0.05),
    ("CO2_d",        0.05),
    ("PAA",          0.05),
    ("NH3",          0.05),
    ("mu_P_calc",    1e-6),    # machine-precision; mu_p is constant
    ("mu_X_calc",    0.05),
])
def test_state_trajectory_matches_matlab(playback_result, col, max_mean_rel_err_pct):
    """Each state's mean relative error vs MATLAB stays under its tolerance."""
    _, summary = playback_result
    if col not in summary.index:
        pytest.skip(f"state {col} not in summary")
    actual = float(summary.loc[col, "mean_rel_err_pct"])
    assert actual < max_mean_rel_err_pct, (
        f"{col}: mean_rel_err {actual:.4g}% exceeds bar {max_mean_rel_err_pct}%"
    )
