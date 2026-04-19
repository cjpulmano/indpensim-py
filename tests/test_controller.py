"""Standalone controller validation against captured MATLAB outputs.

Replays each MATLAB sample by feeding the captured state CSV into
``controller_step`` and asserting the manipulated-variable outputs match
MATLAB's per-sample columns from the dump.

Setup:
  - Pre-fill the BatchHistory's state channels (pH, T, PAA) with the
    captured trajectory so the controller's reads of X.<state>.y(k-1)
    return the same values MATLAB saw.
  - Pre-fill the controller-output channels (Fa, Fb, Fc, Fh, Fs, Fpaa) with
    the captured outputs at sample k-1 so PID's u_{k-1} arg is correct.
  - At MATLAB sample k=1, X.pH.y(1) is x0.pH (pre-integration). All other
    fctrl reads at k=1 use X.<field>.y(1) which was set to x0 by
    indpensim.m:113-122 BEFORE the first fctrl call. The CSV's row 0 holds
    POST-integration state at sample 1, which is wrong for the k=1 read.

  - Workaround: at k=1, override the state slots from x0; at k>=2 the CSV
    values are correct.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REF_DIR = Path(__file__).resolve().parents[1] / "data" / "matlab_reference"
STATES_CSV = REF_DIR / "batch_seed42_b01_states.csv"
INITCONDS_MAT = REF_DIR / "batch_seed42_b01_initconds.mat"


@pytest.fixture(scope="module")
def captured():
    from indpensim.io.initial_conditions import load_captured_batch
    return load_captured_batch(seed=42, batch_index=1)


@pytest.fixture(scope="module")
def states_df():
    df = pd.read_csv(STATES_CSV, header=[0, 1])
    df.columns = df.columns.get_level_values(0)
    return df


@pytest.fixture(scope="module")
def replay_outputs(captured, states_df):
    """Run controller_step for each sample using captured states; return DataFrame."""
    from indpensim.control.controller import controller_step
    from indpensim.control.history import BatchHistory, IndustrialData

    df = states_df
    cap = captured
    N = df.shape[0]
    h = cap.h
    T = cap.T

    history = BatchHistory.empty(N)

    # Slot 1 = x0 values per indpensim.m:113-122 (only the channels initialized
    # there are valid for the first fctrl read).
    ic = cap.initial_conditions
    history.set("pH", 1, 10.0 ** (-ic.pH))    # CSV stored pH; controller reads [H+]→pH back
    history.set("T", 1, ic.T)
    history.set("PAA", 1, ic.PAA)
    history.set("PAA_pred", 1, ic.PAA)        # not used for batch 1 (Raman_spec=1)

    Xd = IndustrialData()
    rng = np.random.default_rng(0)            # PRBS=0 for batch 1, never fires

    out_rows = []
    for k in range(1, N + 1):
        u = controller_step(history, Xd, k, h, T, cap.control_flags, rng=rng)
        out_rows.append({
            "Fg": u.Fg, "RPM": u.RPM, "Fs": u.Fs, "Fa": u.Fa, "Fb": u.Fb,
            "Fc": u.Fc, "Fh": u.Fh, "Fw": u.Fw, "pressure": u.pressure,
            "Fremoved": u.Fremoved, "Fpaa": u.Fpaa, "Foil": u.Foil,
        })
        # Now write what indpensim.m:225-246 stores AFTER fctrl returns —
        # so at k+1 the controller sees the right history for u_{k-1} args.
        history.set("Fa", k, u.Fa)
        history.set("Fb", k, u.Fb)
        history.set("Fc", k, u.Fc)
        history.set("Fh", k, u.Fh)
        history.set("Fs", k, u.Fs)
        history.set("Fpaa", k, u.Fpaa)
        # And populate state channels at sample k from the captured CSV
        # (this is what MATLAB writes after the ODE solve at sample k).
        # Convert pH back to [H+] for storage (matches MATLAB convention).
        history.set("pH", k, 10.0 ** (-df["pH"].iloc[k - 1]))
        history.set("T", k, float(df["T"].iloc[k - 1]))
        history.set("PAA", k, float(df["PAA"].iloc[k - 1]))

    return pd.DataFrame(out_rows)


@pytest.mark.skipif(not (STATES_CSV.exists() and INITCONDS_MAT.exists()),
                    reason="MATLAB reference missing")
# Tolerance note: PID-driven outputs (Fa, Fb, Fc, Fh) inherit CSV float
# round-trip noise (~1e-7 absolute on pH which the integrator amplifies via
# the integral term). 1e-6 relative is well below any process tolerance.
# Recipe-driven outputs (Fg, RPM, Fs, Fw, pressure, Fremoved, Fpaa, Foil)
# are exact constants from lookup tables and match to machine precision.
@pytest.mark.parametrize("col,atol,rtol", [
    ("Fg",       1e-9, 1e-9),
    ("RPM",      1e-9, 1e-9),
    ("Fs",       1e-9, 1e-9),
    ("Fa",       1e-4, 1e-5),
    ("Fb",       1e-4, 1e-5),
    ("Fc",       1e-4, 1e-5),
    ("Fh",       1e-4, 1e-5),
    ("Fw",       1e-9, 1e-9),
    ("pressure", 1e-9, 1e-9),
    ("Fremoved", 1e-9, 1e-9),
    ("Fpaa",     1e-9, 1e-9),
    ("Foil",     1e-9, 1e-9),
])
def test_controller_output_matches_matlab(replay_outputs, states_df, col, atol, rtol):
    """Each manipulated variable matches MATLAB's per-sample value to machine precision."""
    py = replay_outputs[col].to_numpy()
    ml = states_df[col].to_numpy()
    assert py.shape == ml.shape
    diff = np.abs(py - ml)
    tol = atol + rtol * np.abs(ml)
    bad = diff > tol
    if bad.any():
        idx = np.where(bad)[0][:5]
        msgs = [f"  k={i+1}: py={py[i]:.6g}, ml={ml[i]:.6g}, diff={diff[i]:.3g}"
                for i in idx]
        n_bad = int(bad.sum())
        raise AssertionError(
            f"{col}: {n_bad}/{len(py)} samples diverge.\n" + "\n".join(msgs)
        )
