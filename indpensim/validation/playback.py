"""End-to-end ODE playback for validation.

Replays a captured MATLAB batch using *MATLAB's actual controller outputs*
(Fc, Fh, Fs, etc., taken from the dumped CSV at each sample) instead of
running our own controller. This isolates ODE + integration-loop correctness
from controller correctness — when this matches MATLAB tightly, we know
the foundation is solid before we port `fctrl_indpensim.m`.

Approach (mirrors `indpensim.m:107-220`):
  1. Pack `x0` into the 33-element initial state vector (with the
     pH→[H+] conversion that `indpensim.m:105` does).
  2. For each sample interval k = 0 .. N-1:
       - Build the 26-element controller-input vector from the CSV row k
         (these are the inputs MATLAB actually used) plus disturbance[k]
         (so the disturbance regime matches).
       - Integrate `rhs` from t[k] to t[k+1] with `scipy.solve_ivp(method='BDF')`
         (≈ MATLAB ode15s).
       - Apply the same per-step numerical floors MATLAB applies
         (lines 216-220 and 252-257 of indpensim.m).
       - Store the endpoint as `states_py[k]`.
  3. Convert the y[6] trajectory from [H+] back to pH (mirror of
     `indpensim.m:382`) so columns are directly comparable to the CSV.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

from indpensim.io.initial_conditions import CapturedBatch, load_captured_batch
from indpensim.io.parameters import Parameters
from indpensim.ode.rhs import N_STATES, rhs


# Maps Python state index → CSV column name (post-pH-conversion convention).
# Matches indpensim.m:248-319 X.<field>.y assignments.
STATE_TO_COL: dict[int, str] = {
    0: "S", 1: "DO2", 2: "O2", 3: "P", 4: "V", 5: "Wt",
    6: "pH",   # CSV is in pH units; we convert during storage
    7: "T", 8: "Q", 9: "Viscosity", 10: "Culture_age",
    11: "a0", 12: "a1", 13: "a3", 14: "a4",
    15: "n0", 16: "n1", 17: "n2", 18: "n3", 19: "n4",
    20: "n5", 21: "n6", 22: "n7", 23: "n8", 24: "n9",
    25: "nm", 26: "phi0",
    27: "CO2outgas", 28: "CO2_d",
    29: "PAA", 30: "NH3",
    31: "mu_P_calc", 32: "mu_X_calc",
}


def initial_state_from_capture(cap: CapturedBatch) -> np.ndarray:
    """Pack the 33-element initial state, mirroring indpensim.m:131.

    MATLAB:
        x0.pH = 10^(-x0.pH);                                     % line 105
        x00 = [x0.S x0.DO2 x0.O2 x0.P x0.V x0.Wt x0.pH x0.T 0 4 ...
               x0.Culture_age x0.a0 x0.a1 x0.a3 x0.a4 zeros(1,12) ...
               x0.CO2outgas 0 x0.PAA x0.NH3 0 0];
    """
    ic = cap.initial_conditions
    y = np.zeros(N_STATES)
    y[0] = ic.S
    y[1] = ic.DO2
    y[2] = ic.O2
    y[3] = ic.P
    y[4] = ic.V
    y[5] = ic.Wt
    y[6] = 10.0 ** (-ic.pH)             # convert pH → [H+]
    y[7] = ic.T
    y[8] = 0.0                           # Q starts at zero
    y[9] = 4.0                           # Viscosity init (literal in MATLAB)
    y[10] = ic.Culture_age
    y[11] = ic.a0
    y[12] = ic.a1
    y[13] = ic.a3
    y[14] = ic.a4
    # y[15:27] vacuole states all zero (zeros(1,12) in MATLAB)
    y[27] = ic.CO2outgas
    y[28] = 0.0                          # CO2_d starts at zero
    y[29] = ic.PAA
    y[30] = ic.NH3
    y[31] = 0.0
    y[32] = 0.0
    return y


def _apply_matlab_floors(y: np.ndarray) -> None:
    """Mirror of indpensim.m:216-220 and :252-257.

    MATLAB applies a 0.001 floor to states 1..31 (1-based) ONLY when the
    value is <= 0 — it does NOT raise tiny positives (a3 ~ 1e-29, vacuole
    bins ~ 1e-13). Earlier we used np.clip(..., 0.001, None) which silently
    raised those to 0.001 and blew up the trajectory by 10^26.
    Modifies `y` in place.
    """
    # 0.001 floor only for non-positive values, on states 0..30
    mask = y[:31] <= 0.0
    y[:31][mask] = 0.001
    # DO2 special floor (MATLAB:252-257) — applied to the stored value, then
    # propagated forward via x00 in the next iteration.
    if y[1] < 2.0:
        y[1] = 1.0


def _build_inputs_from_csv_row(
    row: pd.Series,
    dist8: np.ndarray,
    cap: CapturedBatch,
    h_ode: float = 0.01,
) -> np.ndarray:
    """Build the 26-element controller-input vector from one dump CSV row
    plus the 8 disturbances at the matching sample.

    `row` is one row of the *_states.csv dump (already de-tupled headers).
    """
    inp = np.zeros(26)
    inp[0]  = cap.control_flags.Inhib
    inp[1]  = float(row["Fs"])
    inp[2]  = float(row["Fg"])
    inp[3]  = float(row["RPM"])
    inp[4]  = float(row["Fc"])
    inp[5]  = max(float(row["Fh"]), 1e-9)    # avoid 0/0 in jacket-heat term
    inp[6]  = float(row["Fb"])
    inp[7]  = float(row["Fa"])
    inp[8]  = h_ode
    inp[9]  = float(row["Fw"])
    inp[10] = float(row["pressure"])
    inp[11] = float(row["Viscosity"])         # only used when vis_flag=1
    inp[12] = float(row["Fremoved"])
    inp[13] = float(row["Fpaa"])
    inp[14] = float(row["Foil"])
    inp[15] = 0.0                             # NH3_shots — not in dump (rare event)
    inp[16] = cap.control_flags.Dis
    inp[17:25] = dist8
    inp[25] = cap.control_flags.Vis
    return inp


@dataclass
class PlaybackResult:
    t: np.ndarray            # shape (N+1,) sample times including t=0
    states: np.ndarray       # shape (N+1, 33) — y[k, :] = state at t[k]; y[0]=initial
    pH_trajectory: np.ndarray  # shape (N+1,) — y[6] converted to pH units


def playback(cap: CapturedBatch, inputs_csv: str | Path,
             rtol: float = 1e-3, atol: float = 1e-6) -> PlaybackResult:
    """Replay one batch using MATLAB's controller outputs as inputs.

    Args:
        cap: CapturedBatch from `load_captured_batch(...)`.
        inputs_csv: path to the matching *_states.csv dump (we use it only
            for the per-sample controller-input columns).
        rtol/atol: solver tolerances. Defaults match MATLAB ode15s.

    Returns:
        PlaybackResult with full trajectory.
    """
    df = pd.read_csv(inputs_csv, header=[0, 1])
    df.columns = df.columns.get_level_values(0)
    h = cap.h
    N = df.shape[0]            # number of sample intervals dumped (=1085)
    t_grid = np.concatenate([[0.0], df["time_h"].to_numpy()])  # length N+1

    par = Parameters.default(
        mu_p=cap.initial_conditions.mup,
        mux_max=cap.initial_conditions.mux,
        alpha_kla=cap.initial_conditions.alpha_kla,
        N_conc_paa=cap.initial_conditions.N_conc_paa,
        PAA_c=cap.initial_conditions.PAA_c,
    ).to_legacy_par_vector()

    states = np.zeros((N + 1, N_STATES))
    states[0] = initial_state_from_capture(cap)
    y_curr = states[0].copy()

    for k in range(N):
        # MATLAB indpensim.m:131-165 resets the last two state slots
        # (X.mu_P_calc, X.mu_X_calc) to 0 in x00 at each iteration.
        # These aren't cumulative integrals — they're per-sample increments.
        y_curr[31] = 0.0
        y_curr[32] = 0.0

        # CSV row k holds the inputs MATLAB used during this interval AND
        # the disturbance trajectory entry for this same sample.
        inp = _build_inputs_from_csv_row(df.iloc[k], cap.disturbances.at(k), cap)

        sol = solve_ivp(
            rhs, [t_grid[k], t_grid[k + 1]], y_curr,
            args=(inp, par), method="BDF",
            rtol=rtol, atol=atol,
            t_eval=[t_grid[k + 1]],
        )
        if not sol.success:
            raise RuntimeError(f"solve_ivp failed at sample {k}: {sol.message}")
        y_curr = sol.y[:, -1]
        _apply_matlab_floors(y_curr)
        states[k + 1] = y_curr

    # Post-loop conversions, mirroring indpensim.m:382-383:
    #   X.pH.y = -log(X.pH.y)/log(10);   % [H+] → pH
    #   X.Q.y  = X.Q.y / 1000;            % Q in kJ → "kcal"-ish
    pH_traj = -np.log10(states[:, 6])
    states[:, 8] = states[:, 8] / 1000.0
    return PlaybackResult(t=t_grid, states=states, pH_trajectory=pH_traj)


def compare_to_matlab(result: PlaybackResult, csv_path: str | Path) -> pd.DataFrame:
    """Build a per-state comparison summary against the dumped MATLAB CSV.

    Returns a DataFrame with one row per state column, showing max/mean
    relative error of the Python trajectory vs MATLAB.
    """
    df = pd.read_csv(csv_path, header=[0, 1])
    df.columns = df.columns.get_level_values(0)
    N = df.shape[0]

    rows = []
    for state_idx, col in STATE_TO_COL.items():
        if col not in df.columns:
            continue
        matlab = df[col].to_numpy()                              # shape (N,)
        if col == "pH":
            python = result.pH_trajectory[1 : N + 1]
        else:
            python = result.states[1 : N + 1, state_idx]
        denom = np.maximum(np.abs(matlab), 1e-9)
        rel_err = np.abs(python - matlab) / denom
        rows.append({
            "state": col,
            "matlab_min": float(matlab.min()),
            "matlab_max": float(matlab.max()),
            "max_rel_err_pct": float(rel_err.max()) * 100,
            "mean_rel_err_pct": float(rel_err.mean()) * 100,
            "max_abs_err": float(np.abs(python - matlab).max()),
        })
    return pd.DataFrame(rows).set_index("state")


def main() -> None:
    """CLI: python -m indpensim.validation.playback"""
    cap = load_captured_batch(seed=42, batch_index=1)
    csv_path = Path("data/matlab_reference/batch_seed42_b01_states.csv")
    print(f"Replaying batch: T={cap.T}h, h={cap.h}h ({int(cap.T/cap.h)} samples)...")
    result = playback(cap, csv_path)
    print(f"Done. Final t={result.t[-1]:.1f}h\n")

    summary = compare_to_matlab(result, csv_path)
    print("Per-state comparison (Python vs MATLAB):")
    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", "{:.3g}".format)
    print(summary)


if __name__ == "__main__":
    main()
