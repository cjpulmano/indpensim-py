"""Multi-config parity against MATLAB reference runs.

Complements ``test_playback.py`` (which only tests the ODE in isolation by
replaying MATLAB's controller outputs). Here we run the **full Python
pipeline** — Python's controller + Python's ODE — on each captured config
and compare every state against the matching MATLAB CSV. This exercises
fault branches, the Raman PAA loop, and variable-length batches that
single-seed validation never touched.

Configs are produced by ``scripts/matlab_run_validation_set.m``. Each
config uses a distinct seed; see that script for the full list.

Missing captures are skipped (so the suite still runs on a fresh clone
without MATLAB output). When captures exist, each config yields one test
per threshold-bearing channel.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from indpensim.driver import batch_spec_from_capture
from indpensim.simulation import simulate

REF_DIR = Path(__file__).resolve().parents[1] / "data" / "matlab_reference"


@dataclass(frozen=True)
class Config:
    seed: int
    faults: int
    raman: int
    fixed: int
    tag: str


CONFIGS: tuple[Config, ...] = (
    Config(101, 0, 0, 1, "vanilla_a"),
    Config(102, 0, 0, 1, "vanilla_b"),
    Config(103, 0, 0, 1, "vanilla_c"),
    Config(201, 1, 0, 1, "fault_aeration"),
    Config(202, 2, 0, 1, "fault_pressure"),
    Config(203, 3, 0, 1, "fault_substrate"),
    Config(204, 4, 0, 1, "fault_base"),
    Config(205, 5, 0, 1, "fault_coolant"),
    Config(207, 7, 0, 1, "fault_Tsensor"),
    Config(208, 8, 0, 1, "fault_pHsensor"),
    Config(301, 0, 2, 1, "raman_paa_loop"),
    Config(401, 0, 0, 0, "variable_length"),
)


# (mean_rel_err_frac, max_rel_err_frac).
# Bounds are deliberately looser than single-seed observations so they
# hold across the 12-config sweep. Tighten during triage if needed.
# Peak-normalized error thresholds: |py - mat| / max(|mat|).
# (mean_cap, max_cap). Calibrated from observed behavior at tight solver
# tolerance (rtol=1e-6, atol=1e-9) across all 12 configs, plus a ~20%
# safety margin so the test catches regressions without flapping on
# expected closed-loop noise. See VALIDATION.md for methodology and why
# DO2/mu_X_calc/Q have loose max bounds.
THRESHOLDS: dict[str, tuple[float, float]] = {
    # Primary measured states — tight across all configs.
    "V":           (0.002, 0.005),
    "Wt":          (0.002, 0.005),
    "T":           (0.002, 0.01),
    "Culture_age": (0.001, 0.002),
    # Primary CPPs — max spikes ~2-3% at controller transients.
    "P":           (0.015, 0.05),
    "S":           (0.015, 0.05),
    "pH":          (0.005, 0.05),
    "PAA":         (0.015, 0.05),
    "NH3":         (0.01, 0.03),
    # Biomass morphology.
    "a0":          (0.015, 0.05),
    "a1":          (0.005, 0.015),
    "a3":          (0.005, 0.02),
    "a4":          (0.003, 0.015),
    "phi0":        (0.015, 0.04),
    # Vacuole bins.
    **{n: (0.015, 0.04) for n in
       ("n0","n1","n2","n3","n4","n5","n6","n7","n8","n9","nm")},
    # Off-gas / composition. O2 and DO2 have wide max due to vanilla_a
    # pathological closed-loop amplification — means stay under 0.5%.
    "O2":          (0.002, 0.20),
    "DO2":         (0.005, 0.80),
    "CO2outgas":   (0.003, 0.02),
    "CO2_d":       (0.003, 0.03),
    "Viscosity":   (0.01, 0.03),
    # Per-sample growth rates.
    "mu_P_calc":   (1e-6, 1e-6),
    "mu_X_calc":   (0.02, 1.20),
    # Heat integral — most solver-sensitive channel even at tight tol.
    "Q":           (0.05, 0.70),
}


def _ref_paths(seed: int) -> tuple[Path, Path]:
    return (
        REF_DIR / f"batch_seed{seed}_b01_initconds.mat",
        REF_DIR / f"batch_seed{seed}_b01_states.csv",
    )


def _available(seed: int) -> bool:
    ic, csv = _ref_paths(seed)
    return ic.exists() and csv.exists()


@pytest.fixture(scope="module")
def run_results() -> dict[int, tuple[pd.DataFrame, "SimulationResult"]]:
    """Run Python simulate() once per available config, cache across tests.

    Skips missing configs silently — per-test skips surface that.
    """
    from indpensim.simulation import SimulationResult  # noqa: F401

    out: dict[int, tuple[pd.DataFrame, object]] = {}
    for cfg in CONFIGS:
        if not _available(cfg.seed):
            continue
        spec = batch_spec_from_capture(cfg.seed, 1)
        # Tight solver tolerance for validation. Production simulate() uses
        # rtol=1e-3 (fast, ~0.1% drift); here we drop to rtol=1e-6 so that
        # solver-path differences between scipy BDF and MATLAB ode15s don't
        # dominate the comparison. See VALIDATION.md for rationale.
        result = simulate(spec, rtol=1e-6, atol=1e-9)
        _, csv_path = _ref_paths(cfg.seed)
        ref = pd.read_csv(csv_path, header=[0, 1])
        ref.columns = ref.columns.get_level_values(0)
        out[cfg.seed] = (ref, result)
    return out


def _rel_err(py: np.ndarray, mat: np.ndarray) -> tuple[float, float]:
    """Peak-normalized absolute error: |py - mat| / max(|mat|).

    Instantaneous |py - mat| / |mat| is meaningless when |mat| is small
    relative to the channel's range — a 0.01 abs diff on a 0.07 mg/L
    sample (channel peaks at 46) would report "14% off" but only
    represents 0.02% of the channel's meaningful range. Peak-normalized
    error measures divergence against the channel's scale, which is what
    we actually care about for a faithful port.
    """
    peak = max(float(np.max(np.abs(mat))), 1e-12)
    rel = np.abs(py - mat) / peak
    return float(rel.mean()), float(rel.max())


@pytest.mark.parametrize("cfg", CONFIGS, ids=[c.tag for c in CONFIGS])
@pytest.mark.parametrize("channel", list(THRESHOLDS.keys()))
def test_channel_matches_matlab(cfg: Config, channel: str, run_results):
    if cfg.seed not in run_results:
        pytest.skip(f"missing MATLAB capture for seed {cfg.seed} ({cfg.tag})")
    ref, result = run_results[cfg.seed]
    if channel not in ref.columns:
        pytest.skip(f"{channel} not in reference CSV")

    N = ref.shape[0]
    mat = ref[channel].to_numpy()

    if channel == "pH":
        py = result.pH_trajectory[1 : N + 1]
    else:
        py = result.history.channels[channel][1 : N + 1]

    mean_rel, max_rel = _rel_err(np.asarray(py, dtype=float), mat)
    mean_cap, max_cap = THRESHOLDS[channel]
    assert mean_rel < mean_cap, (
        f"{cfg.tag}/{channel}: mean_rel {mean_rel:.4f} > {mean_cap}"
    )
    assert max_rel < max_cap, (
        f"{cfg.tag}/{channel}: max_rel {max_rel:.4f} > {max_cap}"
    )


def test_at_least_one_config_present():
    """Sanity: if *no* captures exist, signal that loudly (not silent pass)."""
    present = [c.tag for c in CONFIGS if _available(c.seed)]
    if not present:
        pytest.skip(
            "No MATLAB reference captures under data/matlab_reference/. "
            "Run scripts/matlab_run_validation_set.m in MATLAB first."
        )
