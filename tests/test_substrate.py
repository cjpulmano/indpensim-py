"""Validation: Substrate_prediction (PLS + 3-point smoothing) vs MATLAB.

Loads the captured spectra CSV (1085 columns × 2200 wavenumbers), runs
Python's ``pat.substrate.predict_and_store`` against each sample, and
compares the resulting PAA_pred trajectory to the MATLAB dump.

This test is SKIPPED if ``data/matlab_reference/batch_seed42_b01_paa_pred.csv``
is missing — the user must run ``scripts/matlab_dump_paa_pred.m`` first.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REF_DIR = Path(__file__).resolve().parents[1] / "data" / "matlab_reference"
RAMAN_CSV = REF_DIR / "batch_seed42_b01_raman.csv"
PAA_PRED_CSV = REF_DIR / "batch_seed42_b01_paa_pred.csv"


@pytest.mark.skipif(not (RAMAN_CSV.exists() and PAA_PRED_CSV.exists()),
                    reason="MATLAB Raman CSV or PAA_pred dump missing — "
                           "run scripts/matlab_dump_paa_pred.m")
def test_paa_pred_matches_matlab():
    from indpensim.pat.pls_model import PAAPLSModel
    from indpensim.pat.substrate import predict_and_store

    spectra = pd.read_csv(RAMAN_CSV).iloc[:, 1:].to_numpy()  # (2200, N)
    matlab_paa = pd.read_csv(PAA_PRED_CSV)["PAA_pred"].to_numpy()
    N = spectra.shape[1]
    assert matlab_paa.size == N, f"length mismatch: {matlab_paa.size} vs {N}"

    pls = PAAPLSModel.load()
    py_paa = np.zeros(N + 1)   # 1-based indexing
    for j in range(1, N + 1):
        predict_and_store(
            pls=pls,
            spectrum_at_j=spectra[:, j - 1],
            paa_pred_history=py_paa,
            j=j,
        )

    # PLS is deterministic; smoothing is deterministic. Tolerance is set
    # by sgolayfilt boundary differences (none in our windows) and CSV
    # roundtrip on spectra (~9 sig figs).
    py = py_paa[1:]
    diff = np.abs(py - matlab_paa)
    rel = diff / np.maximum(np.abs(matlab_paa), 1e-9)
    assert rel.max() < 1e-5, (
        f"PAA_pred max rel err {rel.max():.3g} > 1e-5; "
        f"max abs err {diff.max():.3g} (matlab range "
        f"[{matlab_paa.min():.2f}, {matlab_paa.max():.2f}])"
    )


def test_substrate_prediction_smoothing_recurrence():
    """Unit test: 3-point smoothing reads from already-smoothed slots (causal,
    accumulating recurrence — see Substrate_prediction.m:13-15).
    """
    from indpensim.pat.pls_model import PAAPLSModel
    from indpensim.pat.substrate import predict_and_store

    pls = PAAPLSModel.load()
    history = np.zeros(30)
    # Fake spectra of length 1000 with a single non-zero bin so predictions
    # are non-trivial but deterministic.
    rng = np.random.default_rng(42)
    spectra = rng.normal(size=(1000, 25)) * 1000

    # j=1..20: raw value stored (no smoothing)
    for j in range(1, 21):
        predict_and_store(pls=pls, spectrum_at_j=spectra[:, j - 1],
                          paa_pred_history=history, j=j)

    # At j=21, smoothing kicks in. Compute manually and compare.
    raw_21 = pls.predict_raw(spectra[:, 20])
    expected_21 = (history[20] + history[19] + raw_21) / 3.0
    predict_and_store(pls=pls, spectrum_at_j=spectra[:, 20],
                      paa_pred_history=history, j=21)
    assert np.isclose(history[21], expected_21)

    # At j=22, the recurrence reads history[21] (already smoothed) and history[20] (raw).
    raw_22 = pls.predict_raw(spectra[:, 21])
    expected_22 = (history[21] + history[20] + raw_22) / 3.0
    predict_and_store(pls=pls, spectrum_at_j=spectra[:, 21],
                      paa_pred_history=history, j=22)
    assert np.isclose(history[22], expected_22)
