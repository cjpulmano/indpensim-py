"""Tests for the PLS PAA prediction model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from indpensim.pat.pls_model import (
    DEFAULT_NO_LV,
    EXPECTED_FEATURE_LEN,
    PAAPLSModel,
)


REF_DIR = Path(__file__).resolve().parents[1] / "data" / "matlab_reference"
RAMAN_CSV = REF_DIR / "batch_seed42_b01_raman.csv"
STATES_CSV = REF_DIR / "batch_seed42_b01_states.csv"


def test_load_default_path():
    """The bundled PAA_PLS_model.mat loads with the documented shape."""
    model = PAAPLSModel.load()
    assert model.coefficients.shape == (10, 212)
    assert model.no_lv == DEFAULT_NO_LV
    assert model.beta.shape == (EXPECTED_FEATURE_LEN,)


def test_invalid_no_lv_rejected():
    coefs = np.zeros((10, 212))
    with pytest.raises(ValueError):
        PAAPLSModel(coefficients=coefs, no_lv=0)
    with pytest.raises(ValueError):
        PAAPLSModel(coefficients=coefs, no_lv=11)


def test_feature_vector_shape_and_slicing():
    """Feature extraction produces 212 elements from a 2200-bin spectrum."""
    model = PAAPLSModel.load()
    rng = np.random.default_rng(0)
    spectrum = rng.normal(size=2200)
    feats = model.features(spectrum)
    assert feats.shape == (EXPECTED_FEATURE_LEN,)
    # First 151 from window [349:500], next 61 from window [799:860].
    assert feats[:151].shape == (151,)
    assert feats[151:].shape == (61,)


def test_predict_returns_scalar_float():
    model = PAAPLSModel.load()
    rng = np.random.default_rng(0)
    spectrum = rng.normal(size=2200)
    y = model.predict_raw(spectrum)
    assert isinstance(y, float)
    assert np.isfinite(y)


@pytest.mark.skipif(not RAMAN_CSV.exists(), reason="MATLAB reference Raman CSV missing")
def test_predict_on_matlab_spectrum_is_in_paa_range():
    """Sanity check: feeding a real MATLAB-generated spectrum yields a PAA estimate
    in roughly the same ballpark as the dumped true PAA trajectory.

    This is a *magnitude* check, not bit-equivalence. To get bit-equivalence we'd
    need the dump script to also export Substrate_prediction.m's PAA_pred trace.
    True PAA in batch_seed42_b01 ranges ~500..3000 mg/L; a sane prediction should
    be within roughly an order of magnitude of that, not 0 or 1e9.
    """
    model = PAAPLSModel.load()
    raman = pd.read_csv(RAMAN_CSV)
    states = pd.read_csv(STATES_CSV, header=[0, 1])
    states.columns = states.columns.get_level_values(0)

    n_samples = raman.shape[1] - 1
    assert n_samples == states.shape[0], "raman/states sample count mismatch"

    # Pick a mid-batch sample (well past the first-spectrum-zeroed region).
    sample_idx = n_samples // 2
    spectrum = raman.iloc[:, sample_idx + 1].to_numpy()  # +1 to skip wavelength col

    if np.allclose(spectrum, 0):
        pytest.skip(f"sample {sample_idx} spectrum is all zeros (Raman not yet recorded)")

    pred = model.predict_raw(spectrum)
    true_paa = float(states["PAA"].iloc[sample_idx])

    # Loose magnitude check — PLS error is large but bounded.
    assert 0 < pred < 1e5, f"prediction {pred} mg/L is out of any reasonable range"
    rel_err = abs(pred - true_paa) / max(abs(true_paa), 1.0)
    # PLS prediction error vs true PAA is typically <50% in this domain.
    # Don't assert anything tight here — bit-level test belongs in a future
    # MATLAB-vs-Python PAA_pred comparison once the dump script exports it.
    assert rel_err < 5.0, (
        f"prediction {pred:.1f} mg/L vs true {true_paa:.1f} mg/L "
        f"differs by {rel_err*100:.0f}% — sanity threshold breached"
    )
