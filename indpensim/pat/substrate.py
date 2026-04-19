"""Port of Substrate_prediction.m — PAA concentration estimate from one
Raman spectrum, with causal in-place 3-point smoothing.

Mirrors the MATLAB indexing precisely:
  - At controller sample ``k``, predict from the spectrum at sample ``j = k-1``.
  - For ``j > 20`` the stored ``PAA_pred[j]`` is overwritten with
    ``(stored[j-1] + stored[j-2] + new_raw[j]) / 3``. The ``stored[j-1]``
    and ``stored[j-2]`` values are themselves products of prior smoothing
    (the recurrence is causal and accumulating). This is *intentional* per
    the MATLAB source; do not refactor to read from a separate "raw" buffer.
"""
from __future__ import annotations

import numpy as np

from indpensim.pat.pls_model import PAAPLSModel


def predict_and_store(
    *,
    pls: PAAPLSModel,
    spectrum_at_j: np.ndarray,
    paa_pred_history: np.ndarray,
    j: int,
) -> float:
    """Compute PAA prediction at sample ``j`` and write it into
    ``paa_pred_history[j]`` in place.

    Returns the value written (after smoothing if applicable).

    Args:
        pls: loaded PLS model (10×212 coefficients).
        spectrum_at_j: 1-D Raman spectrum (length ≥ 1000) corresponding to
            MATLAB's ``X.Raman_Spec.Intensity(:, j)``.
        paa_pred_history: 1-D array indexed 1-based MATLAB-style; slot ``j``
            will be written. For ``j > 20`` slots ``j-1`` and ``j-2`` must
            already hold prior (smoothed) predictions.
        j: 1-based sample index where prediction is stored.
    """
    raw = pls.predict_raw(spectrum_at_j)
    if j > 20:
        # Mirror Substrate_prediction.m:13-15. The j-1/j-2 slots hold values
        # that were themselves smoothed in earlier calls — recurrence is intended.
        smoothed = (paa_pred_history[j - 1] + paa_pred_history[j - 2] + raw) / 3.0
        paa_pred_history[j] = smoothed
        return float(smoothed)
    paa_pred_history[j] = raw
    return float(raw)
