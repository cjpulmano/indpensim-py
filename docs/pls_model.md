# PAA_PLS_model.mat — structure & usage

Source: `PAA_PLS_model.mat` (16 KB, MATLAB v5 / v7.2-compatible).
Decoded with `scipy.io.loadmat` — see `scripts/inspect_pls_mat.py`.

## File contents

Single variable:

| Name | Type | Shape | Dtype | Notes |
|------|------|-------|-------|-------|
| `b` | ndarray | `(10, 212)` | `float64` | PLS regression coefficient matrix |

No preprocessing scalars (means/stds) — input centering/scaling lives entirely in the spectral preprocessing pipeline (`sgolayfilt` + `diff` + windowed slicing).

## Semantics of `b`

`b[i-1, :]` is the PLS regression coefficient vector you would use if you chose **i latent variables** for the PLS model. Row index → LV count. So 10 rows = 10 candidate models (1 LV through 10 LV); each row is the 212-element coefficient vector you dot with the preprocessed spectrum to get predicted PAA concentration.

Hard-coded in `Substrate_prediction.m:11`: **`No_LV = 4`** — the production model always uses 4 latent variables, i.e. row index 3 in 0-based Python (`b[3, :]`).

## Feature vector construction (212 elements)

In `Substrate_prediction.m`:

```matlab
Raman_Spec_sg   = sgolayfilt(X.Raman_Spec.Intensity(:,j)', 2, 5);   % SG smooth, order=2, window=5
Raman_Spec_sg_d = diff(Raman_Spec_sg);                                % first derivative
PAA_peaks_Spec  = Raman_Spec_sg_d([350:500 800:860], :);              % select 151+61=212 bins
```

Index intervals (1-based MATLAB → inclusive endpoints):
- bins **350..500** → 151 features (PAA-relevant Raman shift region 1)
- bins **800..860** → 61 features (PAA-relevant Raman shift region 2)
- total = **212** ← matches the second axis of `b`

In Python (0-based, `np.diff` shortens by 1, so the indices are off-by-one — careful):

```python
sg   = savgol_filter(spectrum, window_length=5, polyorder=2)
sg_d = np.diff(sg)                # length = N-1
features = np.concatenate([sg_d[349:500], sg_d[799:860]])  # exact MATLAB-equivalent slicing
assert features.shape == (212,)
```

**Off-by-one watch:** MATLAB `Raman_Spec_sg_d([350:500 800:860])` selects post-`diff` indices 350–500 and 800–860 (inclusive, 1-based). Post-`diff` arrays in Python use 0-based, so `350` → `349` and `500` exclusive becomes `500`. The Python slice `[349:500]` gives indices 349..499 inclusive = 151 elements. Same for the second window.

## Prediction step

```matlab
X.PAA_pred.y(j) = PAA_peaks_Spec' * b(No_LV,:)';   % scalar = (1x212) * (212x1)
if j > 20
    X.PAA_pred.y(j) = (X.PAA_pred.y(j-1) + X.PAA_pred.y(j-2) + X.PAA_pred.y(j))/3;
end
```

Three-point moving average kicks in after sample 20 — averages the *just-predicted* value with the previous two stored predictions (in-place; the old `y(j)` is the fresh PLS output). Port carefully — if you average over already-smoothed history you'll double-smooth.

## Python port shape

```python
class PAAPLSModel:
    """PLS regression for PAA concentration from Raman spectrum.

    Loaded once from PAA_PLS_model.mat. Use predict(spectrum) per sample.
    """
    coefficients: np.ndarray  # shape (10, 212) — row i = (i+1) LV model
    no_lv: int = 4            # production model uses 4 latent variables

    @property
    def beta(self) -> np.ndarray:
        return self.coefficients[self.no_lv - 1]  # (212,)

    def predict_raw(self, spectrum: np.ndarray) -> float:
        sg = savgol_filter(spectrum, window_length=5, polyorder=2)
        sg_d = np.diff(sg)
        features = np.concatenate([sg_d[349:500], sg_d[799:860]])
        return float(features @ self.beta)
```

The 3-point averaging is *temporal* (depends on prior predictions), so it belongs in the calling loop, not in `predict_raw`.

## Validation target

Once MATLAB reference is captured (Phase 1), `tests/test_pls_model.py` should:
1. Load one MATLAB-saved Raman spectrum.
2. Run `PAAPLSModel.predict_raw` on it.
3. Compare against the corresponding `X.PAA_pred.y(j)` scalar (pre-averaging).
4. Assert `abs(py - matlab) < 1e-9` — pure linear algebra, should match to machine precision modulo `savgol_filter` boundary handling.

**Boundary handling is the one risk.** `scipy.signal.savgol_filter` defaults to `mode='interp'` (polynomial extrapolation at edges); MATLAB `sgolayfilt` uses a different boundary scheme. With window=5 only the first 2 and last 2 samples differ; for 2200-bin spectra the impact on the 350..500 + 800..860 windows should be zero. Verify empirically once we have a reference spectrum.
