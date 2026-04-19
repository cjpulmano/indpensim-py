"""Port of Raman_Sim.m — synthesizes a 2200-bin Raman spectrum from the
current process state, optionally adding stochastic noise.

The deterministic part is a sum of:
  (1) ``intensity_increase * intensity_shift * scaling_factor``  — broad
      ramp scaled by an empirical combination of biomass / product /
      viscosity / time;
  (2) ``reference_spectra``                                      — fixed baseline;
  (3) Gaussian peaks for glucose, PAA, and product, scaled by the
      corresponding state concentrations.

Noise model (when enabled): per-bin uniform-{−n, 0, +n} noise, cumulative
sum across bins, then a moving-average smoothing with span 25, multiplied
by 10. MATLAB uses a *shrinking-window* moving average at the boundaries
(``smooth(x, 25)``); this port currently runs the noise path with a
constant 25-wide centered window via ``np.convolve(..., 'same')``. That
matches the interior bins exactly; only ~12 bins on each end differ. None
fall inside the PLS feature windows ([350:500] and [800:860]), so the
smoothing-mode mismatch has no effect on PAA prediction.

Noise reproduction: pass ``noise=array_of_2200_floats`` to replay an
exact captured noise vector; pass ``noise=None`` to draw fresh noise from
the supplied ``rng``. MATLAB's per-call ``randi`` cannot be reproduced
bit-for-bit without a captured trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


WAVENUMBER_MAX = 2200
NOISE_FACTOR = 50.0
NOISE_SMOOTH_SPAN = 25
NOISE_GAIN = 10.0          # final multiplier on the smoothed cumulative noise

# Empirical coefficients from Raman_Sim.m:18-21
_A = -0.00178143846614472 * 0.1
_B =  1.05644816081515
_C = -0.0681439987249108 * 0.1
_D = -0.02

# Per-channel scaling factors from Raman_Sim.m:27-31
_SCALING_FACTOR = 370_000.0
_GLUC_INCREASE = 800_000.0 * 3 / 1400        # ≈ 1714.286
_PAA_INCREASE  = 1_700_000.0 / 1000           # = 1700
_PROD_INCREASE = 100_000.0

# Glucose/PAA peaks Michaelis-Menten constants (lines 166-167)
_K_G   = 0.005
_K_PAA = 4000.0


@dataclass(frozen=True)
class RamanReference:
    """Holds the fixed reference spectrum and pre-computed peak shapes.

    Built once per process; the spectrum simulation just composes these
    constants with the current state at each sample.
    """
    wavelength: np.ndarray            # shape (2200,) — wavenumber axis
    reference: np.ndarray             # shape (2200,) — baseline spectrum
    intensity_shift: np.ndarray       # shape (2200,) — exp(2j/W) - 0.5 ramp
    glucose_peaks: np.ndarray         # shape (2200,)
    paa_peaks: np.ndarray             # shape (2200,)
    product_peaks: np.ndarray         # shape (2200,)


def _gaussian_peak(center: int, width: int, length_factor: int = 2,
                   amplitude_scale: float = 1.0) -> np.ndarray:
    """Place a Gaussian on a 2200-bin axis at 1-based center ``center``.

    Mirrors the per-peak loop pattern in Raman_Sim.m:95-159 — std_dev =
    width/2, support = ±length_factor·width bins around the center.
    """
    out = np.zeros(WAVENUMBER_MAX)
    std_dev = width / 2.0
    half_len = length_factor * width
    xs = np.arange(-half_len, half_len + 1)
    vals = (1.0 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (xs / std_dev) ** 2)
    vals = vals * amplitude_scale
    # Convert MATLAB 1-based ``x + center`` indices to 0-based.
    idxs = xs + center - 1
    keep = (idxs >= 0) & (idxs < WAVENUMBER_MAX)
    out[idxs[keep]] = vals[keep]
    return out


def build_reference(reference_spectra_path: Path | str) -> RamanReference:
    """Load the reference spectrum (``reference_Specra.txt``) and pre-compute
    the Gaussian peak overlays. Call once per simulation.
    """
    raw = np.loadtxt(str(reference_spectra_path))
    wavelength = raw[:WAVENUMBER_MAX, 0].astype(float)
    reference = raw[:WAVENUMBER_MAX, 1].astype(float)

    # Intensity_shift1 (Raman_Sim.m:9-13): exp(2j/W) - 0.5 over j in 1..W.
    j = np.arange(1, WAVENUMBER_MAX + 1)
    intensity_shift = np.exp(j / (WAVENUMBER_MAX * 0.5)) - 0.5

    # Glucose: 3 peaks (lines 88-117). The peak-b is divided by 4.3.
    glucose = (
        _gaussian_peak(219, 70)
        + _gaussian_peak(639, 20, amplitude_scale=1.0 / 4.3)
        + _gaussian_peak(1053, 100)
    )
    # PAA: 2 peaks (lines 119-138).
    paa = (
        _gaussian_peak(419, 60)
        + _gaussian_peak(839, 15, amplitude_scale=1.0 / 4.3)
    )
    # Product (Pen G): 2 peaks (lines 140-159).
    product = (
        _gaussian_peak(800, 30, length_factor=4)
        + _gaussian_peak(1200, 30, length_factor=30)
    )
    return RamanReference(
        wavelength=wavelength, reference=reference,
        intensity_shift=intensity_shift,
        glucose_peaks=glucose, paa_peaks=paa, product_peaks=product,
    )


def _build_noise(rng: np.random.Generator | None,
                 noise: np.ndarray | None) -> np.ndarray:
    """Return the smoothed cumulative noise vector to add to the spectrum.

    If ``noise`` is supplied, use it directly (replay). If ``rng`` is given,
    sample from {-NOISE_FACTOR, 0, +NOISE_FACTOR} per bin. If neither is
    given, return a zero vector (fully deterministic).
    """
    if noise is not None:
        if noise.shape != (WAVENUMBER_MAX,):
            raise ValueError(
                f"noise must have shape ({WAVENUMBER_MAX},), got {noise.shape}"
            )
        random_noise = noise
    elif rng is not None:
        choices = rng.integers(1, 4, size=WAVENUMBER_MAX)
        random_noise = np.zeros(WAVENUMBER_MAX)
        random_noise[choices == 2] = NOISE_FACTOR
        random_noise[choices == 3] = -NOISE_FACTOR
    else:
        return np.zeros(WAVENUMBER_MAX)

    cumulative = np.cumsum(random_noise)
    # Centered moving average, span 25. Differs from MATLAB only at the
    # ~12-bin boundaries; not in the PLS feature windows.
    kernel = np.ones(NOISE_SMOOTH_SPAN) / NOISE_SMOOTH_SPAN
    smoothed = np.convolve(cumulative, kernel, mode="same")
    return NOISE_GAIN * smoothed


def simulate_spectrum(
    *,
    reference: RamanReference,
    P: float,
    X: float,
    viscosity: float,
    S: float,
    PAA: float,
    k: int,
    N_samples: int,
    rng: np.random.Generator | None = None,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    """Build one 2200-bin simulated Raman spectrum.

    Args:
        reference: pre-computed reference + peak shapes.
        P, X, viscosity, S, PAA: current state values used by the formula.
        k: 1-based sample index (used for the ``Time_S = k/N`` term).
        N_samples: total samples in the batch (= ``T/h``).
        rng: source for random noise (used only if ``noise`` is None).
        noise: optional pre-computed per-bin noise vector for replay.
    """
    Product_S   = P / 40.0
    Biomass_S   = X / 40.0
    Viscosity_S = viscosity / 100.0
    Time_S      = k / N_samples

    intensity_increase = (
        _A * Biomass_S + _B * Product_S + _C * Viscosity_S + _D * Time_S
    )

    deterministic = (
        intensity_increase * reference.intensity_shift * _SCALING_FACTOR
        + reference.reference
    )

    # Glucose / PAA / product overlays (Raman_Sim.m:166-171), Michaelis-Menten on glucose.
    overlays = (
        reference.glucose_peaks * _GLUC_INCREASE * S / (_K_G + S)
        + reference.paa_peaks * _PAA_INCREASE * PAA
        + reference.product_peaks * _PROD_INCREASE * P
    )

    return deterministic + _build_noise(rng, noise) + overlays
