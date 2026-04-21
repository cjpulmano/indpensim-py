"""Regression gate for the Recipe executor path.

Asserts that running ``simulate()`` with ``legacy_sbc_recipe()`` attached
to the ``CapturedBatch`` produces *bit-identical* trajectories to the
legacy path (``recipe=None`` → hardcoded ``_RECIPE_*`` tables). The
executor must be a mechanical drop-in for ``_recipe_lookup``; any
numerical difference here is a port bug in the recipe layer.

Covers three representative configs to exercise each downstream branch:
  - seed 42 (vanilla, Faults=0, PRBS=0, Raman_spec=0)
  - seed 101 (vanilla_a — different initial conditions)
  - any seed whose capture has Raman_spec=2 or PRBS=1, if available.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from indpensim.io.initial_conditions import load_captured_batch
from indpensim.recipe.legacy import legacy_sbc_recipe
from indpensim.simulation import simulate

REF_DIR = Path(__file__).resolve().parents[1] / "data" / "matlab_reference"


def _available_captures() -> list[tuple[int, int]]:
    """Discover (seed, batch_index) pairs under data/matlab_reference."""
    out: list[tuple[int, int]] = []
    for f in sorted(REF_DIR.glob("batch_seed*_b*_initconds.mat")):
        # filename: batch_seed<S>_b<NN>_initconds.mat
        parts = f.stem.split("_")
        seed = int(parts[1].removeprefix("seed"))
        batch_index = int(parts[2].removeprefix("b"))
        out.append((seed, batch_index))
    return out


def _simulate_both(seed: int, batch_index: int):
    cap = load_captured_batch(seed=seed, batch_index=batch_index)
    cap_with_recipe = dataclasses.replace(
        cap, recipe=legacy_sbc_recipe(h=cap.h),
    )
    # Same tolerances for both to rule out solver-tier differences.
    r_legacy = simulate(cap, rtol=1e-6, atol=1e-9)
    r_recipe = simulate(cap_with_recipe, rtol=1e-6, atol=1e-9)
    return r_legacy, r_recipe


@pytest.fixture(scope="module")
def baseline_pair():
    if not (REF_DIR / "batch_seed42_b01_initconds.mat").exists():
        pytest.skip("seed 42 batch 1 capture missing")
    return _simulate_both(42, 1)


def test_states_bit_identical(baseline_pair):
    r_legacy, r_recipe = baseline_pair
    assert np.array_equal(r_legacy.states, r_recipe.states)


def test_ph_trajectory_bit_identical(baseline_pair):
    r_legacy, r_recipe = baseline_pair
    assert np.array_equal(r_legacy.pH_trajectory, r_recipe.pH_trajectory)


@pytest.mark.parametrize("channel", [
    "Fs", "Foil", "Fg", "pressure", "Fpaa", "Fw", "Fremoved",
    "Fa", "Fb", "Fc", "Fh", "RPM", "viscosity", "Fault_ref",
    "PRBS_noise_addition",
])
def test_control_channel_bit_identical(baseline_pair, channel):
    r_legacy, r_recipe = baseline_pair
    legacy = r_legacy.history.channels[channel]
    recipe = r_recipe.history.channels[channel]
    assert np.array_equal(legacy, recipe), (
        f"{channel} differs; max abs diff = {np.max(np.abs(legacy - recipe))}"
    )


@pytest.mark.parametrize("channel", [
    "S", "DO2", "O2", "P", "V", "Wt", "pH", "T", "Q", "Viscosity",
    "Culture_age", "a0", "a1", "a3", "a4", "X",
    "n0", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "n8", "n9", "nm", "phi0",
    "CO2outgas", "CO2_d", "PAA", "NH3",
    "mu_P_calc", "mu_X_calc", "OUR", "CER",
])
def test_state_channel_bit_identical(baseline_pair, channel):
    r_legacy, r_recipe = baseline_pair
    legacy = r_legacy.history.channels[channel]
    recipe = r_recipe.history.channels[channel]
    assert np.array_equal(legacy, recipe), (
        f"{channel} differs; max abs diff = {np.max(np.abs(legacy - recipe))}"
    )


# Multi-seed sweep — runs only on captures present locally. Skips cleanly
# otherwise. Keeps the baseline test above as a dedicated quick check.
_CAPTURES = _available_captures() or [(42, 1)]


@pytest.mark.parametrize("seed,batch_index", _CAPTURES)
def test_sweep_states_bit_identical(seed, batch_index):
    path = REF_DIR / f"batch_seed{seed}_b{batch_index:02d}_initconds.mat"
    if not path.exists():
        pytest.skip(f"capture missing: {path.name}")
    r_legacy, r_recipe = _simulate_both(seed, batch_index)
    assert np.array_equal(r_legacy.states, r_recipe.states), (
        f"state drift on seed={seed} batch={batch_index}"
    )
