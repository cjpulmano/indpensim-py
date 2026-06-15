"""Tests for indpensim.ui.run — Run-page spec builders.

End-to-end smoke tests are kept short (one captured batch, one
RNG-generated batch) because each one runs the full simulator. The
goal here is to prove that:

  * The studio's Recipe is actually injected into the spec along
    both IC paths.
  * ``simulate()`` accepts the resulting CapturedBatch and produces
    sensible non-zero biology (P > 0, V > 0).
  * ``summarize()`` and ``trajectory_dataframe()`` extract what the
    page renders.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from indpensim.recipe import legacy_sbc_recipe
from indpensim.simulation import simulate
from indpensim.ui.run import (
    RunSpec,
    build_spec,
    captured_seed_choices,
    spec_from_captured,
    spec_from_python_rng,
    summarize,
    trajectory_dataframe,
)


REF_DIR = Path(__file__).resolve().parents[1] / "data" / "matlab_reference"
HAS_CAPTURE = (REF_DIR / "batch_seed42_b01_initconds.mat").exists()


# ---------------------------------------------------------------------------
# Spec builders inject the recipe
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_CAPTURE, reason="MATLAB reference missing")
def test_spec_from_captured_attaches_recipe():
    rec = legacy_sbc_recipe()
    cap = spec_from_captured(rec, seed=42, batch_index=1)
    assert cap.recipe is rec
    # ControlFlags untouched by the swap (T_sp/pH_sp from the .mat).
    assert cap.control_flags.SBC == 0
    assert cap.T > 0


def test_spec_from_python_rng_attaches_recipe_and_is_reproducible():
    rec = legacy_sbc_recipe()
    cap_a = spec_from_python_rng(rec, seed=7)
    cap_b = spec_from_python_rng(rec, seed=7)
    assert cap_a.recipe is rec
    # Same seed → same initial state.
    assert cap_a.initial_conditions.X == cap_b.initial_conditions.X
    assert cap_a.initial_conditions.PAA == cap_b.initial_conditions.PAA
    # Different seed → different IC (sanity check the RNG actually fired).
    cap_c = spec_from_python_rng(rec, seed=8)
    assert cap_c.initial_conditions.X != cap_a.initial_conditions.X


def test_build_spec_dispatches_on_ic_source():
    rec = legacy_sbc_recipe()
    s_rng = RunSpec(recipe=rec, ic_source="python_rng", python_seed=7)
    cap = build_spec(s_rng)
    assert cap.recipe is rec


def test_build_spec_rejects_unknown_ic_source():
    rec = legacy_sbc_recipe()
    bad = RunSpec(recipe=rec, ic_source="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown ic_source"):
        build_spec(bad)


# ---------------------------------------------------------------------------
# End-to-end smoke: spec → simulate → summarize/plot
# ---------------------------------------------------------------------------

def test_python_rng_run_produces_nonzero_biology():
    """One full batch through simulate() — proves the page's hot path works."""
    rec = legacy_sbc_recipe()
    cap = spec_from_python_rng(rec, seed=7)
    result = simulate(cap)
    summary = summarize(result)
    assert summary.duration_h > 0
    assert summary.V_final_L > 1000.0           # vessel grew under feeds
    assert summary.X_final_g_per_L > 0          # biomass non-zero
    # Penicillin yield: legacy recipe at 230h reliably reaches ≥10 g/L.
    assert summary.P_final_g_per_L > 5.0


def test_trajectory_dataframe_shape_and_columns():
    rec = legacy_sbc_recipe()
    cap = spec_from_python_rng(rec, seed=7)
    result = simulate(cap)
    df = trajectory_dataframe(result, ("P", "X", "S"))
    assert list(df.columns) == ["time_h", "P", "X", "S"]
    # N samples = T/h (1150 for default 230h, 0.2h step).
    assert len(df) == int(round(cap.T / cap.h))
    assert df["time_h"].iloc[0] > 0
    assert df["time_h"].iloc[-1] == pytest.approx(cap.T, rel=1e-9)


# ---------------------------------------------------------------------------
# Captured-seed discovery
# ---------------------------------------------------------------------------

def test_captured_seed_choices_returns_sorted_unique_ints(tmp_path):
    # Stub a small ref dir with three seeds, two batches each.
    for seed in (5, 5, 11, 42):
        (tmp_path / f"batch_seed{seed}_b01_initconds.mat").touch()
    seeds = captured_seed_choices(ref_dir=tmp_path)
    assert seeds == [5, 11, 42]


def test_captured_seed_choices_handles_missing_dir(tmp_path):
    # Empty dir → empty list (page falls back to a default).
    assert captured_seed_choices(ref_dir=tmp_path) == []
