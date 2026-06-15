"""Spec builders for the Run page.

Pure (streamlit-free) helpers that turn a Recipe + an IC source choice
into a ``CapturedBatch`` ready for ``simulate()``. Two paths:

  - ``spec_from_captured`` — load a bundled MATLAB reference batch and
    swap in the studio's recipe. Deterministic; uses the recipe to
    drive feeds and (optionally) override per-phase T_sp / pH_sp.
  - ``spec_from_python_rng`` — generate a fresh init from numpy with a
    user-supplied seed. Production-flavored: different starting state
    each seed, but reproducible for a given seed.

Result selection (final P, batch length, etc.) is also pure here so
the page can stay thin.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import numpy as np

from indpensim.driver import (
    BatchConfig,
    CampaignConfig,
    batch_spec_from_capture,
    batch_spec_from_python_rng,
)
from indpensim.io.initial_conditions import CapturedBatch
from indpensim.recipe.types import Recipe
from indpensim.simulation import SimulationResult


ICSource = Literal["captured", "python_rng"]


@dataclass(frozen=True)
class RunSpec:
    """User-chosen knobs for one Run-page batch."""
    recipe: Recipe
    ic_source: ICSource
    captured_seed: int = 42         # used when ic_source="captured"
    captured_batch: int = 1
    python_seed: int = 42           # used when ic_source="python_rng"


def spec_from_captured(recipe: Recipe, seed: int, batch_index: int) -> CapturedBatch:
    """Load a captured init and inject ``recipe`` into it."""
    cap = batch_spec_from_capture(seed, batch_index)
    return replace(cap, recipe=recipe)


def spec_from_python_rng(recipe: Recipe, seed: int) -> CapturedBatch:
    """Generate a fresh init from numpy and attach ``recipe``."""
    rng = np.random.default_rng(seed)
    campaign = CampaignConfig()
    batch_cfg = BatchConfig(faults=0, prbs=0, fixed_length=True,
                            raman_spec=0, recipe=recipe)
    return batch_spec_from_python_rng(rng, batch_no=1,
                                      campaign=campaign, batch=batch_cfg)


def build_spec(spec: RunSpec) -> CapturedBatch:
    """Dispatch to the right spec builder based on ``ic_source``."""
    if spec.ic_source == "captured":
        return spec_from_captured(spec.recipe, spec.captured_seed,
                                  spec.captured_batch)
    if spec.ic_source == "python_rng":
        return spec_from_python_rng(spec.recipe, spec.python_seed)
    raise ValueError(f"unknown ic_source: {spec.ic_source!r}")


# ---------------------------------------------------------------------------
# Result-selection helpers (used by the page; kept pure for testability)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunSummary:
    """Headline metrics extracted from a SimulationResult."""
    duration_h: float
    P_final_g_per_L: float
    X_final_g_per_L: float
    V_final_L: float
    n_phase_transitions: int


def summarize(result: SimulationResult,
              n_phase_transitions: int = 0) -> RunSummary:
    h = result.history
    N = h.N
    return RunSummary(
        duration_h=float(result.t[N]),
        P_final_g_per_L=float(h.channels["P"][N]),
        X_final_g_per_L=float(h.channels["X"][N]),
        V_final_L=float(h.channels["V"][N]),
        n_phase_transitions=n_phase_transitions,
    )


def trajectory_dataframe(result: SimulationResult, channels: tuple[str, ...]):
    """Pull (time_h, ch1, ch2, ...) into a small pandas DataFrame.

    Imports pandas lazily so this module stays import-cheap when only
    ``build_spec`` is needed (e.g. in unit tests).
    """
    import pandas as pd
    h = result.history
    N = h.N
    data = {"time_h": result.t[1 : N + 1]}
    for ch in channels:
        data[ch] = h.channels[ch][1 : N + 1]
    return pd.DataFrame(data)


def captured_seed_choices(ref_dir: Path | None = None) -> list[int]:
    """Discover bundled captured seeds by scanning the reference dir."""
    base = ref_dir if ref_dir is not None else (
        Path(__file__).resolve().parents[2] / "data" / "matlab_reference"
    )
    seeds: set[int] = set()
    for p in base.glob("batch_seed*_b01_initconds.mat"):
        # filename: batch_seed<NN>_b01_initconds.mat
        try:
            seeds.add(int(p.name.split("_")[1].removeprefix("seed")))
        except ValueError:
            continue
    return sorted(seeds)
