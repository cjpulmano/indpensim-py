"""Legacy SBC recipe — reconstitutes the hardcoded ``_RECIPE_*`` tables.

When fed through ``RecipeExecutor``, this Recipe produces bit-identical
setpoints to the hardcoded lookup path in ``controller_step``. That
property is the regression gate that proves the new recipe code path
is a drop-in replacement.

### Why the phase-slicing rule looks the way it does

``_recipe_lookup`` is first-match with fall-through to the last SP. So
for a phase covering ``k ∈ [k_lo, k_hi]``, each channel's per-phase
slice must include **the first global bp ≥ k_hi** — otherwise the
executor inside ``k_hi`` falls through to the wrong last-SP. Likewise
the slice must start at the first bp ≥ k_lo so that bps before the
phase don't change the first-match outcome (they can never trigger, so
dropping them is safe and gives a tighter slice).

``_slice_for_range`` below is the helper; write phase tables through it
rather than hand-picking pairs.
"""
from __future__ import annotations

from indpensim.recipe.types import (
    Phase,
    Recipe,
    SetpointProfile,
    SetpointSchedule,
    TransitionTrigger,
)


# Verbatim from controller.py:44-67. Kept here so the legacy recipe is
# self-contained and the controller's tables remain the single source of
# truth for the non-recipe path.
_FS_K = (15, 60, 80, 100, 120, 140, 160, 180, 200, 220,
         240, 260, 280, 300, 320, 340, 360, 380, 400, 800, 1750)
_FS_SP = (8, 15, 30, 75, 150, 30, 37, 43, 47, 51,
          57, 61, 65, 72, 76, 80, 84, 90, 116, 90, 80)

_FOIL_K = (20, 80, 280, 300, 320, 340, 360, 380, 400, 1750)
_FOIL_SP = (22, 30, 35, 34, 33, 32, 31, 30, 29, 23)

_FG_K = (40, 100, 200, 450, 1000, 1250, 1750)
_FG_SP = (30, 42, 55, 60, 75, 65, 60)

_PRES_K = (62.5, 125, 150, 200, 500, 750, 1000, 1750)
_PRES_SP = (0.6, 0.7, 0.8, 0.9, 1.1, 1.0, 0.9, 0.9)

_DISCH_K = (500, 510, 650, 660, 750, 760, 850, 860, 950, 960,
            1050, 1060, 1150, 1160, 1250, 1260, 1350, 1360, 1750)
_DISCH_SP = (0, 4000, 0, 4000, 0, 4000, 0, 4000, 0, 4000,
             0, 4000, 0, 4000, 0, 4000, 0, 4000, 0)

_WATER_K = (250, 375, 750, 800, 850, 1000, 1250, 1350, 1750)
_WATER_SP = (0, 500, 100, 0, 400, 150, 250, 0, 100)

_PAA_K = (25, 200, 1000, 1500, 1750)
_PAA_SP = (5, 0, 10, 4, 0)


def _slice_for_range(
    bps: tuple[float, ...],
    sps: tuple[float, ...],
    k_lo: int,
    k_hi: int,
) -> SetpointSchedule:
    """Slice (bps, sps) so a per-phase first-match lookup on k ∈ [k_lo, k_hi]
    returns the same SP as a first-match lookup on the full global table.

    Rule: take every (bp, sp) pair from the first bp ≥ k_lo up through
    and including the first bp ≥ k_hi. If no bp reaches k_hi, extend
    to the end of the table (the last SP becomes the fall-through).
    """
    assert len(bps) == len(sps)
    start = next((i for i, b in enumerate(bps) if b >= k_lo), None)
    if start is None:
        # Phase runs past every breakpoint — executor just needs the last SP.
        return ((float(bps[-1]), float(sps[-1])),)
    end = next((i for i, b in enumerate(bps) if b >= k_hi), len(bps) - 1)
    return tuple((float(bps[i]), float(sps[i])) for i in range(start, end + 1))


# Phase boundaries (absolute k, inclusive upper). With h=0.2h these are
# 4h / 40h / 280h / 350h — matching the original batch recipe structure.
_K_INOCULATE_END = 20
_K_GROWTH_END = 200
_K_PRODUCTION_END = 1400
_K_HARVEST_END = 1750


def _profile_for(k_lo: int, k_hi: int) -> SetpointProfile:
    return SetpointProfile(
        Fs         = _slice_for_range(_FS_K,    _FS_SP,    k_lo, k_hi),
        Foil       = _slice_for_range(_FOIL_K,  _FOIL_SP,  k_lo, k_hi),
        Fg         = _slice_for_range(_FG_K,    _FG_SP,    k_lo, k_hi),
        pressure   = _slice_for_range(_PRES_K,  _PRES_SP,  k_lo, k_hi),
        Fdischarge = _slice_for_range(_DISCH_K, _DISCH_SP, k_lo, k_hi),
        Fwater     = _slice_for_range(_WATER_K, _WATER_SP, k_lo, k_hi),
        Fpaa       = _slice_for_range(_PAA_K,   _PAA_SP,   k_lo, k_hi),
        # T_sp / pH_sp left as None: the controller continues reading
        # ControlFlags, preserving legacy behavior exactly.
    )


def legacy_sbc_recipe(h: float = 0.2) -> Recipe:
    """Four-phase Recipe equivalent to the hardcoded SBC tables.

    Transitions are time-in-phase only, with ``max_hours`` set so that
    each phase boundary aligns with the legacy table k-values at the
    default sample period ``h=0.2`` (12 min). With a different h, the
    phase boundaries would shift; the caller must supply the same h the
    executor is initialized with.
    """
    phases = (
        Phase(
            name="INOCULATE",
            setpoints=_profile_for(1, _K_INOCULATE_END),
            transition=TransitionTrigger(max_hours=_K_INOCULATE_END * h),
        ),
        Phase(
            name="GROWTH",
            setpoints=_profile_for(_K_INOCULATE_END + 1, _K_GROWTH_END),
            transition=TransitionTrigger(
                max_hours=(_K_GROWTH_END - _K_INOCULATE_END) * h,
            ),
        ),
        Phase(
            name="PRODUCTION",
            setpoints=_profile_for(_K_GROWTH_END + 1, _K_PRODUCTION_END),
            transition=TransitionTrigger(
                max_hours=(_K_PRODUCTION_END - _K_GROWTH_END) * h,
            ),
        ),
        Phase(
            name="HARVEST",
            setpoints=_profile_for(_K_PRODUCTION_END + 1, _K_HARVEST_END),
            transition=TransitionTrigger(
                max_hours=(_K_HARVEST_END - _K_PRODUCTION_END) * h,
            ),
        ),
    )
    return Recipe(name="legacy_sbc", phases=phases)
