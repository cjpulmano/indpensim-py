"""Recipe dataclasses — frozen, JSON-serializable.

Each setpoint channel on ``SetpointProfile`` is a tuple of
``(breakpoint_k, value)`` pairs interpreted with the same first-match,
fall-through-to-last semantics as ``controller._recipe_lookup``.

``TransitionTrigger`` is a hybrid: either field subset is optional, and
the executor checks ``max_hours`` first, then the state threshold —
whichever fires first advances the phase.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass


class PhaseState(enum.Enum):
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    HELD = "HELD"
    COMPLETE = "COMPLETE"
    ABORTED = "ABORTED"


SetpointSchedule = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class SetpointProfile:
    """Piecewise-constant setpoint schedules for one phase.

    Each schedule is interpreted against absolute sample index k
    (1-based, same as ``_recipe_lookup``). An empty tuple means the
    phase does not drive that channel — the executor falls through to
    the recipe's default behavior (for now: zero).

    ``T_sp`` (Kelvin) and ``pH_sp`` are per-phase overrides of the
    temperature / pH setpoints. ``None`` (the default) means the
    controller falls back to ``ControlFlags.T_sp`` /
    ``ControlFlags.pH_sp``. When set, the value applies for every
    sample while the phase is active; mid-batch phase advances cause
    a step change in setpoint that the PID integral state carries
    across. ``legacy_sbc_recipe()`` keeps both fields ``None`` to
    preserve bit-for-bit MATLAB parity in the validation tier.
    """
    Fs:         SetpointSchedule = ()
    Foil:       SetpointSchedule = ()
    Fg:         SetpointSchedule = ()
    pressure:   SetpointSchedule = ()
    Fpaa:       SetpointSchedule = ()
    Fwater:     SetpointSchedule = ()
    Fdischarge: SetpointSchedule = ()   # positive; controller applies sign
    T_sp:       float | None = None
    pH_sp:      float | None = None


@dataclass(frozen=True)
class TransitionTrigger:
    """Hybrid time / state trigger. At least one field must be set.

    - ``max_hours`` is time-in-phase (measured from when the phase
      became RUNNING), not elapsed batch time.
    - State triggers read ``history.y(state_var, k-1)`` and compare to
      ``state_value`` with ``state_op`` ∈ {``">="``, ``"<="``}.
    """
    max_hours: float | None = None
    state_var: str | None = None
    state_op: str | None = None
    state_value: float | None = None

    def __post_init__(self) -> None:
        has_time = self.max_hours is not None
        has_state = (
            self.state_var is not None
            and self.state_op is not None
            and self.state_value is not None
        )
        if not (has_time or has_state):
            raise ValueError("TransitionTrigger needs max_hours or a full state triplet")
        if self.state_op is not None and self.state_op not in (">=", "<="):
            raise ValueError(f"unsupported state_op: {self.state_op!r}")


@dataclass(frozen=True)
class Phase:
    name: str
    setpoints: SetpointProfile
    transition: TransitionTrigger


@dataclass(frozen=True)
class Recipe:
    name: str
    phases: tuple[Phase, ...]

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("Recipe must have at least one phase")
        names = [p.name for p in self.phases]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate phase names: {names}")
