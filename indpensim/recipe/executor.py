"""Stateful Recipe executor — consumed by ``controller_step``.

One executor per batch. ``.step(k, history)`` must be called with
strictly increasing k starting at 1, the same cadence as the outer
simulation loop. It:

  1. If the active phase is RUNNING, checks its transition trigger
     against time-in-phase and (optionally) a state threshold. If the
     trigger fires, advances to the next phase (or marks COMPLETE if
     this was the last one). Multiple transitions can fire in a single
     call — e.g. a phase with ``max_hours=0``.
  2. Resolves the 7 feed setpoints from the active phase's
     ``SetpointProfile`` using first-match / fall-through-to-last
     semantics identical to the legacy ``controller._recipe_lookup``.

Mutator hooks (``pause``/``resume``/``advance_phase``/``abort``) are
deferred to Phase D of the plan — the MVP is step-only. The method
stubs are present so call sites can reference them, but they raise.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

from indpensim.control.history import BatchHistory
from indpensim.recipe.types import (
    Phase,
    PhaseState,
    Recipe,
    SetpointSchedule,
    TransitionTrigger,
)


class ResolvedSetpoints(NamedTuple):
    """The 7 feed setpoints resolved at one sample k.

    ``Fdischarge`` is positive; the controller applies the sign.
    ``Fwater`` mirrors the legacy ``Fw``.
    """
    Fs: float
    Foil: float
    Fg: float
    pressure: float
    Fdischarge: float
    Fwater: float
    Fpaa: float


@dataclass(frozen=True)
class PhaseTransitionLog:
    from_phase: str
    to_phase: str | None          # None when the last phase completes
    at_k: int
    at_time_h: float
    reason: str                    # "trigger" | "advance" | "abort" | "complete"


def _lookup(k: int, schedule: SetpointSchedule, default: float = 0.0) -> float:
    """First-match / fall-through-to-last, matching ``_recipe_lookup``."""
    if not schedule:
        return default
    for bp, sp in schedule:
        if k <= bp:
            return float(sp)
    return float(schedule[-1][1])


def _trigger_fires(
    trig: TransitionTrigger,
    time_in_phase_h: float,
    history: BatchHistory,
    k: int,
) -> bool:
    if trig.max_hours is not None and time_in_phase_h >= trig.max_hours:
        return True
    if trig.state_var is not None and trig.state_value is not None:
        # Read the most recent populated history slot.
        idx = max(k - 1, 1)
        val = history.y(trig.state_var, idx)
        if trig.state_op == ">=" and val >= trig.state_value:
            return True
        if trig.state_op == "<=" and val <= trig.state_value:
            return True
    return False


@dataclass
class RecipeExecutor:
    recipe: Recipe
    h: float                                # sample period, hours
    _phase_idx: int = 0
    _phase_start_k: int = 1
    _phase_state: PhaseState = PhaseState.RUNNING
    _transitions: list[PhaseTransitionLog] = field(default_factory=list)
    _drained: int = 0                        # cursor for streaming drain

    @property
    def current_phase(self) -> Phase:
        return self.recipe.phases[self._phase_idx]

    @property
    def phase_state(self) -> PhaseState:
        return self._phase_state

    @property
    def transitions(self) -> tuple[PhaseTransitionLog, ...]:
        return tuple(self._transitions)

    def drain_new_transitions(self) -> list[PhaseTransitionLog]:
        """Return transitions logged since the last drain (streaming hook)."""
        new = self._transitions[self._drained :]
        self._drained = len(self._transitions)
        return list(new)

    # ------------------------------------------------------------------
    def step(self, k: int, history: BatchHistory) -> ResolvedSetpoints:
        # Advance phases while triggers fire (normally at most once per step,
        # but a chain of zero-duration phases could cascade).
        while self._phase_state == PhaseState.RUNNING:
            time_in_phase = (k - self._phase_start_k) * self.h
            phase = self.current_phase
            if not _trigger_fires(phase.transition, time_in_phase, history, k):
                break
            self._advance(k, reason="trigger")

        sp = self.current_phase.setpoints
        return ResolvedSetpoints(
            Fs=_lookup(k, sp.Fs),
            Foil=_lookup(k, sp.Foil),
            Fg=_lookup(k, sp.Fg),
            pressure=_lookup(k, sp.pressure),
            Fdischarge=_lookup(k, sp.Fdischarge),
            Fwater=_lookup(k, sp.Fwater),
            Fpaa=_lookup(k, sp.Fpaa),
        )

    # ------------------------------------------------------------------
    def _advance(self, k: int, *, reason: str) -> None:
        from_name = self.current_phase.name
        if self._phase_idx + 1 < len(self.recipe.phases):
            to_name = self.recipe.phases[self._phase_idx + 1].name
            self._phase_idx += 1
            self._phase_start_k = k
            self._phase_state = PhaseState.RUNNING
            self._transitions.append(PhaseTransitionLog(
                from_phase=from_name, to_phase=to_name,
                at_k=k, at_time_h=k * self.h, reason=reason,
            ))
        else:
            self._phase_state = PhaseState.COMPLETE
            self._transitions.append(PhaseTransitionLog(
                from_phase=from_name, to_phase=None,
                at_k=k, at_time_h=k * self.h, reason="complete",
            ))

    # ---- Mutator hooks — Phase D of the plan. MVP raises. --------------
    def pause(self) -> None:                                  # pragma: no cover
        raise NotImplementedError("pause() is Phase D")

    def resume(self) -> None:                                 # pragma: no cover
        raise NotImplementedError("resume() is Phase D")

    def advance_phase(self, reason: str | None = None) -> None:  # pragma: no cover
        raise NotImplementedError("advance_phase() is Phase D")

    def abort(self, reason: str | None = None) -> None:       # pragma: no cover
        raise NotImplementedError("abort() is Phase D")
