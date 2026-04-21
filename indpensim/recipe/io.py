"""JSON round-trip for Recipe.

Schedules serialize as lists of ``[breakpoint, value]`` pairs.
``None`` fields on ``SetpointProfile`` (``T_sp``, ``pH_sp``) and
``TransitionTrigger`` (unused state fields) survive the round-trip.
"""
from __future__ import annotations

from typing import Any

from indpensim.recipe.types import (
    Phase,
    Recipe,
    SetpointProfile,
    SetpointSchedule,
    TransitionTrigger,
)


def _sched_to_list(s: SetpointSchedule) -> list[list[float]]:
    return [[bp, sp] for bp, sp in s]


def _sched_from_list(xs: list[list[float]]) -> SetpointSchedule:
    return tuple((float(bp), float(sp)) for bp, sp in xs)


def setpoint_profile_to_dict(sp: SetpointProfile) -> dict[str, Any]:
    return {
        "Fs":         _sched_to_list(sp.Fs),
        "Foil":       _sched_to_list(sp.Foil),
        "Fg":         _sched_to_list(sp.Fg),
        "pressure":   _sched_to_list(sp.pressure),
        "Fpaa":       _sched_to_list(sp.Fpaa),
        "Fwater":     _sched_to_list(sp.Fwater),
        "Fdischarge": _sched_to_list(sp.Fdischarge),
        "T_sp":       sp.T_sp,
        "pH_sp":      sp.pH_sp,
    }


def setpoint_profile_from_dict(d: dict[str, Any]) -> SetpointProfile:
    return SetpointProfile(
        Fs=_sched_from_list(d.get("Fs", [])),
        Foil=_sched_from_list(d.get("Foil", [])),
        Fg=_sched_from_list(d.get("Fg", [])),
        pressure=_sched_from_list(d.get("pressure", [])),
        Fpaa=_sched_from_list(d.get("Fpaa", [])),
        Fwater=_sched_from_list(d.get("Fwater", [])),
        Fdischarge=_sched_from_list(d.get("Fdischarge", [])),
        T_sp=d.get("T_sp"),
        pH_sp=d.get("pH_sp"),
    )


def transition_to_dict(t: TransitionTrigger) -> dict[str, Any]:
    return {
        "max_hours":   t.max_hours,
        "state_var":   t.state_var,
        "state_op":    t.state_op,
        "state_value": t.state_value,
    }


def transition_from_dict(d: dict[str, Any]) -> TransitionTrigger:
    return TransitionTrigger(
        max_hours=d.get("max_hours"),
        state_var=d.get("state_var"),
        state_op=d.get("state_op"),
        state_value=d.get("state_value"),
    )


def phase_to_dict(p: Phase) -> dict[str, Any]:
    return {
        "name": p.name,
        "setpoints": setpoint_profile_to_dict(p.setpoints),
        "transition": transition_to_dict(p.transition),
    }


def phase_from_dict(d: dict[str, Any]) -> Phase:
    return Phase(
        name=d["name"],
        setpoints=setpoint_profile_from_dict(d["setpoints"]),
        transition=transition_from_dict(d["transition"]),
    )


def to_dict(recipe: Recipe) -> dict[str, Any]:
    return {
        "name": recipe.name,
        "phases": [phase_to_dict(p) for p in recipe.phases],
    }


def from_dict(d: dict[str, Any]) -> Recipe:
    return Recipe(
        name=d["name"],
        phases=tuple(phase_from_dict(p) for p in d["phases"]),
    )
