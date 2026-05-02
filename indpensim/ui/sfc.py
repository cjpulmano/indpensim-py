"""Sequential Function Chart (SFC) rendering.

Produces a GraphViz DOT string from a Recipe. Consumed by
``st.graphviz_chart`` on the visualization page. Kept streamlit-free so
the DOT builder can be unit-tested in isolation.

Layout is vertical (rankdir=TB): steps are rectangles, transitions are
dark rounded pills sitting between successive steps. Initial/final
bars bracket the chart per SFC convention.

Phase state is reflected by fill color:

- active → bright green (currently running)
- complete → muted green (finished, transition fired)
- held → amber (paused)
- aborted → red
- pending → pale lavender (default, config view)
"""
from __future__ import annotations

from indpensim.recipe.types import Recipe, SetpointProfile, TransitionTrigger
from indpensim.ui.units import k_to_c


# Fill palette for phase states.
_COLOR_ACTIVE    = "#6DBE72"
_COLOR_COMPLETE  = "#BAD5BA"
_COLOR_PENDING   = "#E6DEF5"
_COLOR_HELD      = "#F5D76E"
_COLOR_ABORTED   = "#E57A7A"

_TRANSITION_FILL = "#2A2A2A"

_CHANNELS = ("Fs", "Foil", "Fg", "pressure", "Fpaa", "Fwater", "Fdischarge")


def _safe_id(name: str) -> str:
    return "s_" + "".join(c if c.isalnum() else "_" for c in name)


def _trigger_text(trig: TransitionTrigger) -> str:
    parts: list[str] = []
    if trig.max_hours is not None:
        parts.append(f"t≥{trig.max_hours:g}h")
    if trig.state_var is not None:
        parts.append(f"{trig.state_var}{trig.state_op}{trig.state_value:g}")
    return " or ".join(parts) if parts else "—"


def _action_summary(setpoints: SetpointProfile, max_items: int = 4) -> str:
    """One-line-per-channel summary for the step-box body."""
    bits: list[str] = []
    for ch in _CHANNELS:
        sched = getattr(setpoints, ch)
        if sched:
            _, sp = sched[-1]
            bits.append(f"{ch}={sp:g}")
    if setpoints.T_sp is not None:
        bits.append(f"T={k_to_c(setpoints.T_sp):.1f}°C")
    if setpoints.pH_sp is not None:
        bits.append(f"pH={setpoints.pH_sp:g}")
    if not bits:
        return ""
    shown = bits[:max_items]
    if len(bits) > max_items:
        shown.append(f"+{len(bits) - max_items} more")
    return r"\n".join(shown)


def _fill_and_border(
    phase_name: str,
    active_phase: str | None,
    completed_phases: tuple[str, ...],
    held_phase: str | None,
    aborted_phase: str | None,
) -> tuple[str, str, int]:
    if phase_name == aborted_phase:
        return _COLOR_ABORTED, "solid", 2
    if phase_name == held_phase:
        return _COLOR_HELD, "bold", 3
    if phase_name == active_phase:
        return _COLOR_ACTIVE, "bold", 3
    if phase_name in completed_phases:
        return _COLOR_COMPLETE, "solid", 1
    return _COLOR_PENDING, "solid", 1


def build_sfc_dot(
    recipe: Recipe,
    active_phase: str | None = None,
    completed_phases: tuple[str, ...] = (),
    held_phase: str | None = None,
    aborted_phase: str | None = None,
) -> str:
    """Return a GraphViz DOT string for an SFC view of the recipe.

    Defaults produce a pure config view (no highlighting). Pass state
    flags to render an execution snapshot.
    """
    lines: list[str] = [
        "digraph SFC {",
        "  rankdir=TB;",
        "  nodesep=0.25;",
        "  ranksep=0.22;",
        '  bgcolor="transparent";',
        '  node [fontname="Helvetica"];',
        '  edge [arrowhead=none, color="#666", penwidth=1.2];',
    ]

    # Initial bar (SFC convention: solid horizontal bar at the top).
    lines.append(
        '  _start [shape=box, width=1.6, height=0.06, style=filled, '
        'fillcolor="#111", label=""];'
    )

    for i, phase in enumerate(recipe.phases):
        pid = _safe_id(phase.name)
        fill, border, penw = _fill_and_border(
            phase.name, active_phase, completed_phases, held_phase, aborted_phase
        )
        actions = _action_summary(phase.setpoints)
        label = phase.name + (r"\n" + actions if actions else "")
        lines.append(
            f'  {pid} [shape=box, style="filled,{border}", fillcolor="{fill}", '
            f'penwidth={penw}, label="{label}", fontsize=11, width=2.6, '
            'margin="0.2,0.12"];'
        )
        # Transition after this phase (always — last phase's trigger is the
        # recipe-complete signal).
        t_label = _trigger_text(phase.transition)
        lines.append(
            f'  t{i} [shape=box, style="filled,rounded", fillcolor="{_TRANSITION_FILL}", '
            f'fontcolor=white, fontsize=9, height=0.22, width=1.9, '
            f'label="{t_label}"];'
        )

    prev = "_start"
    for i, phase in enumerate(recipe.phases):
        pid = _safe_id(phase.name)
        lines.append(f"  {prev} -> {pid};")
        lines.append(f"  {pid} -> t{i};")
        prev = f"t{i}"

    lines.append(
        '  _end [shape=box, width=1.6, height=0.06, style=filled, '
        'fillcolor="#111", label=""];'
    )
    lines.append(f"  {prev} -> _end;")
    lines.append("}")

    return "\n".join(lines)


def sfc_state_from_active(
    recipe: Recipe, active_phase: str | None
) -> tuple[tuple[str, ...], str | None]:
    """Derive (completed, active) from a single "currently at phase N" cursor.

    Small convenience so the UI can expose a single "which phase is
    running?" dropdown and have the prior phases automatically flip to
    completed.
    """
    if active_phase is None:
        return (), None
    names = [p.name for p in recipe.phases]
    if active_phase not in names:
        return (), None
    idx = names.index(active_phase)
    return tuple(names[:idx]), active_phase
