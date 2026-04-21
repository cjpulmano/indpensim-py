"""Pure plotly figure builders for the recipe visualization page.

Kept free of streamlit so the figure construction can be unit-tested
in isolation. Streamlit pages just call ``build_recipe_timeline_figure``
and hand the result to ``st.plotly_chart``.

Layout: one row per setpoint channel that has data; each phase
boundary drawn as a colored vertical band spanning all rows. The
x-axis is sample index k; a hover annotation surfaces hours when
appropriate.
"""
from __future__ import annotations

import itertools

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from indpensim.recipe.types import Recipe, SetpointSchedule


# Color palette for phase bands — Plotly's qualitative Set2.
_PHASE_COLORS = (
    "rgba(102, 194, 165, 0.18)",
    "rgba(252, 141, 98,  0.18)",
    "rgba(141, 160, 203, 0.18)",
    "rgba(231, 138, 195, 0.18)",
    "rgba(166, 216, 84,  0.18)",
    "rgba(255, 217, 47,  0.18)",
)

_CHANNEL_DISPLAY_ORDER = (
    "Fs", "Foil", "Fg", "pressure", "Fpaa", "Fwater", "Fdischarge",
)

_CHANNEL_UNITS = {
    "Fs": "L/h", "Foil": "L/h", "Fg": "L/h", "pressure": "bar",
    "Fpaa": "L/h", "Fwater": "L/h", "Fdischarge": "L/h",
}


def _phase_boundaries_k(recipe: Recipe, h: float) -> list[tuple[int, int, str]]:
    """Walk the recipe and infer (start_k, end_k, name) per phase using
    the time-in-phase trigger if present. State-only triggers fall back
    to a placeholder span = batch length so the band still appears."""
    phases = []
    cursor_k = 1
    for p in recipe.phases:
        max_h = p.transition.max_hours
        if max_h is None:
            # State-only trigger: no a-priori end k. Mark with -1.
            phases.append((cursor_k, -1, p.name))
            return phases
        # +1 because trigger fires at the next sample; the phase covers
        # k=cursor_k..cursor_k+samples_in_phase, and the next phase starts at
        # cursor_k + samples + 1 (matching executor semantics).
        samples_in_phase = int(round(max_h / h))
        end_k = cursor_k + samples_in_phase - 1
        phases.append((cursor_k, end_k, p.name))
        cursor_k = end_k + 1
    return phases


def _max_k_in_schedule(sched: SetpointSchedule) -> int:
    if not sched:
        return 0
    return int(max(bp for bp, _ in sched))


def _step_xy(schedule: SetpointSchedule, x_max: int) -> tuple[list[float], list[float]]:
    """Convert a piecewise-constant schedule into step-plot xs/ys."""
    if not schedule:
        return [], []
    xs: list[float] = []
    ys: list[float] = []
    last_x = 1.0
    for bp, sp in schedule:
        xs.extend([last_x, bp])
        ys.extend([sp, sp])
        last_x = bp
    if x_max > last_x:
        xs.extend([last_x, x_max])
        ys.extend([schedule[-1][1], schedule[-1][1]])
    return xs, ys


def build_recipe_timeline_figure(recipe: Recipe, h: float = 0.2) -> go.Figure:
    """Build a stacked-subplot timeline. One row per non-empty channel."""
    # Find which channels actually have data anywhere in the recipe.
    active_channels = []
    for ch in _CHANNEL_DISPLAY_ORDER:
        if any(getattr(p.setpoints, ch) for p in recipe.phases):
            active_channels.append(ch)
    if not active_channels:
        # Degenerate: empty recipe. Return a placeholder figure.
        fig = go.Figure()
        fig.add_annotation(text="(recipe has no setpoint data)",
                            xref="paper", yref="paper",
                            x=0.5, y=0.5, showarrow=False)
        return fig

    bounds = _phase_boundaries_k(recipe, h)
    # Compute x extent from the largest schedule breakpoint or the last
    # phase end — whichever is greater.
    schedule_max = max(
        (_max_k_in_schedule(getattr(p.setpoints, ch))
         for p in recipe.phases for ch in active_channels),
        default=100,
    )
    bound_max = max((b[1] for b in bounds if b[1] > 0), default=schedule_max)
    x_max = max(schedule_max, bound_max)

    fig = make_subplots(
        rows=len(active_channels), cols=1, shared_xaxes=True,
        subplot_titles=[f"{ch} ({_CHANNEL_UNITS[ch]})" for ch in active_channels],
        vertical_spacing=0.04,
    )

    color_iter = itertools.cycle(_PHASE_COLORS)
    for start_k, end_k, name in bounds:
        color = next(color_iter)
        # If end_k == -1 (state-only trigger), span to x_max.
        actual_end = end_k if end_k > 0 else x_max
        fig.add_vrect(
            x0=start_k, x1=actual_end,
            fillcolor=color, line_width=0,
            annotation_text=name, annotation_position="top left",
            annotation_font_size=10,
            row="all", col=1,
        )

    for row_idx, ch in enumerate(active_channels, start=1):
        for p_idx, p in enumerate(recipe.phases):
            sched = getattr(p.setpoints, ch)
            if not sched:
                continue
            # Limit each phase's trace to its k-range so phase boundaries
            # show real piecewise behavior.
            phase_start, phase_end, _ = bounds[p_idx]
            phase_end_resolved = phase_end if phase_end > 0 else x_max
            xs, ys = _step_xy(sched, phase_end_resolved)
            # Trim xs/ys to the phase's k-range.
            trimmed_x, trimmed_y = [], []
            for x, y in zip(xs, ys):
                if phase_start <= x <= phase_end_resolved:
                    trimmed_x.append(x)
                    trimmed_y.append(y)
            if not trimmed_x:
                continue
            fig.add_trace(
                go.Scatter(
                    x=trimmed_x, y=trimmed_y,
                    mode="lines", name=f"{p.name}::{ch}",
                    showlegend=(row_idx == 1),
                    legendgroup=p.name,
                    line=dict(width=2),
                ),
                row=row_idx, col=1,
            )

    fig.update_layout(
        height=180 * len(active_channels) + 80,
        title=dict(text=f"Recipe: {recipe.name}", x=0.5),
        margin=dict(l=40, r=20, t=80, b=40),
        hovermode="x unified",
    )
    fig.update_xaxes(title_text="Sample k", row=len(active_channels), col=1)
    return fig


def phase_summary_rows(recipe: Recipe, h: float = 0.2) -> list[dict[str, str]]:
    """Tabular summary for the visualize page's info table."""
    rows = []
    for start_k, end_k, name in _phase_boundaries_k(recipe, h):
        if end_k > 0:
            duration_h = (end_k - start_k + 1) * h
            duration_str = f"{duration_h:.1f}h ({end_k - start_k + 1} samples)"
        else:
            duration_str = "state-trigger (variable)"
        # Find the trigger description on the matching phase
        phase = next((p for p in recipe.phases if p.name == name), None)
        trig = phase.transition if phase else None
        if trig is None:
            trigger_str = "—"
        else:
            parts = []
            if trig.max_hours is not None:
                parts.append(f"t≥{trig.max_hours}h")
            if trig.state_var is not None:
                parts.append(f"{trig.state_var}{trig.state_op}{trig.state_value}")
            trigger_str = " or ".join(parts) if parts else "—"
        rows.append({
            "Phase": name,
            "k range": f"{start_k}..{end_k if end_k > 0 else '?'}",
            "Duration": duration_str,
            "Trigger": trigger_str,
        })
    return rows
