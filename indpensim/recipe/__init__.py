"""ISA-88 subset recipe layer — alternative to hardcoded SBC tables.

A ``Recipe`` is a tuple of ``Phase`` objects, each carrying its own
``SetpointProfile`` (piecewise-constant schedules for the 7 feed channels
plus optional T/pH scalars) and a ``TransitionTrigger`` (time-in-phase
and/or a state threshold).

``RecipeExecutor`` wraps a Recipe, consumes sample k + history per step,
advances phases when triggers fire, and resolves setpoints using the
same first-match / fall-through semantics as the legacy
``controller._recipe_lookup``.

``legacy_sbc_recipe()`` returns a 4-phase Recipe whose setpoint slices
reconstitute the original hardcoded ``_RECIPE_*`` tables bit-for-bit —
the regression anchor for any recipe-driven path.
"""
from indpensim.recipe.executor import (
    PhaseTransitionLog,
    RecipeExecutor,
    ResolvedSetpoints,
)
from indpensim.recipe.io import from_dict, to_dict
from indpensim.recipe.legacy import legacy_sbc_recipe
from indpensim.recipe.types import (
    Phase,
    PhaseState,
    Recipe,
    SetpointProfile,
    TransitionTrigger,
)

__all__ = [
    "Phase",
    "PhaseState",
    "PhaseTransitionLog",
    "Recipe",
    "RecipeExecutor",
    "ResolvedSetpoints",
    "SetpointProfile",
    "TransitionTrigger",
    "from_dict",
    "legacy_sbc_recipe",
    "to_dict",
]
