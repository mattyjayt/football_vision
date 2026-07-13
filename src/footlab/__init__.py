"""footlab — a research lab for tactical path-planning on a 2D football pitch.

The stack, bottom to top:
    pitch, state, simulate, viz   — foundations (coords, FrozenFrame, scenarios).
    geometry                      — Voronoi / dominant regions.
    pitch_control                 — Spearman potential pitch control field.
    value_surface                 — xT-style value + the combined planner cost map.
    planners.grid_search          — Dijkstra / A* over the cost map.
    planners.potential_fields     — artificial potential fields.
    planners.pso_path             — PSO over Bezier trajectories.
    planners.reactive             — reacting defenders: rollout + replanning.

Everything consumes a :class:`FrozenFrame` and nothing else, so simulated frames
now and real (drone/tracking-derived) frames later are interchangeable.
"""

from . import (
    geometry,
    pitch,
    pitch_control,
    planners,
    simulate,
    state,
    value_surface,
    viz,
)
from .state import (
    MAX_PLAYER_SPEED,
    TEAM_ATTACK,
    TEAM_DEFEND,
    FrozenFrame,
)

__all__ = [
    # modules
    "pitch",
    "state",
    "simulate",
    "viz",
    "geometry",
    "pitch_control",
    "value_surface",
    "planners",
    # common symbols
    "FrozenFrame",
    "TEAM_ATTACK",
    "TEAM_DEFEND",
    "MAX_PLAYER_SPEED",
]
