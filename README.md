# football-path-lab (`footlab`)

A research-grade repository for **multi-agent spatial optimization on a 2D
football pitch**. It builds, phase by phase, the tactical "brain" that turns a
frozen frame of play into an optimal path for the ball carrier — and then stress-
tests that path against a defense that reacts.

> **Core objective.** Given a frozen frame of play — 22 player positions and
> velocities plus the ball on a 2D pitch — compute a **tactically optimal path
> for the ball carrier** under different scenarios (route to goal, breaking a
> block, evaluating options), **accounting for opponents** and, crucially, for
> the fact that those opponents *move*.

The priority is **understanding, not a product**. Every algorithm lives in its
own module, cites the paper it implements, has a standalone demo, and has its own
fast, deterministic tests. Components are combined only at the top of the stack.

**Status: complete.** All phases (0–6) are implemented, tested (91 passing
tests), and documented. See [`applications.md`](applications.md) for where this
delivers value, and [`notes/`](notes/) for a plain-language write-up of each
phase.

---

## The pipeline

A frozen frame flows up a stack of independently-built layers into a planner, and
finally into a reacting-defender rollout:

```
FrozenFrame ─► space geometry ─► pitch control ─► value + cost map ─► PLANNER ─► rollout vs. reacting defense
 (foundations)   (Voronoi /       (Spearman        (xT value +         (A* / APF /   (score a plan; replan
                  dominant regions) PPCF surface)    defender cost)      PSO)          as defenders move)
```

Everything consumes a `FrozenFrame` and nothing else, so **simulated frames now
and real drone/tracking-derived frames later are interchangeable**.

## Quick start

`uv`-managed, Python 3.11+, CPU-only (developed on an Apple M1).

```bash
uv sync                                    # install deps into .venv
uv run pytest                              # 91 tests, ~2 s

uv run python scripts/00_demo_scenarios.py # every demo writes to figures/ (gitignored)
uv run python scripts/03_demo_grid_search.py
uv run python scripts/06_demo_reactive.py  # the capstone: static plan fails vs. replanning (GIF)
```

Each `scripts/NN_demo_*.py` is self-contained: it builds simulated data, runs one
phase's algorithm, prints a summary, and saves a figure to `figures/`.

## Phases

| Phase | Topic | Demo | Notes |
|-------|-------|------|-------|
| 0 | Pitch, state, scenarios, plotting | `scripts/00_demo_scenarios.py` | [foundations](notes/phase0-foundations.md) |
| 1 | Space geometry (Voronoi / dominant regions) | `scripts/01_demo_voronoi.py` | [space-geometry](notes/phase1-space-geometry.md) |
| 2 | Pitch control surface (Spearman 2018) | `scripts/02_demo_pitch_control.py` | [pitch-control](notes/phase2-pitch-control.md) |
| 2b | Value surface & combined cost map (xT-style) | `scripts/02b_demo_cost_map.py` | [value-and-cost-map](notes/phase2b-value-and-cost-map.md) |
| 3 | Graph search — Dijkstra / A* (the MVP) | `scripts/03_demo_grid_search.py` | [graph-search](notes/phase3-graph-search.md) |
| 4 | Artificial potential fields (Khatib 1986) | `scripts/04_demo_potential_fields.py` | [potential-fields](notes/phase4-potential-fields.md) |
| 5 | PSO over Bezier trajectories | `scripts/05_demo_pso.py` | [pso-trajectories](notes/phase5-pso-trajectories.md) |
| 6 | Reactive defenders + replanning (capstone) | `scripts/06_demo_reactive.py` | [reactive-defenders](notes/phase6-reactive-defenders.md) |

## Repository layout

```
src/footlab/
  pitch.py             # pitch constants, coordinate helpers, make_grid, renderer
  state.py             # FrozenFrame dataclass + velocity helpers (clip_speed)
  simulate.py          # scenario generators + augmentation (jitter/mirror/perturb)
  viz.py               # shared plotting: frames, surfaces, regions, paths
  geometry.py          # Phase 1: Voronoi + dominant regions, time_to_arrive
  pitch_control.py     # Phase 2: Spearman potential pitch control field (PPCF)
  value_surface.py     # Phase 2b: xT-style value + defender penalty + cost_map
  planners/
    grid_search.py     # Phase 3: Dijkstra, A*, shortcut smoothing
    potential_fields.py# Phase 4: attractor/repulsor field + gradient descent
    pso_path.py        # Phase 5: PSO over cubic-Bezier control points
    reactive.py        # Phase 6: reacting-defender rollout + replanning
scripts/               # one runnable demo per phase (numbered)
tests/                 # pytest, mirrors src modules; fast, seeded, deterministic
notes/                 # plain-language write-up per phase (intuition + equations)
figures/               # generated demo images (gitignored)
applications.md        # potential applications and value added
CLAUDE.md              # project brief, conventions, and citation policy
```

## Coordinate conventions (fixed in Phase 0)

- Pitch **105 m × 68 m**, **origin at the center spot**.
- `x ∈ [−52.5, +52.5]`, `y ∈ [−34, +34]`; **meters and m/s** throughout.
- The **attacking team always plays toward +x** (goal at `(+52.5, 0)`).
- Matches the Metrica / Friends-of-Tracking convention, so real tracking data
  drops in later without rescaling.

## Documentation

- [`applications.md`](applications.md) — who this is for and the value it adds.
- [`notes/`](notes/) — one write-up per phase: intuition, key equations, what the
  demo showed, failure modes, and how each phase connects to the objective.
- [`CLAUDE.md`](CLAUDE.md) — the project brief, hard constraints, and the citation
  policy every algorithm docstring follows.

## Testing

```bash
uv run pytest           # all tests
uv run pytest -q tests/test_pitch_control.py   # one module
```

Tests assert **invariants**, not just "it runs": probabilities in `[0,1]` and
attack+defense ≈ 1 for pitch control; A* cost equals a Dijkstra-verified optimum;
PSO global-best is monotonically non-increasing; paths stay on the pitch and
respect speed caps; the vectorized pitch-control surface matches the readable
reference. Everything is seeded and fast (~2 s total).

## Data

The repo uses **simulated** frames (`simulate.py`) throughout. The conventions
above mean real data drops in later without changes:

- [`metrica-sports/sample-data`](https://github.com/metrica-sports/sample-data) — open tracking data.
- [`SkillCorner/opendata`](https://github.com/SkillCorner/opendata) — broadcast-derived tracking (matches a detection+homography pipeline's noise profile).

Reference implementations worth knowing (not vendored): Laurie Shaw's
[`LaurieOnTracking`](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)
and [`mplsoccer`](https://github.com/andrewRowlinson/mplsoccer) (we implement our
own minimal pitch renderer to avoid the dependency).

## Scope

**In scope (built):** the full static + reactive planning stack on simulated data.

**Out of scope for this repo:** the upstream computer-vision pipeline (drone
footage → pitch/player/ball detection → homography → `FrozenFrame`), learned
models (GNNs / TacticAI-style, MARL), and real-time performance. These live
elsewhere; `footlab` is the tactics/planning brain they feed into.

## References

Every algorithm module cites its source in its docstring (author, title, year,
equation/section, parameter values, and any simplifications). The full citation
list is in [`CLAUDE.md`](CLAUDE.md). Key papers: Taki & Hasegawa (2000) and
Fujimura & Sugihara (2005) for dominant regions and player motion; Spearman
(2018) for pitch control; Singh (2019) for Expected Threat; Hart, Nilsson &
Raphael (1968) for A*; Khatib (1986) for potential fields; Kennedy & Eberhart
(1995) and Shi & Eberhart (1998) for PSO.
