# Football Tactical Path-Planning Research Lab

## Role & Mission

You are helping me build a **research-grade learning repository** for multi-agent spatial optimization on a 2D football pitch. The end objective (NOT to be built immediately) is: given a frozen frame of play (22 player positions + ball position on a 2D pitch), compute a tactically optimal path for the ball carrier under different scenarios (route to goal, defensive recovery, passing option evaluation), accounting for opponents.

**The priority is MY UNDERSTANDING, not a product.** Every component must be built as a small, isolated, independently runnable and testable script. We combine components only at the end, and only when each has been understood in isolation. Timeline is 2–4 weeks of part-time work. A *working* solution beats an *optimal* one.

## Project Status

**All phases (0–6) are complete, tested (91 passing tests), and documented.** The
static + reactive planning stack is fully built on simulated data. See
`README.md` for the overview, `applications.md` for value/use cases, and
`notes/phaseN-*.md` for per-phase write-ups.

The constraints, conventions, and citation policy below remain in force for
future work — chiefly the upstream computer-vision pipeline (drone footage →
pitch/player/ball detection → homography → `FrozenFrame`), which is out of scope
for `footlab` itself but which this repo is designed to feed into.

## Hard Constraints

1. **Environment**: `uv`-managed Python project (already `uv init`-ed), Python 3.11+, MacBook M1 (CPU only — never assume CUDA). Run scripts with `uv run`.
2. **Dependencies**: keep minimal. Core: `numpy`, `scipy`, `matplotlib`, `pandas`. Add `pytest` as dev dependency. Do NOT add heavy frameworks (no PyTorch, no gym, no networkx unless a phase explicitly justifies it — and ask me first).
3. **Code style**: simple, readable, research-grade. Prefer plain functions and small dataclasses over class hierarchies. No premature abstraction, no config frameworks, no CLI frameworks — a `if __name__ == "__main__":` demo block in every script is enough.
4. **Modularity & isolation**: every algorithm lives in its own module and has its own demo script that runs standalone on simulated data. A person should be able to run `uv run python scripts/demo_pitch_control.py` and see a plot without touching anything else.
5. **Citations are mandatory**: every function implementing a published algorithm must cite, in its docstring: author(s), paper title, year, the specific equation/section implemented, parameter meanings and the values used in the original paper, and any simplifications we made. Example format below.
6. **Full files**: when modifying code, always output/write complete updated files, never fragments.
7. **Explain as you build**: before writing each module, give me a short (5–15 line) plain-language explanation of the algorithm: the intuition, inputs/outputs, and why it matters for the football problem. Treat me as a strong software engineer who is NEW to this specific domain.
8. **One phase at a time**: complete a phase, let me run and inspect it, wait for my go-ahead before starting the next.

## Citation Docstring Format (use everywhere)

```python
def pitch_control_at_target(...):
    """Probability each team controls a target location if the ball arrived there.

    Reference:
        Spearman, W. (2018). "Beyond Expected Goals." MIT Sloan Sports
        Analytics Conference. Implements the potential pitch control field
        (PPCF) model, Eq. (3)-(5): players race to the target under a
        time-to-intercept model; control accumulates via coupled ODEs.

    Parameters follow Spearman (2018) defaults:
        reaction_time: 0.7 s
        max_speed: 5.0 m/s   (we use per-player estimates when available)
        tti_sigma: 0.45      (uncertainty in time-to-intercept, logistic)
        lambda_att/def: 4.3  (control rate 1/s)

    Simplifications vs. paper: [list them honestly].
    """
```

## Repository Structure to Create

```
football-path-lab/
├── pyproject.toml
├── README.md                  # index of phases, how to run each demo
├── src/footlab/
│   ├── pitch.py               # pitch constants (105 x 68 m), coordinate helpers, plotting (draw pitch with matplotlib, no extra deps)
│   ├── state.py               # FrozenFrame dataclass: player positions (x,y), velocities (vx,vy), team ids, ball pos, ball carrier id
│   ├── simulate.py            # scenario generators (see Data section)
│   ├── geometry.py            # Phase 1: Voronoi / dominant regions
│   ├── pitch_control.py       # Phase 2: Spearman pitch control surface
│   ├── value_surface.py       # Phase 2b: xT-style positional value grid + combined cost map
│   ├── planners/
│   │   ├── grid_search.py     # Phase 3: Dijkstra + A* over cost grid
│   │   ├── potential_fields.py# Phase 4: attractor/repulsor fields + gradient descent path
│   │   ├── pso_path.py        # Phase 5: PSO over Bezier control points
│   │   └── reactive.py        # Phase 6: defender reaction rollout
│   └── viz.py                 # shared plotting: surfaces, paths, arrows
├── scripts/                   # one runnable demo per phase, numbered
│   ├── 00_demo_scenarios.py
│   ├── 01_demo_voronoi.py
│   ├── 02_demo_pitch_control.py
│   ├── 03_demo_grid_search.py
│   ├── 04_demo_potential_fields.py
│   ├── 05_demo_pso.py
│   └── 06_demo_reactive.py
├── tests/                     # pytest, mirrors src modules; fast, deterministic (seeded)
├── data/                      # gitignored; loaders may download Metrica sample data here later
└── notes/                     # one markdown note per phase: what I learned, key equations, open questions (you draft, I edit)
```

## Data Strategy

**Phase 0 — Simulated data (primary; build first).** A frozen frame is just positions and velocities. Implement in `simulate.py`:
- `scenario_counter_attack()`: 3 attackers vs 2 defenders + GK, ball carrier on halfway line, space behind the defense.
- `scenario_low_block()`: 10 defenders compact in their own third, ball carrier at the edge of the box.
- `scenario_wing_overload()`: 3v2 on the right flank.
- `scenario_random(seed)`: plausible random frame (players clustered around the ball with formation-ish structure, capped speeds ≤ 9 m/s).
- Augmentation helpers: mirror across pitch axes, jitter positions with Gaussian noise (σ configurable — this simulates my real pipeline's homography/detection error), perturb velocities.

Conventions (document in `pitch.py` and enforce everywhere): pitch 105 × 68 m, origin at pitch center, attacking team plays left→right toward goal at (52.5, 0). All units meters and m/s. This matches the Metrica/Friends-of-Tracking convention so real data drops in later.

**Later (only after Phase 3, and only if I ask):** loader for Metrica Sports sample tracking data (github.com/metrica-sports/sample-data), and note SkillCorner open data (github.com/SkillCorner/opendata) as the broadcast-tracking analogue of my own YOLO+homography pipeline output.

## Phased Plan (one PR-sized chunk each)

**Phase 0 — Pitch, state, scenarios, plotting.** Deliver: `pitch.py`, `state.py`, `simulate.py`, `viz.py`, demo script that renders all scenarios. Tests: coordinate transforms, scenario invariants (player counts, speed caps, positions on pitch).

**Phase 1 — Space geometry.** Voronoi partition of the pitch by player positions (`scipy.spatial.Voronoi`), then the *dominant region* refinement: cells assigned by shortest *arrival time* rather than distance, using a simple motion model (constant max speed with reaction time; cite Taki & Hasegawa 1996/2000 for dominant regions and Fujimura & Sugihara 2005 for the motion model). Demo: side-by-side Voronoi vs dominant region for the same frame, showing how velocity shifts control. This is the conceptual foundation for everything after.

**Phase 2 — Pitch control surface.** Spearman (2018) potential pitch control field on a grid (~1 m cells: 105×68). Per-cell probability the attacking team controls the ball if it arrived there. Vectorize with numpy where reasonable, but clarity beats speed. Demo: heatmap overlaid on pitch with players and velocity arrows.

**Phase 2b — Value surface & cost map.** A static positional value grid in the spirit of Karun Singh's Expected Threat (xT) — for now a hand-crafted analytic surrogate (value increasing toward goal, boosted in the box) is fine; cite xT as the inspiration and note the simplification. Then define the planner cost map: `cost(cell) = w1·(1 − pitch_control) + w2·(1 − value) + w3·(proximity-to-defender penalty)`, with weights exposed as parameters. Demo: show pitch control, value, and combined cost side by side.

**Phase 3 — Graph search over the cost map (first real planner).** Dijkstra, then A* with an admissible distance heuristic, 8-connected grid, path smoothing (e.g., simple gradient/shortcut smoothing). Cite Hart, Nilsson & Raphael (1968). Demo: optimal ball-carrier path to goal on each scenario; show how changing cost weights changes the route (safe wide route vs. direct risky route). This is the MVP of the whole project.

**Phase 4 — Artificial potential fields.** Goal as attractor, defenders as repulsors (velocity-aware: project defender positions forward), gradient-descent path extraction. Cite Khatib (1986). Demonstrate the classic failure mode (local minima trap between defenders) on purpose — understanding *why it fails* is a learning goal. Demo: quiver plot of the field + extracted path, plus a local-minimum failure case.

**Phase 5 — PSO over trajectory parameters.** Parameterize the path as a cubic Bezier (2–3 free control points; start fixed at carrier, end at goal or chosen target). Fitness = path integral of the Phase 2b cost + curvature/length penalties + feasibility (max speed/turn) penalties. Implement PSO from scratch (~60 lines; cite Kennedy & Eberhart 1995; standard inertia-weight variant, cite Shi & Eberhart 1998). Compare against the Phase 3 A* path on the same frames: quality, smoothness, runtime, sensitivity to swarm size/iterations. Note in `notes/`: PSO shines when the fitness is non-differentiable or the parameterization is low-dimensional; A* on a grid is usually the stronger baseline here — verify or refute this empirically.

**Phase 6 — Reactive defenders (multi-agent step).** The frozen-frame assumption is wrong: defenders react. Implement a simple rollout: advance time in Δt = 0.2 s steps; ball carrier follows the planned path; each defender moves toward an intercept point (pure pursuit or proportional navigation toward a point ahead of the carrier) under speed/acceleration caps (cite Fujimura & Sugihara motion model). Then: (a) score any static path by rolling it out against reacting defenders (fraction of path completed before interception), and (b) implement replanning — re-run A* every k steps on the updated frame. Demo: animation (matplotlib FuncAnimation, save as GIF) showing a static plan failing vs. a replanned path succeeding.

**Optional stretch (only if time remains, ask me first):** RRT* comparison (cite Karaman & Frazzoli 2011); CMA-ES instead of PSO (via `cmaes` package); ORCA/velocity obstacles for smoother multi-agent motion (cite van den Berg et al. 2011).

**Explicitly OUT of scope for this repo:** GNNs/TacticAI-style learned models, MARL, real-time performance, integration with my Next.js app. These come later, elsewhere.

## Key References (cite these; add others as needed)

Papers:
1. Taki, T. & Hasegawa, J. (2000). "Visualization of dominant region in team games and its application to teamwork analysis." — dominant regions.
2. Fujimura, A. & Sugihara, K. (2005). "Geometric analysis and quantitative evaluation of sport teamwork." — player motion model with acceleration/drag.
3. Spearman, W. et al. (2017). "Physics-Based Modeling of Pass Probabilities in Soccer." MIT SSAC. — time-to-intercept groundwork.
4. Spearman, W. (2018). "Beyond Expected Goals." MIT SSAC. — pitch control (PPCF) + off-ball scoring opportunity (OBSO).
5. Fernández, J. & Bornn, L. (2018). "Wide Open Spaces: A statistical technique for measuring space creation in professional soccer." MIT SSAC. — alternative pitch influence model.
6. Fernández, J., Bornn, L. & Cervone, D. (2021). "A framework for the fine-grained evaluation of the instantaneous expected value of soccer possessions." Machine Learning. — EPV.
7. Singh, K. (2019). "Introducing Expected Threat (xT)." karun.in/blog/expected-threat.html — positional value.
8. Kennedy, J. & Eberhart, R. (1995). "Particle Swarm Optimization." IEEE ICNN. (+ Shi & Eberhart 1998 inertia weight.)
9. Hart, P., Nilsson, N. & Raphael, B. (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths." — A*.
10. Khatib, O. (1986). "Real-Time Obstacle Avoidance for Manipulators and Mobile Robots." — artificial potential fields.
11. Karaman, S. & Frazzoli, E. (2011). "Sampling-based algorithms for optimal motion planning." — RRT*.
12. Wang, Z. et al. (2024). "TacticAI: an AI assistant for football tactics." Nature Communications. — context for where the field is heading (out of scope to implement).

Repos / data to know about (reference in README, don't vendor code):
- Laurie Shaw, `LaurieOnTracking` (Friends of Tracking) — reference implementation of Metrica loading + Spearman pitch control.
- `metrica-sports/sample-data` — open tracking data.
- `SkillCorner/opendata` — broadcast-derived tracking (matches my pipeline's noise profile).
- `mplsoccer` — pitch plotting reference (we implement our own minimal version to avoid the dependency; borrow conventions only).

## Testing & Notes Policy

- Every module gets pytest tests: deterministic (seed everything), fast (<5 s total), asserting invariants (probabilities in [0,1] and attack+defense ≈ 1 for pitch control; A* path cost ≤ Dijkstra-verified optimum on small grids; PSO fitness monotonically non-increasing for global best; paths stay on the pitch and respect speed caps).
- After each phase, draft `notes/phaseN-<topic>.md` (e.g. `notes/phase2-pitch-control.md`): intuition, key equations (plain text/LaTeX), what the demo showed, failure modes observed, open questions, and how this phase connects to the end objective.

## First Action (historical — phases now complete)

The build started with Phase 0 and proceeded one phase at a time, each preceded
by a short plain-language explainer and followed by the user running/inspecting
the demo before moving on. All phases are now done; this section is kept as a
record of how the repo was built. For any *new* work, follow the same rhythm:
explain first, build in isolation, test invariants, write a note, then stop for
review.