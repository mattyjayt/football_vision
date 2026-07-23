# ENGINEERING.md — footlab technical walkthrough

> A deeper tour of how footlab works: the data model, why each design decision
> exists, how to run every demo, what output to expect, and the published
> research each component is built from.
>
> Audience: a strong software engineer who is **new to sports analytics**. Read
> top to bottom once; thereafter use it as a reference.

---

## Table of contents

1. [The one-sentence mental model](#1-the-one-sentence-mental-model)
2. [The `FrozenFrame` — the heart of the design](#2-the-frozenframe--the-heart-of-the-design)
3. [Why velocities? (and why we derive them)](#3-why-velocities-and-why-we-derive-them)
4. [The pipeline, stage by stage](#4-the-pipeline-stage-by-stage)
5. [The planners compared — A* vs PSO vs potential fields](#5-the-planners-compared--a-vs-pso-vs-potential-fields)
6. [Real data: the SkillCorner bridge](#6-real-data-the-skillcorner-bridge)
7. [How to run everything](#7-how-to-run-everything)
8. [Expected results & how to read the figures](#8-expected-results--how-to-read-the-figures)
9. [Testing philosophy](#9-testing-philosophy)
10. [References & sources](#10-references--sources)

---

## 1. The one-sentence mental model

> **Freeze one instant of a match into pure numbers, then ask: what is the best
> thing the ball carrier could do from here — and does that plan survive when
> the defense reacts?**

Everything in the repo serves that question. The "pure numbers" are a single
object — the `FrozenFrame` — and the rest of the repo is a stack of
increasingly sophisticated ways to answer it.

```
FrozenFrame ─► space geometry ─► pitch control ─► value + cost map ─► PLANNER ─► rollout vs. reacting defense
 (foundations)   (Voronoi /       (Spearman        (xT value +         (A* / APF /   (score a plan; replan
                  dominant regions) PPCF surface)    defender cost)      PSO)          as defenders move)
```

---

## 2. The `FrozenFrame` — the heart of the design

Defined in `src/footlab/state.py`. This is **the only object any algorithm
reads**, and it is the single most important design decision in the repo.

```python
@dataclass
class FrozenFrame:
    positions:    np.ndarray   # (N, 2) player (x, y) in meters, origin at pitch center
    velocities:   np.ndarray   # (N, 2) player (vx, vy) in m/s
    team_ids:     np.ndarray   # (N,)   0 = TEAM_ATTACK, 1 = TEAM_DEFEND
    ball_pos:     np.ndarray   # (2,)   ball (x, y) in meters
    ball_carrier: int          # index into the arrays above, or -1 if ball is loose
    player_ids:   np.ndarray | None  # optional jersey numbers, for debugging
```

### Shape & invariants

- `N` is the number of players visible in the frame (22 for a full match).
- All four core arrays agree on `N`; `__post_init__` validates shapes and value
  ranges and **fails loudly at construction** rather than deep inside a planner.
- `team_ids` may only contain `0` (attack) or `1` (defend) — enforced.
- `ball_carrier` is an index (not an id), so `positions[ball_carrier]` is the
  carrier's position. `-1` means the ball is loose.

### Why a single rigid object?

Because it **decouples the brain from the eyes**. Every downstream algorithm —
Voronoi, pitch control, A*, PSO, the reactive rollout — consumes a
`FrozenFrame` and nothing else. It neither knows nor cares whether the numbers
came from:

- a **simulated** scenario generator (`simulate.py`), used by demos 00–06;
- **real professional tracking** (`skillcorner.py`), used by demo 07 / complete;
- or, in future, **your own drone footage** run through detection + homography.

That is the whole point. The tactics engine was finished and tested against
simulation long before real data existed; when the SkillCorner bridge was
written, **not one line of the planners changed**. Any future data source that
can emit player/ball coordinates per frame just needs its own `*_to_frozen()`
function.

### Coordinate convention (fixed in Phase 0, `pitch.py`)

| | |
|---|---|
| Pitch | 105 m × 68 m |
| Origin | the center spot |
| x | along the length, `[-52.5, +52.5]` |
| y | across the width, `[-34, +34]` |
| Units | meters, m/s |
| Attack direction | **always toward +x** (goal at `(+52.5, 0)`) |

This matches the Metrica / Friends-of-Tracking convention, which is also the
convention SkillCorner uses — so real data drops in with essentially no
rescaling.

---

## 3. Why velocities? (and why we derive them)

A static photo of a match is not enough to reason about it. A defender 5 m
away who is **standing still** and a defender 5 m away **sprinting at you** are
completely different problems, and only velocity tells them apart. Velocities
feed:

- **Dominant regions** (Phase 1) — who *arrives first* at a point depends on
  current motion, not just current position.
- **Pitch control** (Phase 2) — the time-to-intercept model needs to know how
  fast each player is already moving.
- **Defender projection** (Phases 4, 6) — defenders are projected *forward* in
  time so the planner avoids where they *will be*, not where they are.

### Where velocities come from

- **Simulated frames** — generated directly by the scenario, capped at the
  physiological sprint limit (`MAX_PLAYER_SPEED = 9 m/s`).
- **Real SkillCorner frames** — the data provides **positions only** at 10 fps.
  `skillcorner.py` *derives* velocities by a **centered finite difference**
  between neighbouring frames (`(x[t+1] − x[t−1]) / (2·Δt)`, Δt = 0.1 s), then
  clips to the sprint cap so single-frame jitter can't produce a 40 m/s ghost.
  These are **estimates**, noisier than the simulated ones — treat any single
  velocity with suspicion, trust the aggregate.

---

## 4. The pipeline, stage by stage

Each stage is its own module with its own demo, tests, and note. They compose
bottom-up.

### Phase 0 — Foundations (`pitch.py`, `state.py`, `simulate.py`, `viz.py`)
Pitch constants and renderer, the `FrozenFrame`, scenario generators
(counter-attack, low block, wing overload, random), and shared plotting.
Everything else imports from here.

### Phase 1 — Space geometry (`geometry.py`)
Who owns each patch of grass. Two answers:
- **Voronoi** — a point belongs to the *nearest* player (pure distance).
- **Dominant region** — a point belongs to whoever *arrives first*, using a
  simple motion model (max speed + reaction time). Velocity visibly shifts
  ownership toward players already moving the right way.
*(Taki & Hasegawa 2000; Fujimura & Sugihara 2005.)*

### Phase 2 — Pitch control (`pitch_control.py`)
A smooth surface giving, for every cell of the pitch, the probability the
**attacking** team would control the ball if it arrived there. Players "race"
to each target under a time-to-intercept model; control accumulates through
coupled ODEs. *(Spearman 2018, "Beyond Expected Goals.")*

### Phase 2b — Value + cost map (`value_surface.py`)
Two layers fused into one planner objective:
- a **positional value** grid (xT-inspired: value rises toward goal, boosted in
  the box), and
- a **defender-proximity penalty**.

```
cost(cell) = w_control · (1 − pitch_control)
           + w_value   · (1 − value)
           + w_defender· defender_penalty
```

The weights are the coaching dial: high `w_defender` = patient, safe buildup;
high `w_value` = direct, risky verticality. *(Singh 2019, Expected Threat, as
inspiration — we use an analytic surrogate, not learned xT.)*

### Phase 3 — Graph search (`planners/grid_search.py`)
The first real planner. Dijkstra and A* over the cost map (8-connected grid,
admissible straight-line heuristic), plus shortcut smoothing. A* returns the
same optimum as Dijkstra while expanding far fewer nodes — this is the project
MVP. *(Hart, Nilsson & Raphael 1968.)*

### Phase 4 — Artificial potential fields (`planners/potential_fields.py`)
Goal as attractor, defenders as repulsors (projected forward); extract a path
by gradient descent. Fast and smooth, but can **trap in a local minimum**
between two defenders — demonstrated on purpose, because understanding *why* it
fails is a learning goal. *(Khatib 1986.)*

### Phase 5 — PSO over Bezier control points (`planners/pso_path.py`)
Parameterise the path as a cubic Bezier (2 interior control points = a 4-D
search), and optimise with particle swarm. Fitness = path cost integral +
curvature/off-pitch penalties. *(Kennedy & Eberhart 1995; Shi & Eberhart 1998
inertia weight.)*

### Phase 6 — Reactive defenders (`planners/reactive.py`)
The frozen-frame assumption is false: defenders react. Roll time forward in
Δt = 0.2 s steps; the carrier follows a plan while each defender pursues an
intercept point under speed/acceleration caps. Two modes: score a **static**
plan (how far before interception), and **replan** (re-run A* every few steps
on the updated frame). The capstone: static plans fail, replanned ones succeed.
*(Fujimura & Sugihara 2005 motion model.)*

---

## 5. The planners compared — A* vs PSO vs potential fields

Three different planners solve the *same* problem three different ways. The
point of having all three is to understand their trade-offs — and on real data
they genuinely disagree.

| | **A\*** (grid search) | **PSO** (Bezier) | **Potential fields** |
|---|---|---|---|
| Path shape | polyline on a grid | smooth parametric curve | smooth, reactive |
| Optimality | **guaranteed optimal** on the grid | good, not guaranteed | none (local method) |
| Speed | fast | medium (many fitness evals) | **fastest** |
| Failure mode | grid discretisation artifacts | sensitive to swarm/iter settings | **local minima traps** |
| Best for | the **baseline / source of truth** | smooth, natural-looking paths | quick reactive steering |

**The honest verdict (Phase 5 note):** on a grid cost map, **A\* is usually the
stronger baseline** — guaranteed optimal and fast. PSO shines when the
parameterisation is low-dimensional and the objective is non-differentiable or
you want an inherently smooth curve. On the real SkillCorner frame the two take
**visibly different routes** (A\* hugs one side of a defensive block, PSO arcs
around the other) — a genuine, inspectable disagreement that is exactly what
this lab exists to surface. **Potential fields are included mainly to be
understood and to fail instructively**, not to be used in anger.

---

## 6. Real data: the SkillCorner bridge

`src/footlab/skillcorner.py` turns professional broadcast tracking into
`FrozenFrame`s, so the entire stack runs on a real match.

**Source:** [SkillCorner/PySport open data](https://github.com/SkillCorner/opendata)
— 10 matches of the 2024/25 Australian A-League, broadcast tracking at 10 fps,
plus dynamic events and phases of play. **Broadcast tracking** = coordinates
extracted from TV video by SkillCorner's CV/ML pipeline. Important: the repo
contains **coordinates, not video** — it is the *output* of a vision pipeline,
which makes it perfect for validating the tactics engine but useless for
training the homography/detection side.

**What the bridge does:**

1. Parses the per-frame JSONL (player `x, y`, ball, possession).
2. **Derives velocities** by finite difference (see §3).
3. **Assigns teams by possession** — the ball carrier's team is always the
   attacking team (footlab plans a path *for the carrier*). This matters: in a
   given frame the team on the ball might be home *or* away, and naively
   hard-coding "home = attack" silently plans for the wrong team.
4. **Normalises orientation** using the per-period `home_team_side` flag so the
   attack always runs toward +x (the footlab convention), mirroring x when the
   possessing team attacks right-to-left.

**Honest caveats:**

- Velocities are *estimates*, noisy by nature.
- SkillCorner **extrapolates** off-camera players (`is_detected=False`). We keep
  them (a full 22-player frame is more useful for planning) but
  `only_detected=True` drops them if you prefer purity.
- Pitches here are 104 × 68 (vs footlab's 105 × 68) — a negligible difference
  the planner tolerates.

This bridge is also the **template** for the real goal: your own footage. Any
pipeline that emits per-frame player/ball coordinates can be adapted the same
way — write a `your_source_to_frozen()` and the whole engine works unchanged.

---

## 7. How to run everything

### Setup

```bash
uv sync          # install deps into .venv
uv run pytest    # 97 tests, ~2 s
```

### Simulated-data demos (phases 0–6)

```bash
uv run python scripts/00_demo_scenarios.py
uv run python scripts/01_demo_voronoi.py
uv run python scripts/02_demo_pitch_control.py
uv run python scripts/02b_demo_cost_map.py
uv run python scripts/03_demo_grid_search.py
uv run python scripts/04_demo_potential_fields.py
uv run python scripts/05_demo_pso.py
uv run python scripts/06_demo_reactive.py   # writes figures/06_reactive.gif
```

Each is self-contained, prints a summary, and saves figure(s) to `figures/`.

### Real-data demos (need the SkillCorner clone)

```bash
# one-time (needs git-lfs):
git clone https://github.com/SkillCorner/opendata.git data/skillcorner

uv run python scripts/07_demo_skillcorner.py          # one real frame, 3 panels
uv run python scripts/07_demo_skillcorner.py 12000    # pick another frame

uv run python scripts/complete_demo.py                # ALL stages on one real frame
uv run python scripts/complete_demo.py 12000
```

`complete_demo.py` writes one figure per stage to `figures/complete/`. Toggle
stages via the `STAGES` dict at the top of the file (no code edits); set
`"reactive_gif": True` to also render the animated rollout (the slow step).

---

## 8. Expected results & how to read the figures

`complete_demo.py` (frame 5000, an away counter-attack) produces, in
`figures/complete/`:

| Figure | What you should see |
|---|---|
| `01_voronoi_dominant.png` | Two territory maps. Dominant regions **shift** vs Voronoi toward players moving toward a region (velocity effect). |
| `02_pitch_control.png` | Green where the attacking team controls, red where defenders do. Control is high *behind* the carrier, low near the defensive block and goal. |
| `03_cost_map.png` | Bright (high cost) ridges over defender clusters; dark (low cost) channels of open space. |
| `04_astar.png` | The green A* path threading low-cost space from carrier to goal. |
| `05_potential_fields.png` | A quiver field flowing around defenders toward goal; the descent path following it. |
| `06_pso.png` | **PSO (blue) and A\* (green) taking different routes** around the block — the interesting disagreement. |
| `07_reactive_summary.png` | Final trajectories: static plan vs replanning, with the interception point marked. |
| `07_reactive.gif` | The two rollouts animated — defenders converging as the carrier runs. |

**On the reactive outcome being "both intercepted":** that is *honest*, not a
bug. Frame 5000 is a congested counter-attack — the carrier is surrounded, so
under the reactive parameters he gets swarmed. On a tight frame, replanning can
even fare *worse* than committing (it keeps darting into space defenders then
close). The point of the stage is that **defenders move**, and they do. Try
other frames (`complete_demo.py 12000`, etc.) for cleaner attacking situations.

---

## 9. Testing philosophy

Tests assert **invariants**, not "it ran without error":

- pitch-control probabilities in `[0, 1]` and attack + defense ≈ 1;
- A* cost equals a Dijkstra-verified optimum on small grids;
- PSO global-best fitness is monotonically non-increasing;
- paths stay on the pitch and respect speed caps;
- the vectorized pitch-control surface matches the readable reference;
- the SkillCorner bridge yields 22 players, an 11/11 split, speeds within the
  physiological cap, on-pitch positions, and a carrier who is always an
  attacker (these **skip gracefully** when the data isn't downloaded).

Everything is **seeded and deterministic**, ~2 s total.

---

## 10. References & sources

Every algorithm module carries a full citation in its docstring (author, title,
year, equation/section, parameter values, simplifications). The canonical list
lives in [`CLAUDE.md`](CLAUDE.md). Key sources:

**Papers**
- Taki, T. & Hasegawa, J. (2000). *Visualization of dominant region in team
  games.* — dominant regions.
- Fujimura, A. & Sugihara, K. (2005). *Geometric analysis and quantitative
  evaluation of sport teamwork.* — player motion model.
- Spearman, W. (2018). *Beyond Expected Goals.* MIT SSAC. — pitch control (PPCF).
- Singh, K. (2019). *Introducing Expected Threat (xT).* — positional value
  (inspiration for the value surface).
- Hart, P., Nilsson, N. & Raphael, B. (1968). *A Formal Basis for the Heuristic
  Determination of Minimum Cost Paths.* — A*.
- Khatib, O. (1986). *Real-Time Obstacle Avoidance for Manipulators and Mobile
  Robots.* — artificial potential fields.
- Kennedy, J. & Eberhart, R. (1995). *Particle Swarm Optimization.* (+ Shi &
  Eberhart 1998 inertia weight.) — PSO.
- Wang, Z. et al. (2024). *TacticAI: an AI assistant for football tactics.*
  Nature Communications. — context: a GNN corner-kick assistant; **not** what
  footlab builds (see below).

**Data & code (referenced, not vendored)**
- [SkillCorner/opendata](https://github.com/SkillCorner/opendata) — the real
  tracking data powering `skillcorner.py` (MIT license; please credit
  SkillCorner).
- [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) —
  alternative open tracking data.
- Laurie Shaw, [LaurieOnTracking](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)
  — reference implementation of Metrica loading + Spearman pitch control.
- [mplsoccer](https://github.com/andrewRowlinson/mplsoccer) — pitch-plotting
  reference (we implement our own minimal renderer).

**How footlab differs from TacticAI.** TacticAI is a learned GNN for **corner
kicks** at an elite club, recommending player repositioning from a large
proprietary dataset. footlab is an **explainable, open-play, path-planning**
engine that runs on a single frozen frame — no training data, no black box, and
(eventually) usable on footage of any team, not just those with tracking-data
contracts. Different tool, different user.

---

*License: MIT. Every module reimplements published, openly cited algorithms.*
