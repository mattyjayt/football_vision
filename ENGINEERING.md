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
   - [6.1 SkillCorner data dictionary](#61-skillcorner-data-dictionary-the-four-files-per-match)
   - [6.2 How SkillCorner produces this data](#62-how-skillcorner-produces-this-whats-public-what-isnt)
   - [6b The vision module — design & internals](#6b-the-vision-module-detection_pipeline--design--internals)
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

### 6.1 SkillCorner data dictionary (the four files per match)

Each match lives in `data/skillcorner/data/matches/{id}/` as four files. Two
are **raw perception output** (the tracking), two are **derived analytics**
(computed *from* the tracking). Understanding the split matters: the JSONL is
what a vision pipeline produces; the CSVs are what an analytics layer adds.

| File | Tier | Contents |
|---|---|---|
| `{id}_match.json` | metadata | lineups, teams, referees, pitch size, `home_team_side` |
| `{id}_tracking_extrapolated.jsonl` | **raw perception** | per-frame player + ball coordinates at 10 fps |
| `{id}_dynamic_events.csv` | **derived** | on-ball events (possessions, passes, carries) + context |
| `{id}_phases_of_play.csv` | **derived** | attacking/defending team phases with start/end frames |

**`{id}_match.json`** — the registry. Key fields:
- `home_team` / `away_team` → `{id, short_name}` (used to map team ids).
- `home_team_side` → per-period attack direction, e.g. `["right_to_left",
  "left_to_right"]` (this is what `skillcorner.py` uses to normalise to +x).
- `players` → list mapping each player's `id` → `team_id`, jersey `number`,
  `player_role` (position group), `trackable_object`. **This is how you attach
  identity/team to a raw `player_id` in the tracking.**

**`{id}_tracking_extrapolated.jsonl`** — one JSON object per line, one line per
frame (10 fps). Each frame object:
- `frame` (int), `timestamp`, `period` (1 or 2)
- `ball_data` → `{x, y, z, is_detected}` (meters, center-origin)
- `possession` → `{player_id, group}` — **inferred** possession; `null` on
  loose balls / passes in flight (this is why `ball_carrier` can be `-1`)
- `image_corners_projection` → polygon of the detected on-screen area
- `player_data` → list of `{x, y, player_id, is_detected}`; `is_detected=False`
  marks **extrapolated** (off-camera) players

Coordinates are **meters, origin at pitch center, x along the long side** — the
same convention as `footlab.pitch` (and Metrica/FoT), so no rescaling needed.

**`{id}_dynamic_events.csv`** — one row per on-ball event. ~250 columns;
highlights:
- identity: `event_id`, `frame_start/end`, `time_start/end`, `period`
- actor: `player_id/name/position`, `team_id/shortname`
- type: `event_type` (e.g. `player_possession`) + `event_subtype`
- space: `x_start/y_start → x_end/y_end`, `channel_*`, `third_*`,
  `penalty_area_*` (note: **x/y here are normalised, not meters** — they need
  scaling to the pitch before use)
- outcome/context: `pass_outcome`, `lead_to_shot`, `lead_to_goal`, `xthreat`,
  line-break and defensive-shape metrics, `speed_avg`, and many more

**`{id}_phases_of_play.csv`** — one row per phase (only while the ball is in
play). Columns include `frame_start/end`, `team_in_possession_id`,
`team_in_possession_phase_type` (e.g. `build_up`, `create`, `direct`) and the
simultaneous `team_out_of_possession_phase_type` (e.g. `high_block`,
`medium_block`), plus team width/length at phase start/end. Each in-possession
phase maps to an out-of-possession phase.

> Full official specs: SkillCorner's data glossary and the Dynamic Events /
> Phases of Play CSV specification PDFs linked from the repo README. Known
> limitations (their words): ~97% of player identities are accurate; some
> smoothing/control should be applied to raw speed/acceleration.

### 6.2 How SkillCorner produces this (what's public, what isn't)

Their **code is closed-source** — the tracking pipeline is the commercial
product. What is open is the *output* (this data) plus tutorial notebooks and a
small `src/` of loader helpers. But the *architecture* is standard and maps
directly onto the pipeline football_vision is building:

```
Broadcast video (single panning/zooming camera)
  → 1. camera calibration / pitch registration (lines/keypoints → homography)
  → 2. player & ball detection per frame (object detection)
  → 3. multi-object tracking (persistent IDs) + re-identification / team & jersey
  → 4. image coords → pitch coords via homography (meters, center-origin)
  → 5. extrapolation for off-camera players (is_detected=False)
  → 6. smoothing + possession inference
  → RAW tier: tracking.jsonl
  → DERIVED tier: dynamic_events.csv, phases_of_play.csv (computed from the raw)
```

**Path B status (as of 2026-07):** steps 1, 2 (players), 3 (teams), and 4 are
working via `detection_pipeline/` — local 32-keypoint pitch model
(`models/football-pitch-detection.pt`), fine-tuned player detector
(`models/player_detector.pt`), TeamClassifier (SigLIP → UMAP → KMeans) fitted
across the video, homography RMS ~0.8 m, and `scripts/11_demo_e2e.py` running
the full chain video → FrozenFrame → pitch control on a real frame. Remaining:
step 2-ball (dedicated ball model + slicer), step 3-tracking (ByteTrack,
persistent IDs + velocities), step 6 (temporal smoothing of H). See §6b.

---

## 6b. The vision module (`detection_pipeline/`) — design & internals

The repo has two halves, deliberately decoupled:

```
┌──────────────────────────────────────────────────────────────────────┐
│  VISION MODULE ("the eyes")        BRAIN MODULE ("the brain")        │
│  detection_pipeline/               src/footlab/                      │
│                                                                      │
│  video ─► detection ─► keypoints ─► homography ─► teams ─► JSONL ─► FrozenFrame ─► analytics  │
│                                                                      │
│  Perception tier: pixels in,       Analytics tier: coordinates in,   │
│  coordinates out.                  decisions out.                    │
└──────────────────────────────────────────────────────────────────────┘
                    THE JSONL LINE IS THE ONLY CONTRACT
```

The seam between them is one JSONL line per frame. Neither side imports the
other's logic; `footlab.io_radar` reads the file, nothing more. That is what
let us build and test the entire brain on simulation and SkillCorner data
before a single frame of our own video existed — and it is what will let the
vision side one day run on a drone or a Colab GPU while the brain runs
anywhere.

### 6b.1 Module flow, in detail

```
video.mp4
   │
   ▼
┌─────────────────────┐   models/player_detector.pt (fine-tuned YOLO, local)
│ detect_players_and  │   ──► players, ball (pixel bboxes + class)
│ _ball()             │
└─────────────────────┘
   │
   ▼
┌─────────────────────┐   models/football-pitch-detection.pt (32-kpt pose, local)
│ detect_keypoints()  │   ──► KeypointSet{kp_id → (px, py), conf}
└─────────────────────┘
   │
   ▼
┌─────────────────────┐   config.KEYPOINT_VERTICES_M maps kp_id → (x_m, y_m)
│ compute_homography()│   cv2.findHomography(all points, no RANSAC)
│  (transform.py)     │   ──► H (3×3), RMS reprojection error (meters)
└─────────────────────┘   reject if RMS > threshold (garbage in ≠ plan out)
   │
   ▼
┌─────────────────────┐   pixel (bx, by) → pitch (x_m, y_m) via
│ project_points()    │   cv2.perspectiveTransform; feet anchor = bbox
│                     │   bottom-center (where the player meets the ground)
└─────────────────────┘
   │
   ▼
┌─────────────────────┐   fit once per video: sample frames (stride 60),
│ teams_pipeline.py   │   crop outfield players, SigLIP embed → UMAP(3D)
│                     │   → KMeans(k=2); per frame: predict label per crop,
│                     │   GK by nearest team centroid, referees → -1,
│                     │   attacking side from GK's x (defends −x ⇒ attacks +x)
└─────────────────────┘
   │
   ▼
RadarFrame ──► write_jsonl() ──► data/radar_frame_N.jsonl
   │                                   (THE CONTRACT — one line per frame)
   ▼
footlab.io_radar.load_radar_jsonl() ──► FrozenFrame ──► every footlab phase
```

### 6b.2 The homography, honestly

A homography H is the 3×3 projective transform between two planes — here, the
pitch as the camera sees it and the pitch as it is. Eight degrees of freedom,
so ≥4 point correspondences; we feed every detected keypoint (≥6 required,
typically 10–15 on broadcast frames) and solve least-squares with
`cv2.findHomography(src, dst, 0)` — deliberately **not** RANSAC, because with
a good keypoint model outliers are rare, and RANSAC can lock onto a collinear
subset (e.g. only halfway-line points) and produce an H that fits those points
perfectly while being wrong everywhere else. Instead we validate with the RMS
reprojection error in **meters**: project the true landmark positions back
through H and measure the average miss. Under ~1 m is good for broadcast;
over 5 m means the model misidentified a landmark (left/right confusion), and
we reject the frame rather than feed the brain a warped world.

Two hard-won lessons now encoded in `config.py` / `transform.py`:

1. **The vertex map is ground truth — guard it.** A single swapped pair of
   keypoint→pitch coordinates (our 18/19 bug) silently degrades every frame.
   The demo's projection-check view (true landmarks as circles, projected
   points as X's) exists so this class of bug is *visible*, not statistical.
2. **Filter garbage, not confidence.** Undetected keypoints arrive at (0,0)
   and are masked; low-confidence points at real-but-wrong pixel positions
   poison least-squares, so a modest confidence floor (0.3) outperforms both
   "use everything" and "only high confidence" (which starves the solver of
   coverage — 7 clustered points fit their region beautifully and the rest of
   the pitch drifts).

### 6b.3 Team classification — why this stack

```
crop ─► SigLIP (frozen ViT, 768-dim embedding) ─► L2 normalize
     ─► UMAP (768 → 3, preserves neighbourhoods, supports .transform())
     ─► KMeans(k=2) fitted on outfield crops pooled across the video
```

- **SigLIP is never trained** — it is a frozen feature extractor. `.fit()`
  trains only UMAP's projection and KMeans's 2 centroids (~seconds). Same-
  jersey crops land near each other in embedding space; that is all we need.
- **UMAP over t-SNE** in the pipeline because only UMAP can `.transform()` new
  points at predict time. t-SNE lives in `viz_embeddings.py` as a *second
  opinion* for data auditing — two different math families agreeing on the
  cluster structure is how you learn to trust it.
- **Labels are arbitrary.** KMeans knows "two groups", not "home/away" or
  "attack/defend". `resolve_attack_side()` grounds the labels in football
  logic: a goalkeeper defends the goal he stands in front of, so his team's
  attacking direction is the opposite side. Fallback when no GK is visible:
  the team's defensive-depth percentile.
- **Referees are excluded** (`team = -1`) so they never contaminate KMeans or
  footlab's masks. footlab has no referee concept; keeping them out of the
  attack is the honest default.

### 6b.4 The JSONL contract (schema)

One line per frame, self-contained:

```json
{
  "frame_id": 80, "source": "08fd33_0.mp4",
  "pitch": {"length_m": 105.0, "width_m": 68.0, "origin": "center", "attack": "+x"},
  "homography": [[...3×3...] or null],
  "keypoints_used": [0, 5, 13, ...],
  "players": [{"track_id": 3, "class_name": "player", "team": "0",
               "pitch_xy_m": [-12.4, 8.1], "pitch_vxy_ms": [0.0, 0.0]}],
  "ball": {"pitch_xy_m": [2.1, -3.4]} | null,
  "carrier_track_id": 7 | null,
  "attacking_team": 1 | null
}
```

`team` ∈ `"0" | "1" | "referee" | null`; `attacking_team` says which label
attacks +x, and `io_radar` maps through it to `TEAM_ATTACK`/`TEAM_DEFEND`.
Frames with `homography: null` are skipped by the loader — a rejected frame is
a frame the brain never has to reason about.

### 6b.5 The exploration workbench (`exploration/`)

Experiments live off the production path: pluggable embedders (SigLIP v1/v2,
DINOv2), cluster-quality metrics (silhouette, Davies-Bouldin, Calinski-
Harabasch), 2D comparison grids and interactive 3D plotly plots. First result:
**DINOv2-small edges SigLIP v1 on jersey clustering at 10× smaller size** —
candidates for promotion into the pipeline graduate from here.

**The lesson for our own ingestion design:** there are two tiers. The
**perception tier** (steps 1–5) turns video into coordinates — that is exactly
what Path B (keypoint detection + homography) builds. The **analytics tier**
(step 6 onward) consumes coordinates — that is footlab, already built. Keep
them separate, just as SkillCorner does, and mirror their data shapes where
sensible (per-frame dicts of `{x, y, id, is_detected}`) so our future pipeline
drops into `FrozenFrame` the same way. Possession being an *inference* (and
sometimes `null`) is a tier-6 behaviour — so downstream code must always
tolerate a missing/loose carrier rather than assuming one exists.

---

## 7. How to run everything

### Setup

```bash
uv sync          # install deps into .venv
uv run pytest    # 133 tests, ~2 s
```

### Vision-module demos (own video → FrozenFrame)

```bash
# single frame: detection + homography + keypoint projection check
uv run python scripts/10_demo_radar_single.py --source data/08fd33_0.mp4 --frame 80

# the E2E milestone: video → teams → FrozenFrame → pitch control (real frame)
uv run python scripts/11_demo_e2e.py --source data/08fd33_0.mp4 --frame 80

# exploration workbench: embedder comparison (SigLIP v1/v2, DINOv2)
uv run python exploration/compare_embedders.py --source data/08fd33_0.mp4 --frame 80
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
