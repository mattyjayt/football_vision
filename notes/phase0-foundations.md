# Phase 0 — Pitch, state, scenarios, plotting


## Goal of this phase

Lay the foundations everything else stands on: a **coordinate convention**, a
**state object** every algorithm consumes, **simulated scenarios** to develop
against, and **plotting** to see what we're doing. No tactics or optimization
yet — just make the data real and inspectable.

## The coordinate convention (memorize this)

- Pitch **105 m × 68 m**, **origin at the center spot**.
- `x ∈ [−52.5, +52.5]` along the length; `y ∈ [−34, +34]` across the width.
- **Attacking team always plays left → right**, toward the goal at **(+52.5, 0)**;
  defends the goal at **(−52.5, 0)**.
- Units: **meters** and **m/s** everywhere.

This is the Metrica / Friends-of-Tracking (Laurie Shaw `LaurieOnTracking`)
convention. Fixing it now means (a) real tracking data drops in later with no
rescaling, and (b) no algorithm ever needs a "which direction are we attacking?"
flag — it's baked into the sign of `x`.

Defined once in `src/footlab/pitch.py`; everything imports from there.

## The `FrozenFrame` (the one object that matters)

A moment of play = pure numbers:

| field | shape | meaning |
|-------|-------|---------|
| `positions` | (N, 2) | player positions, m |
| `velocities` | (N, 2) | player velocities, m/s |
| `team_ids` | (N,) | `0` = attack, `1` = defend |
| `ball_pos` | (2,) | ball position, m |
| `ball_carrier` | int | index of carrier, or `-1` if loose |

It validates its own shapes on construction (bad frame → `ValueError` at
creation, not deep inside a planner) and exposes convenience views
(`attacker_positions`, `speeds`, `carrier_position`, …). Numpy arrays, not
per-player objects, so later phases vectorize cleanly.

`MAX_PLAYER_SPEED = 9.0 m/s` is our sprint ceiling; `clip_speed()` rescales any
over-speed velocity while preserving direction. This is a placeholder kinematic
assumption — Phase 1/6 refine motion with reaction time and acceleration limits
(Fujimura & Sugihara 2005).

## Scenarios (`simulate.py`)

Hand-built `FrozenFrame`s, one per tactical idea, all attacking toward +x:

- **counter_attack** — 3 attackers vs 2 recovering defenders + GK; carrier on
  the halfway line; deliberate space in behind (the thing a planner should find).
- **low_block** — 10 compact defenders in the final third; carrier at the edge
  of the box; almost no direct lane (naive straight-line plans should fail here).
- **wing_overload** — 3v2 on the +y flank near the box.
- **random(seed)** — deterministic 11v11, noisy 4-3-3 pulled toward the ball,
  velocities capped at the sprint ceiling.

**Augmentation** (emulates my real YOLO+homography pipeline's noise, and expands
data for free):

- `mirror_lengthwise` — reflect `y → −y`. Roles preserved (obviously correct).
- `mirror_widthwise` — reflect `x → −x` **and swap team labels**, so the
  attack→+x convention still holds (it's the same picture from the other end).
- `jitter_positions(σ)` — Gaussian position noise (homography/detection error),
  clipped back onto the pitch.
- `perturb_velocities(σ)` — Gaussian velocity noise, re-capped to the speed ceiling.

All augmentation is **non-destructive** (returns a copy; original untouched).

## What the demo showed

`uv run python scripts/00_demo_scenarios.py` → `figures/00_scenarios.png`:

- Pitch markings render to scale, including the penalty arcs correctly bulging
  *out* of the box.
- Each scenario reads the way it should tactically; velocity arrows are
  1-second projections (arrow tip = where the player would be after 1 s).
- The mirrored random frame flips `y` (carrier ring moves to the opposite side),
  confirming the augmentation.

## Failure modes / gotchas noticed

- Velocity arrows scaled at `1.0` (one second) look long for a 9 m/s sprint
  (~9 m) but stay physically honest — I preferred honesty over prettiness.
- `is_on_pitch` uses a small tolerance so touchline points count as "on"; jitter
  then clips, so augmented frames never leak off the pitch.
- `mirror_widthwise` swapping teams is the subtle one — without it, "attack" would
  suddenly mean "moving toward −x" and every later phase would silently break.

## Open questions (for later phases)

- Should the ball ever be *off* the carrier's feet (in-flight pass)? Phase 0 puts
  it at the carrier's feet; pass evaluation (Phase 2/5) will need a free ball.
- Speed cap is a hard clip now; a smoother max-speed model (logistic, or
  acceleration-limited) may matter for the Phase 6 defender rollout.
- Do we need per-player `max_speed` (Spearman uses per-player estimates)? Deferred
  until pitch control (Phase 2) actually consumes it.

## How this connects to the end objective

Every downstream phase — Voronoi/dominant regions, pitch control, cost maps,
A*/potential-field/PSO planners, reactive defender rollouts — takes a
`FrozenFrame` in and draws onto a pitch `Axes`. Phase 0 is the shared vocabulary;
get the convention right here and nothing later has to second-guess it.
