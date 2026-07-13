# Phase 1 — Space geometry: Voronoi vs. dominant regions


## Goal of this phase

Turn a frozen frame into a **map of who owns each patch of grass**. That
ownership map is the conceptual foundation for every planner later: a safe path
stays in space your team controls; a risky one cuts through space the opponent
owns. Phase 2 (pitch control) makes this map *probabilistic* and Phase 2b turns
it into a *cost*; here we build the crisp, categorical version first.

## Two ways to assign ownership

### Voronoi (nearest player) — `voronoi_control`
Each grid cell goes to the **closest player by straight-line distance**. That's
the textbook Voronoi partition. We compute it on the `pitch.make_grid` grid by
taking, per cell, the `argmin` over players of Euclidean distance. `scipy.
spatial.Voronoi` (`geometry.voronoi_diagram`) gives the *exact* polygon edges; in
the demo we overlay those finite ridge segments on the grid map and they line up,
which is a nice correctness check (grid `argmin` == exact Voronoi, up to cell
resolution).

**Blind spot:** it ignores momentum. A defender sprinting the wrong way still
"owns" the ground right behind him, which is obviously false.

### Dominant region (soonest to arrive) — `dominant_region`
Replace "who is *closest*?" with "**who can get there *soonest*?**". That needs a
tiny motion model to convert distance → time.

## The motion model (key equation)

For a player at position `p` with velocity `v`, the time to reach a target `x`:

```
t(x) = τ + || x − (p + v·τ) || / v_max
```

- `τ` = reaction time (default **0.7 s**): the player *coasts at current velocity*
  before redirecting, so the effective start point is `p + v·τ`.
- `v_max` = control speed (default **5.0 m/s**).

The `v·τ` term is the whole point: a moving player's reachable set is shifted in
their direction of travel, so their region **stretches ahead and shrinks behind**.

Two subtleties worth internalizing:
- The additive `τ` is the *same for everyone*, so it **cancels in the `argmin`** —
  it doesn't change *who* owns a cell, only the absolute arrival time (which
  Phase 2 reuses for time-to-intercept). Verified in a test.
- There are **two different "max speeds"** in the repo. `state.MAX_PLAYER_SPEED
  = 9 m/s` is the instantaneous *sprint cap* used to generate plausible
  velocities. `geometry.CONTROL_MAX_SPEED = 5 m/s` is the *effective running
  speed toward a target* used by control models (Spearman's default). Different
  jobs; kept separate on purpose.

Simplification vs. Fujimura & Sugihara (2005): no explicit acceleration ramp or
drag — the reaction phase is pure coasting, then instant top speed. Good enough
to show the qualitative effect; refine later if Phase 6 needs it.

## What the demo showed

`figures/01_voronoi_vs_dominant.png` (left = Voronoi, right = dominant):

| scenario | attack share (Voronoi → dominant) | cells that changed owner |
|----------|-----------------------------------|--------------------------|
| Counter attack | 64.5% → 68.0% | 6.7% |
| Wing overload  | 96.6% → 97.7% | 2.9% |

In the counter-attack the two defenders are *recovering* (moving toward their own
goal at +x, i.e. away from the ball). Velocity-awareness makes them **concede the
space in behind** — exactly the running lane a counter-attack planner wants — so
attack's share grows. The boundary visibly bows toward the defenders' goal on the
right panel.

The dominant-region boundary is **jagged** at 1 m resolution — that's grid
sampling, not a bug. Finer cells smooth it at the cost of compute.

## Failure modes / gotchas

- `scipy`'s `voronoi_plot_2d` draws *unbounded* ridges to infinity, which spray
  across a multi-panel figure. Fix: draw only the finite ridge segments by hand.
- `scipy.spatial.Voronoi` needs ≥ 4 non-degenerate points and can raise
  `QhullError` on collinear inputs — the demo wraps the overlay in try/except.
  The grid-based `voronoi_control` has no such fragility, which is a point in its
  favor as the analysis workhorse.

## Open questions (for later phases)

- Our model gives a *hard* winner per cell. Spearman (Phase 2) makes control a
  smooth probability via a logistic on the time-to-intercept *difference* — that
  softness is what makes the cost map differentiable-ish and more realistic.
- Should `v_max` be per-player (Spearman uses per-player estimates)? Deferred
  until we actually have per-player speed data.
- Is straight-line-at-top-speed too generous for a player who has to turn 180°?
  A turn/acceleration penalty would shrink the "behind" region further.

## How this connects to the end objective

`dominant_region` is the categorical ancestor of the **pitch-control surface**
(Phase 2) that feeds the **cost map** (Phase 2b) that the **A\*/potential-field/
PSO planners** (Phases 3–5) search over. The `time_to_arrive` function written
here is the same time-to-intercept primitive Spearman's model and the Phase 6
reactive-defender rollout will reuse — so this phase is load-bearing well beyond
its own demo.
