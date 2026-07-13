# Phase 3 — Graph search over the cost map (Dijkstra / A*) — the MVP


## Goal of this phase

The first thing that actually **moves**. Everything before built the terrain
(pitch control, value, cost); now we find the cheapest route across it for the
ball carrier. This is the project MVP: a tactically-aware optimal path to goal.

## The graph

The cost grid becomes an **8-connected weighted graph**. The cost of stepping
between adjacent cells is geometric distance times a per-meter cost:

```
edge(a → b) = ‖b − a‖ · (1 + terrain_weight · ½·(cost[a] + cost[b]))
```

- The **base `1`** is a distance term: all else equal, shorter is better. Without
  it, paths wander arbitrarily far through cheap grass to shave a rounding error.
- The **terrain term** adds danger. `terrain_weight` sets how much danger
  matters relative to distance (the demo turns it up to 2 to make routes bend).
- Summed along a route, this is a **line integral of danger-weighted distance**.

## Dijkstra vs. A* (Hart, Nilsson & Raphael 1968)

Both find the guaranteed-cheapest path. **Dijkstra** expands cells in order of
cost-so-far `g`. **A\*** expands in order of `f = g + h`, where `h` is an
optimistic estimate of the cost-to-go, so it heads toward the goal first.

Our heuristic is **straight-line distance to goal**. It is **admissible** (never
overestimates) because the minimum per-meter cost is exactly `1` (terrain ≥ 0),
so no route can cost less than its straight-line length. Admissible ⇒ A* returns
the **same optimal path** as Dijkstra.

Measured on the demo scenarios:

| scenario | cost (A* == Dijkstra) | nodes expanded A* / Dijkstra |
|----------|----------------------|------------------------------|
| Counter attack | 122.7 (equal ✓) | 3876 / 6872 |
| Low block | 48.9 (equal ✓) | 511 / 1698 |
| Wing overload | 49.4 (equal ✓) | 408 / 1643 |

A* expands **~40–75% fewer** nodes for the identical answer. The
`A*.cost == Dijkstra.cost` equality is asserted in the tests — that's the real
correctness guarantee.

## Path smoothing (and a bug it exposed)

An 8-connected grid only moves in 45° steps, so raw paths **staircase**. We
shortcut-smooth: greedily replace a subpath with a straight segment when the
segment's integrated cost is within `tol` of the original.

**The bug worth remembering:** my first version let a single shortcut span the
*whole* path. It happily straightened a genuine danger-avoiding detour **back
through the defenders** — because the continuous line integral used for
smoothing differs slightly from the planner's discrete edge cost (they only agree
for orthogonal steps; a diagonal cuts through the other two corners of its 2×2
block, so the metrics diverge). Fix: **cap each shortcut to a local `window` of
waypoints**. That removes jaggedness while preserving the route's overall shape.
Lesson: a post-process must not be free to undo the optimizer's decisions under a
subtly different cost model.

## What the demos showed

`figures/03_grid_search_paths.png` — the smoothed A* path per scenario. It
threads the gap *between* the two off-center defenders in the counter-attack, and
curves around the cluster in the wing overload.

`figures/03_safe_vs_risky.png` — **the headline result.** On one frame (a
defensive wall with a gap on the +y side), the *same planner* produces:
- **Cautious** weights `(w_control, w_value, w_defender) = (2, 0.5, 3)` → a wide
  arc over the wall through the gap (max |y| ≈ 15.5 m).
- **Direct / risky** `(1, 2, 0.4)` → straight through the wall to goal
  (max |y| ≈ 0.5 m).

Turning the Phase 2b weights turns a safe route into a risky one. That is the
whole thesis of the cost-map approach, made visible.

## Failure modes / honest findings

- **A lone defender is never worth a big detour.** In the low-block, the center
  is genuinely costlier at the blocking midfielder, but detouring ~8 m to avoid
  one localized bump costs more *distance* than it saves — so the optimal route
  goes fairly straight. Route-bending needs a **sustained barrier**, which is why
  the safe-vs-risky demo uses a purpose-built wall, not a canonical scenario.
- **Two cost metrics.** The planner (discrete edges) and `path_cost` (continuous
  bilinear integral) are close but not identical on diagonals. Keep them
  conceptually separate; don't assert equality except on orthogonal paths (a test
  does exactly that).
- **Static defenders.** The whole plan assumes a frozen frame — the defenders
  don't react. That assumption is wrong, and fixing it is the entire point of
  Phase 6.

## Open questions (for later phases)

- The 8-connected grid caps turn resolution at 45°; smoothing helps but a
  sampling planner (RRT*) or a parametric one (PSO over a Bézier, Phase 5) gives
  naturally smooth curves. How do they compare on cost and runtime?
- Should the "goal" always be the goal mouth? For passing-option evaluation the
  target is a teammate or a space — same planner, different `goal_xy`.
- `terrain_weight` and the Phase 2b weights interact. Is there a principled way to
  set them, or is it inherently a tactical preference to expose?

## How this connects to the end objective

This is the MVP of the entire project: frozen frame → cost map → optimal
tactical path. Phases 4 (potential fields) and 5 (PSO) are *alternative* planners
solving the same problem, to be compared against this A* baseline. Phase 6 drops
the frozen-frame assumption and makes the defenders react — at which point this
planner gets re-run in a loop (replanning).
