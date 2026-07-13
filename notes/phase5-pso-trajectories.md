# Phase 5 — PSO over Bezier trajectory parameters


## Goal of this phase

A third planner, and a direct **benchmark against the A\* baseline**. Where A*
searches a discrete grid and potential fields follow a local force, PSO globally
searches a tiny *continuous* space of smooth curve shapes.

## The parameterization (the whole trick)

Describe the path as a **cubic Bezier**:

```
B(t) = (1−t)³·start + 3(1−t)²t·P1 + 3(1−t)t²·P2 + t³·goal ,  t ∈ [0,1]
```

Start (carrier) and goal are fixed; only the two interior control points **P1,
P2** move. That's **4 numbers**. Planning = find the 4 numbers whose curve is
cheapest. A whole path collapses to a 4-D optimization.

## Fitness (reuses the Phase 2b cost)

```
fitness = ∫ cost along the curve         (reuses the grid_search edge model)
        + w_curvature · total_turning     (drivability / turn-radius proxy)
        + w_offpitch  · off-pitch overflow
```

The cost integral is the *same* one A* minimizes, so the comparison is
apples-to-apples. It's a black box (non-differentiable), which is exactly the
regime where PSO is meant to beat gradient methods.

## PSO from scratch (Kennedy-Eberhart 1995; Shi-Eberhart 1998)

Scatter a swarm of 4-vectors. Each particle moves under momentum plus attraction
to its own best and the swarm's global best:

```
v ← w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)
x ← x + v
```

The **inertia weight** `w` is linearly decayed 0.9 → 0.4 (explore early, exploit
late). Positions clamped to the pitch; velocities clamped to a fraction of the
range. ~60 lines, no dependencies.

## What the benchmark showed (the honest result)

`figures/05_pso_vs_astar.png`, pure cost-map cost / total turning / runtime:

| scenario | A* cost | PSO cost | A* turn | PSO turn | A* time | PSO time |
|----------|--------:|---------:|--------:|---------:|--------:|---------:|
| Counter attack | 122.7 | 124.1 | 0.00 | 0.05 | 0.024 | 0.119 |
| Low block | 48.9 | 50.4 | 0.00 | 0.00 | 0.004 | 0.120 |
| Wing overload | 49.0 | 52.7 | 0.32 | 0.00 | 0.003 | 0.118 |
| Defensive wall | 197.7 | 198.7 | 0.00 | 0.00 | 0.038 | 0.125 |

**A\* is the stronger baseline here** — it wins on cost in every scenario (it's
grid-optimal) and is 5–40× faster. PSO's paths are close (1–7% costlier) and
smooth, but it does not beat A*. This **verifies** the hypothesis in the brief.

Nuances worth keeping:
- In the wing overload, A* found a slightly *curved* cheaper route (cost 49.0)
  that PSO's 4-parameter Bezier missed, settling for a near-straight 52.7. Low
  dimensionality is a double edge: fast to search, but it can't represent every
  route A* can.
- We *smooth* the A* path, which erases PSO's main selling point (inherent
  smoothness). Against **raw** (staircased) A*, PSO would look smoother — but the
  fair comparison is against smoothed A*, and there PSO has no edge.

## Convergence vs. swarm size

`figures/05_pso_convergence.png`: all swarm sizes (10/30/60) converge to the same
optimum (198.7). Bigger swarms start from a better best (more initial samples) and
converge more smoothly, but for a **4-D** problem even 10 particles suffices. The
global-best history is **monotonically non-increasing** by construction (asserted
in a test) — you can read that straight off the curves.

## When would PSO actually win?

Not on this grid. PSO earns its keep when:
- the fitness is genuinely **non-differentiable / discontinuous** and there's no
  clean graph to search (A*'s requirement),
- the parameterization is **low-dimensional** and you want a **naturally smooth**
  trajectory without a post-hoc smoother,
- you need to optimize things a grid can't express (e.g. a speed profile along the
  curve, or joint pass-then-run trajectories).

For "cheapest route across a cost grid", A* is the right tool.

## Failure modes / gotchas

- **Speed.** Naively reusing `grid_search.path_cost` (a per-segment Python loop)
  made the tests take 18 s. Vectorizing the cost integral over the sampled Bezier
  (`_polyline_cost`) cut it to <1 s. PSO evaluates fitness thousands of times, so
  the inner loop must be vectorized.
- **Feasibility is only a curvature proxy.** With no time parameter we can't
  enforce a true max-speed/turn constraint; `w_curvature` stands in for it.
- **Local dimensionality trap.** 2 control points can't represent an S-curve
  through multiple gaps; more control points add dimensions (slower, harder
  search). The parameterization caps what PSO can find.

## Open questions

- Would 3 control points (quartic, 6-D) close the wing-overload gap vs. A*, or
  just slow convergence? Easy to test by extending the encoding.
- CMA-ES (stretch goal) on the same fitness — better sample efficiency than PSO?

## How this connects to the end objective

PSO is the third and final *static* planner. The takeaway for the project: **A* is
the default planner**; PSO/APF are valuable as comparisons and for regimes A*
can't handle. Next, Phase 6 drops the frozen-frame assumption entirely — defenders
react — which is where all these static plans get stress-tested and replanning
enters.
