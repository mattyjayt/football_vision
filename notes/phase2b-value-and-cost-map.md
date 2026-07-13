# Phase 2b — Value surface & the combined cost map


## Goal of this phase

Pitch control (Phase 2) says where the ball is **safe**. It says nothing about
where it is **useful** — the middle of your own half can be 100% controlled and
worthless. This phase adds a **value** layer and fuses everything into the single
non-negative **cost** the Phase 3+ planners minimize.

## Value surface (xT-style surrogate)

Karun Singh's Expected Threat (xT) assigns each pitch location the probability a
possession there leads to a goal soon — learned from event data, rising toward
goal and jumping in the box. We don't fit data yet, so we use a **hand-crafted
analytic surrogate** with the same shape, in `[0,1]`:

```
value(x,y) = 0.40 · progress_up_pitch(x)
           + 0.55 · proximity_to_goal(x,y)      # exp(-dist_to_goal / 18 m)
           + 0.15 · inside_penalty_area(x,y)     # smooth box indicator
           (clipped to [0,1])
```

Three properties that make it behave like xT:
- **Monotone toward goal** — all three terms grow as `x → +52.5`.
- **Central-biased near goal** — `proximity_to_goal` uses distance to the goal
  *center*, so a wide angle by the goal is worth less than straight on (verified
  in a test).
- **Box bonus** — a smooth "inside the penalty area" bump.

It's a *static* property of the pitch (no players), exactly like an xT grid, so a
real data-driven xT can be dropped in later without touching the cost interface.

## Defender-proximity penalty

A cheap "don't run at a defender" term: each defender is a Gaussian bump
`exp(-d² / 2σ²)` (σ = 4 m) that is 1 at the defender and decays over a few
meters; the cell penalty is the **max over defenders**. Kept as pure spatial
proximity — velocity is already in pitch control, so projecting defenders forward
here would double-count (the `project_time` knob defaults to 0).

## The cost map

```
cost(cell) = w_control · (1 − pitch_control)
           + w_value   · (1 − value)
           + w_defender · defender_penalty
```

- Every term is in `[0,1]` and every weight `≥ 0`, so **cost is non-negative** —
  a hard requirement for Dijkstra / A* in Phase 3.
- The weights are the **tactical dial**, exposed via the `CostWeights` dataclass.

## What the demo showed

`figures/02b_cost_layers.png` lays out control | value | penalty | cost for the
counter-attack and low-block. The combined cost is **cheap (light) through open
attacking space and expensive (dark) at the defenders and the contested zone** —
precisely the landscape a planner should thread.

`figures/02b_cost_weights.png` is the key result — same frame, three weightings:

| weighting | cost range | behaviour it will induce |
|-----------|-----------|--------------------------|
| Balanced (1,1,1) | [0.69, 2.61] | compromise |
| Cautious (2, 0.5, 2) | [0.35, 4.29] | whole attacking half cheap, defenders strongly avoided → **safe wide routes** |
| Direct/risky (0.5, 2, 0.5) | [0.76, 2.24] | own half expensive, box cheapest, danger discounted → **direct routes at goal** |

Turning the weights visibly reshapes where the "cheap grass" is. That is the
mechanism Phase 3 uses to produce a safe route vs. a risky route from the *same*
frame.

## Failure modes / gotchas

- **Value normalization.** We normalize analytically (weights sum ~1, then clip)
  rather than by grid min/max, so `value_at_point` and the surface agree exactly
  and neither depends on grid resolution. Grid-relative normalization would have
  made the value of a point depend on which other cells you sampled — a subtle
  trap.
- **Cost is not in `[0,1]`.** It's a weighted sum, range `[0, Σw]`. That's fine
  for a planner (only relative costs matter) but don't mistake it for a
  probability.
- **Double-counting velocity.** Tempting to make the defender penalty
  velocity-aware, but pitch control already is — so the penalty stays static by
  default.

## Open questions (for later phases)

- Are these the right default weights? Phase 3 will tell us by whether the routes
  look tactically sane; expect to tune.
- Should cost use `value` or the *gain* in value along the path (xT added)? A path
  that ends where it started adds no threat even through high-value cells.
- A real xT grid (Singh's 16×12, or fit from open data) would replace `_value`
  wholesale — a clean future swap.

## How this connects to the end objective

This is the objective function. `cost_map` is literally what Phase 3 A\*, Phase 4
potential fields, and Phase 5 PSO each search or descend. Everything before this
phase was building the terrain; from here on we build things that move across it.
