# Phase 4 — Artificial potential fields (Khatib 1986)


## Goal of this phase

A completely different planning philosophy from A*, built partly **so we can watch
it fail**. No graph, no global search: treat the carrier as a charged particle,
let the goal attract it and defenders repel it, and roll downhill. It's reactive
and cheap — but it can get stuck. Understanding *why* is the learning goal.

## The field (Khatib 1986)

Total force = attractive (toward goal) + repulsive (away from each defender).

**Attractor.** Parabolic well near the goal so the pull eases as you arrive,
switching to a constant-magnitude "conic" pull far away so a distant goal doesn't
produce an absurd force:

```
||x − goal|| ≤ d0 :  F_att = k_att · (goal − x)
||x − goal|| > d0 :  F_att = k_att · d0 · (goal − x)/||goal − x||
```

**Repulsor** (per defender, active only within influence radius ρ0):

```
F_rep = k_rep · (1/ρ − 1/ρ0) · (1/ρ²) · (x − obs)/ρ ,   ρ ≤ ρ0
```

It blows up as `ρ → 0` (strong close-range push) and is exactly zero beyond ρ0
(far defenders ignored). **Velocity-aware:** each defender is projected forward
along its velocity by `project_time` before being treated as a static obstacle,
so you avoid where they're *going*.

**Path extraction:** from the carrier, step a fixed arc length in the force
direction each iteration until the goal is reached, the force vanishes, or
progress stalls.

## What the demo showed

`figures/04_potential_fields.png`:

- **A — Working.** The quiver flows toward goal, bulging around the two offset
  defenders; the descent curves by them and reaches goal.
- **B — Local-minimum trap.** A concave "cup" of three defenders `(2,0), (8,±3)`
  creates a spot where repulsion exactly cancels attraction. The descent stalls
  in open space at **x ≈ −0.5** and the goal is **never reached** — even though a
  path obviously exists. This is the headline failure.
- **C — A\* on the same frame.** Global search has no local minima, so A* finds a
  path to goal. Same situation, opposite outcome.

## Why the trap happens (the core insight)

On the approach axis, `F_att` points toward goal and the cup's `F_rep` points
back. There is a point where they are equal and opposite → **net zero force**.
Because the arrangement is concave, that equilibrium is *stable* in both axes: nudge
the particle and the field pushes it back in. Gradient descent is **local** — it
only knows the force here, not that a detour exists — so it settles and stops.

Two classic APF pathologies show up:
- **Local minima** (panel B) — the big one.
- **GNRON** ("goal non-reachable with obstacles nearby") — if a defender sits
  right on the goal (e.g. a keeper), its repulsion fights the attractor and the
  goal becomes unreachable. This is why the demo puts the goal in clear space.

A perfectly *colinear* single defender also traps (tested), but it's an unstable
saddle — real noise escapes it. The concave cup is a genuinely stable trap, which
is the more honest demonstration.

## APF vs. A* (the trade-off)

| | A* (Phase 3) | Potential fields (Phase 4) |
|---|---|---|
| search | global (graph) | local (follow the force) |
| optimality | optimal path | no guarantee |
| local minima | impossible | **the defining flaw** |
| cost | expand many nodes | evaluate one force per step |
| reactivity | replan from scratch | instantaneous, smooth |

APF isn't useless — its cheapness and smooth reactivity are exactly why it's used
for real-time robot control, and it will resurface in spirit in Phase 6 (defenders
reacting via simple pursuit forces). But as a *global* planner it's strictly
weaker than A* here.

## Failure modes / gotchas

- **Tuning is fiddly.** Whether a given arrangement traps depends on `k_rep`,
  `ρ0`, and geometry. A symmetric *pair with a gap* funnels through (no trap); a
  *concave cup* traps. Small parameter changes flip the outcome — itself a
  cautionary tale about APF robustness.
- **Step size vs. stall detection.** With a fixed step, a trap shows up as
  *oscillation* (the particle jitters around the minimum) rather than a clean
  stop, so we detect "distance-to-goal hasn't improved for `patience` steps"
  rather than only "force ≈ 0".

## Open questions (for later phases)

- Standard fixes exist (random restarts, wall-following, navigation functions,
  harmonic potentials) — worth a note but out of scope; A* already avoids the
  problem.
- Phase 5 (PSO over a Bézier) is another non-graph planner but *global* in its
  parameter search — does it dodge the local-minimum problem that sinks APF?

## How this connects to the end objective

Potential fields are an alternative planner to compare against the A* baseline —
and a concrete lesson in why global search matters when defenders sit between the
ball and goal. The velocity-aware repulsion idea (project defenders forward)
carries directly into Phase 6, where defenders actively move to intercept.
