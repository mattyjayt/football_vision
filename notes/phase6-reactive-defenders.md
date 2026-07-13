# Phase 6 — Reactive defenders: rollout & replanning (the capstone)


## Goal of this phase

Drop the lie every earlier phase told: that defenders stand still while we plan.
They react. This phase steps time forward, lets defenders pursue, and shows the
consequence — a static plan that looks optimal on the frozen frame gets cut off,
while **replanning** adapts and reaches goal.

## The simulation

Advance time in Δt = 0.2 s steps. Each step:
1. The **carrier** moves along its current path by `carrier_speed · Δt` (arc
   length along the polyline).
2. Each **defender** steers toward a **lead-pursuit** aim point —
   `carrier + carrier_velocity · lead_time` (where the carrier *will be*) — and
   updates its velocity under a **bounded acceleration** and a **top-speed cap**
   (Fujimura & Sugihara motion model). Vectorized over defenders.
3. **Interception** if any defender comes within `intercept_radius` of the carrier.

Two entry points:
- `rollout_static(frame, path, goal)` — roll a *fixed* path out against reacting
  defenders. Returns whether it reached goal / was intercepted and how far it got.
- `rollout_replanning(frame, centers, goal, replan_every=k)` — every k steps,
  rebuild a `FrozenFrame` from the *current* positions/velocities, recompute the
  pitch-control cost map, and re-run smoothed A* from the carrier's current spot.

Both return full `pos_history` + `plan_history` for animation.

## What the demo showed

`figures/06_reactive.gif` (and `06_reactive_summary.png`), **same opening frame**
— carrier behind a two-defender gate:

| | reached goal | intercepted | progress | time |
|---|---|---|---|---|
| **Static plan** | no | **yes** | 23% | 3.4 s |
| **Replanning** | **yes** | no | 99% | 13.6 s |

The static carrier follows the plan computed for the *initial* defender
positions; the defenders react, cut across, and tackle it at 23%. The replanning
carrier continuously re-routes away from where the defenders *actually are* now,
loops wide around them, and reaches goal. Same situation, opposite outcome — the
entire thesis of the project in one animation.

## Why this is the point of the whole project

Phases 1–5 answer "given this instant, what's the best path?" But football isn't
an instant — it's a pursuit. Phase 6 is the reality check:
- It turns every static plan into a **robustness score** (fraction completed
  before interception), making plans comparable under reaction.
- It shows **replanning** — the thing a real system must do — beating a
  one-shot plan.
- It directly answers the product question ("which route/pass survives the
  defense reacting?") that the frozen-frame analysis only approximates.

## Tuning notes / honest caveats

- **Speeds are equal (6.5 m/s each).** The carrier's early advantage comes only
  from the defenders' *standing start* (acceleration cap), not a speed cheat —
  which is the realistic reason a reacting defense is beatable at all. If
  defenders were much faster, no plan escapes (a fair conclusion, just not a
  demo).
- **Strong defender-avoidance weights** (`w_defender = 4`, `terrain_weight = 3`)
  make the replanner take a wide evasive loop. It's dramatic and it works;
  gentler weights give straighter, riskier routes (the Phase 2b/3 dial again).
- **Other attackers are frozen.** Only the carrier and defenders move; teammates
  hold position. Fine for this demo; a fuller sim would move everyone.
- **Pursuit is greedy lead-pursuit**, not optimal interception. Good enough to be
  a credible, reacting adversary; not a claim about how real defenders think.
- **Replanning cost.** Each replan recomputes the whole pitch-control surface,
  so the sim runs in seconds, not real time. That's fine offline; a real-time
  system would need the faster short-circuited pitch control (noted in Phase 2).

## Open questions / where this goes next

- **Game-theoretic planning:** the carrier plans assuming a defender *model*;
  the defender is really adversarial. Minimax / receding-horizon control is the
  principled version of "replan every k steps".
- **Pass evaluation:** the same rollout scores a *pass* (ball travels, receiver
  runs, defenders react) — the natural next use of this engine.
- **Real data:** with the drone→detection→homography pipeline, these rollouts run
  on real frames; the noise augmentation from Phase 0 is exactly the robustness
  test for that.

## How this closes the loop

This is the top of the stack: `FrozenFrame` → geometry → pitch control → cost map
→ planner → **rollout against a reacting defense → replanning**. Every module
built in isolation now composes into a system that behaves like the moving game.
The frozen-frame planners aren't wrong — they're the inner loop that replanning
calls repeatedly as the world changes.
