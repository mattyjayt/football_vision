# Phase 2 — Spearman pitch control (PPCF)


## Goal of this phase

Turn Phase 1's hard "who owns this cell" into a smooth **probability the
attacking team would control the ball if it arrived at that cell**,
`P_att(cell) ∈ [0,1]`. That soft surface is the raw material for the Phase 2b
cost map: safe grass is high `P_att`, dangerous grass is low.

## The model (Spearman 2018, Eq. 3–5)

Three ingredients:

1. **Time-to-intercept (TTI)** — reused verbatim from Phase 1
   (`geometry.time_to_arrive`): `τ + ‖target − (p + v·τ)‖ / v_max`. Momentum is
   already baked in, so this surface inherits the velocity-awareness of the
   dominant region.

2. **Arrival is uncertain.** A player isn't instantly "present" at exactly their
   TTI. The probability they have arrived *by* time `T` is a logistic ramp
   centered on their TTI:

   ```
   f_j(T) = σ( (π / (√3 · σ_tti)) · (T − TTI_j) )
   ```

   `σ_tti = 0.45 s` sets the width. The `π/√3` factor makes the logistic match a
   Gaussian of that standard deviation.

3. **Control accrues over time via coupled ODEs.** Integrate time forward from
   the moment the ball could arrive (`ball_travel_time = ‖target − ball‖ / 15`):

   ```
   dPPCF_j/dT = (1 − Σ_k PPCF_k(T)) · f_j(T) · λ_j
   ```

   - `(1 − Σ PPCF)` = probability the ball is *still loose*. This is the coupling:
     once someone starts controlling it, there's less left for everyone else.
   - `λ_j ≈ 4.3 s⁻¹` = rate a present player converts "loose" into "controlled".

   As `T → ∞` every `f_j → 1`, the loose probability decays to 0, and
   `P_att + P_def → 1`: the surface is a partition of unity.

## Two implementations, and why

- **`pitch_control_at_target`** — the readable reference. Loops over players,
  integrates one target. This is the one to read to understand the model.
- **`pitch_control_surface`** — the same math vectorized with numpy so all ~7000
  grid cells integrate together (whole 105×68 surface in **<0.3 s**).

A test (`test_surface_matches_reference_pointwise`) asserts they agree to 1e-2 on
random cells. **The vectorized version is only trustworthy because it matches the
obvious one** — that equivalence test is the real correctness guarantee here.

## Parameters (Spearman 2018 / LaurieOnTracking defaults)

| param | value | meaning |
|-------|-------|---------|
| `reaction_time` | 0.7 s | coast at current velocity before redirecting |
| `max_speed` | 5.0 m/s | effective running speed to a target |
| `tti_sigma` | 0.45 s | TTI uncertainty (logistic width) |
| `lambda_att`, `lambda_def` | 4.3 s⁻¹ | control-accrual rate (kappa_def = 1) |
| `average_ball_speed` | 15 m/s | assumed ball speed to the target |
| `int_dt` | 0.04 s | Euler step |

Stored in the frozen `PitchControlParams` dataclass so a demo can sweep them.

## What the demo showed

`figures/02_pitch_control.png`, mean attacking control:

| scenario | mean `P_att` | at carrier's feet |
|----------|-------------|-------------------|
| Counter attack | 68.1% | 0.99 |
| Low block | 84.7% | 0.84 |
| Wing overload | 91.3% | 0.98 |
| Random 11v11 | 50.4% | 0.98 |

Reads to check that convinced me it's right:
- **Random 11v11 → 50.4%.** Two mirror-image formations should split the pitch
  ~50/50, and they do. Strong sanity check.
- **Counter attack 68.1%** matches the Phase 1 dominant-region share (68.0%) — the
  probabilistic and categorical models agree on the balance of territory.
- **Low block** paints almost the whole pitch red *except* a tight blue pocket
  over the compact defenders at the box: cede territory, defend the danger zone.
- Every ball carrier controls their own feet (`~0.98`), as they must.

## Failure modes / gotchas

- **Numerical overflow** in the logistic: use `scipy.special.expit`, not a raw
  `1/(1+exp(...))`, or large negative arguments overflow.
- **Discrete overshoot**: clip the "loose" probability to `[0,1]` each step so the
  surface stays a valid probability and accrual stays monotone.
- **Integration window**: far cells need a longer window to converge to
  `sum = 1`. We set `t_max = max(ball_time, min TTI) + max_int_time` per cell so
  every cell's earliest arriver is covered.
- The **space "in behind" a recovering defense is not free** once you include the
  GK and the defenders' backward momentum — the counter-attack panel shows that
  zone going blue. Realistic, and a caution against reading Phase 1/2 too
  greedily.

## Simplifications vs. the reference implementation

- No short-circuit for cells one team clearly wins (we always integrate). Slower
  in principle, but vectorization makes it a non-issue here.
- Single global `max_speed` rather than per-player estimates.
- `kappa_def = 1` (no defensive control advantage); the paper uses 1.72.

## Open questions (for later phases)

- Phase 2b needs a *value* surface (xT-style) and a way to combine
  `1 − P_control`, `1 − value`, and a defender-proximity penalty into one cost
  grid. What weights make the A* routes look tactically sane?
- Should the planner use `P_att` directly, or the *gradient* of `P_att` (space
  being created/lost)? Fernández & Bornn's "Wide Open Spaces" argues for the
  latter — worth revisiting.

## How this connects to the end objective

`pitch_control_surface` is the safety layer of the planner's cost map. Combined
in Phase 2b with a value surface, it becomes `cost(cell)` that the Phase 3 A\*
search, Phase 4 potential field, and Phase 5 PSO all optimize a path through. The
`time_to_arrive` primitive it shares with Phase 1 will appear again in the Phase 6
reactive-defender rollout.
