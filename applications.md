# Applications & value added

`footlab` turns a moment of football — where the players and ball are, and how
they're moving — into quantitative answers about **space, threat, and the best
thing the ball carrier could do**, accounting for a defense that reacts. This
document lays out who that is useful for and why.

The building blocks it produces are reusable on their own:

- **Space ownership** (Voronoi / dominant regions) — who controls each patch of
  grass, accounting for momentum.
- **Pitch control** (Spearman PPCF) — the probability each team would win the ball
  at any location, as a smooth surface.
- **Positional value** (xT-style) — how dangerous each location is.
- **A cost map** fusing safety, value, and defender proximity into one tunable
  objective.
- **Planners** (A*, potential fields, PSO) — optimal paths across that objective.
- **Reactive rollout + replanning** — how a plan actually fares against defenders
  that chase, and how re-planning beats committing to one route.

Combined with the intended upstream pipeline (drone footage → pitch/player/ball
detection → homography → `FrozenFrame`), a short clip becomes annotated,
simulated tactical insight.

---

## Where it delivers value

### 1. Coaching & tactical analysis (post-match)
Answer *"what should the carrier have done here?"* concretely. Overlay the optimal
route to goal, show the space that was available, and quantify how much better a
different decision would have been. The safe-vs-risky weighting turns coaching
philosophy (patient buildup vs. direct verticality) into a dial you can *see* on
the same frame.

### 2. Opponent scouting & defensive-shape analysis
Pitch control and dominant regions **quantify the space a defense concedes**.
Which zones does this opponent's low block leave open? How much territory does
their high line give up in behind? Aggregate over a match to profile a team's
defensive structure numerically instead of by eye.

### 3. Player decision & recruitment evaluation
Score a player's *actual* decisions against the model's optimum: did they pick the
best route, hold the ball too long, or miss a higher-value lane? Done at scale,
this is a decision-quality metric for recruitment — separating players who
*create* value from those who merely *finish* it.

### 4. Automated insight & highlight generation (the drone-pipeline payoff)
Feed a ~5-minute clip through detection + homography and let `footlab` annotate
moments automatically: pitch-control heatmaps, the optimal path overlaid, the
rollout showing whether a plan survives the defense. Turns raw footage into a
tactics reel without a human analyst scrubbing frame by frame.

### 5. Broadcast & fan engagement
The same overlays — space control shading, "the pass that was on", the optimal run
— are compelling on-screen graphics that explain *why* a moment worked or didn't,
in real broadcast language.

### 6. Academy & player development
Pitch-control and dominant-region visuals are a **teaching tool**: they make the
invisible (space, time-to-arrive, the value of a run) visible, helping young
players learn positioning and decision-making.

### 7. Research & education
The repo itself is a worked, cited, test-backed reference for **spatial
optimization and sports analytics** — dominant regions, pitch control, cost-map
path planning, potential fields, PSO, and pursuit/replanning — each isolated and
runnable. A strong base to fork for experiments.

---

## Three concrete examples

**A. The counter-attack gap that closes.**
A frozen-frame planner threads a route between two recovering defenders. It looks
perfect. The Phase 6 rollout shows the defenders react and close the gap *before
the carrier arrives* — the static plan is intercepted at ~25% progress.
Replanning re-routes around the shifting defenders and reaches goal. Value: it
distinguishes a plan that only works on paper from one that survives contact,
and it teaches *timing*, not just geometry.

**B. Breaking a low block.**
Against ten compact defenders, pitch control shows the attack owns ~85% of the
pitch but not the dangerous zone. The cost map with cautious weights routes wide
and patient; with aggressive weights it drives directly at the box. Value:
quantifies exactly where the block is vulnerable and makes the risk/reward of each
route explicit.

**C. Evaluating passing options.**
Which of several lanes actually survives the defense reacting? A static
pitch-control snapshot may say lane B is open; a rollout of the pass (ball
travels, receiver runs, defenders close) can reveal it shuts in 0.4 s while lane A
holds. Value: ranks options by *robustness under reaction*, which is the real
question, not by an instantaneous snapshot.

---

## Value by stakeholder

| Stakeholder | What they get |
|-------------|---------------|
| Coaches / analysts | Objective answers on space, routes, and decision quality; a tunable tactical dial |
| Players / academies | Visual, teachable feedback on positioning and movement |
| Clubs / recruitment | Scalable decision-quality metrics beyond box-score stats |
| Broadcasters / media | Explanatory overlays that make tactics legible to fans |
| Researchers / students | A cited, tested, modular reference for spatial optimization |

---

## Honest limitations (what productionizing would need)

- **Upstream accuracy.** Everything depends on detection + homography quality;
  the 45°-angle drone view will be noisier than top-down. The repo's noise
  augmentation (`jitter_positions`, `perturb_velocities`) is exactly the tool to
  test robustness to this, but garbage-in still means garbage-out.
- **Model calibration.** Pitch-control and value parameters are literature
  defaults / hand-crafted surrogates, not fit to *your* league. A data-driven xT
  and per-player speed estimates would sharpen results.
- **Reactive realism.** Defenders here use greedy lead-pursuit, not real defensive
  intelligence. It is a credible adversary for stress-testing, not a claim about
  how a specific team defends.
- **Compute.** Replanning recomputes the pitch-control surface each step, so the
  current rollout runs offline in seconds, not in real time. A real-time assistant
  would need the short-circuited pitch-control optimization.

None of these are blockers for the analysis, coaching, and education use cases —
they are the roadmap for turning the research brain into a deployed product.
