"""Scenario generators and augmentation helpers (Phase 0 data strategy).

A ``FrozenFrame`` is just positions + velocities, so we can hand-build tactically
meaningful situations without any tracking data. These give every later phase a
deterministic, inspectable input.

Four scenarios (all with the attacking team playing toward +x, defending goal at
+52.5):

    * ``scenario_counter_attack`` — 3 attackers vs 2 defenders + GK, carrier on
      the halfway line, space in behind.
    * ``scenario_low_block``      — 10 defenders compact in their own third,
      carrier at the edge of the box.
    * ``scenario_wing_overload``  — 3 attackers vs 2 defenders on the flank.
    * ``scenario_random``         — a plausible 11v11 frame, formation-ish with
      players pulled toward the ball, seeded for reproducibility.

Augmentation helpers (``mirror_lengthwise``, ``mirror_widthwise``,
``jitter_positions``, ``perturb_velocities``) exist to stress-test downstream
methods and to emulate the homography/detection noise of a real YOLO+homography
pipeline.
"""

from __future__ import annotations

import numpy as np

from . import pitch
from .state import (
    MAX_PLAYER_SPEED,
    TEAM_ATTACK,
    TEAM_DEFEND,
    FrozenFrame,
    clip_speed,
)


def _nearest_attacker_to(positions: np.ndarray, team_ids: np.ndarray,
                         point: np.ndarray) -> int:
    """Index of the attacking player closest to ``point`` (the ball carrier)."""
    att_idx = np.where(team_ids == TEAM_ATTACK)[0]
    d = np.linalg.norm(positions[att_idx] - point, axis=1)
    return int(att_idx[int(np.argmin(d))])


# --- Hand-built scenarios ----------------------------------------------------
def scenario_counter_attack() -> FrozenFrame:
    """3v2 (+GK) counter attack, carrier on the halfway line with space behind.

    The two defenders are still recovering (velocity toward their own goal, +x),
    the attackers are breaking forward at pace, and the space between the last
    defender and the GK is the tactical prize a planner should exploit.
    """
    positions = np.array([
        [0.0,   0.0],    # 0 carrier, on halfway line
        [3.0,  16.0],    # 1 support, wide right
        [1.0, -14.0],    # 2 support, wide left
        [24.0,  6.0],    # 3 defender, recovering
        [22.0, -8.0],    # 4 defender, recovering
        [50.0,  0.0],    # 5 goalkeeper
    ])
    velocities = np.array([
        [6.5,  0.0],
        [6.0,  0.5],
        [6.2, -0.4],
        [4.5,  1.0],   # defenders sprinting back toward +x
        [4.8, -0.6],
        [0.5,  0.0],
    ])
    team_ids = np.array([TEAM_ATTACK, TEAM_ATTACK, TEAM_ATTACK,
                         TEAM_DEFEND, TEAM_DEFEND, TEAM_DEFEND])
    ball_pos = positions[0].copy()
    return FrozenFrame(positions, velocities, team_ids, ball_pos, ball_carrier=0)


def scenario_low_block() -> FrozenFrame:
    """Compact 10-defender low block; carrier at the edge of the penalty area.

    Two attacking support runners are included, but the point of this frame is a
    dense wall of defenders in the final third with almost no direct lane to
    goal — the case where a naive straight-line plan should fail.
    """
    goal_x = pitch.HALF_LENGTH
    pa_x = goal_x - pitch.PENALTY_AREA_LENGTH  # 36.0, edge of the box

    # Carrier just outside the box, two support runners either side.
    attackers = np.array([
        [pa_x - 1.0,  0.0],   # 0 carrier at the top of the box
        [pa_x - 3.0, 14.0],   # 1 support right
        [pa_x - 4.0, -12.0],  # 2 support left
    ])
    # 10 defenders: a back line on the 6-yard box + a screen in front + GK.
    defenders = np.array([
        [goal_x - 5.5, -8.0], [goal_x - 5.5, -3.0],
        [goal_x - 5.5,  3.0], [goal_x - 5.5,  8.0],   # deep line (4)
        [goal_x - 10.0, -12.0], [goal_x - 10.0, -4.0],
        [goal_x - 10.0,  4.0], [goal_x - 10.0, 12.0],  # screening line (4)
        [goal_x - 14.0,  0.0],                          # holding mid (1)
        [goal_x - 1.0,  0.0],                           # goalkeeper (1)
    ])
    positions = np.vstack([attackers, defenders])
    velocities = np.zeros_like(positions)
    # Slow shuffling defenders, drifting toward the ball side.
    velocities[3:] = np.array([-0.5, 0.0])
    team_ids = np.array([TEAM_ATTACK] * len(attackers)
                        + [TEAM_DEFEND] * len(defenders))
    ball_pos = attackers[0].copy()
    return FrozenFrame(positions, velocities, team_ids, ball_pos, ball_carrier=0)


def scenario_wing_overload() -> FrozenFrame:
    """3v2 overload on the right flank (high +y), near the attacking box."""
    attackers = np.array([
        [32.0, 24.0],   # 0 carrier, wide right
        [38.0, 14.0],   # 1 underlapping runner
        [40.0, 28.0],   # 2 overlapping runner, hugging touchline
    ])
    defenders = np.array([
        [40.0, 20.0],   # 3 covering fullback
        [44.0, 26.0],   # 4 recovering winger
    ])
    positions = np.vstack([attackers, defenders])
    velocities = np.array([
        [3.5,  1.0],
        [4.0, -0.5],
        [4.5,  0.3],
        [1.0, -1.5],
        [-1.0, -3.0],
    ])
    team_ids = np.array([TEAM_ATTACK] * 3 + [TEAM_DEFEND] * 2)
    ball_pos = attackers[0].copy()
    return FrozenFrame(positions, velocities, team_ids, ball_pos, ball_carrier=0)


def _formation_positions(rng: np.random.Generator, *, attacking: bool
                         ) -> np.ndarray:
    """11 positions in a rough 4-3-3, oriented for the given team.

    The attacking team (toward +x) sits on the -x side pushing forward; the
    defending team mirrors it on the +x side. Small Gaussian noise breaks the
    grid so frames look organic.
    """
    # Columns are (x_line, y) in "attacking" orientation (goal at +x ahead).
    template = np.array([
        [-48.0,  0.0],                                   # GK
        [-32.0, -20.0], [-34.0, -7.0],
        [-34.0,  7.0], [-32.0, 20.0],                    # back 4
        [-14.0, -14.0], [-16.0, 0.0], [-14.0, 14.0],     # mid 3
        [4.0, -18.0], [8.0, 0.0], [4.0, 18.0],           # front 3
    ])
    pos = template + rng.normal(0.0, 2.5, size=template.shape)
    if not attacking:
        pos[:, 0] *= -1.0  # reflect onto the +x half; now defending goal at +x
    return pos


def scenario_random(seed: int = 0) -> FrozenFrame:
    """A plausible full 11v11 frame; deterministic given ``seed``.

    Players follow a noisy 4-3-3 and are then nudged toward the ball to mimic
    play collapsing around it. Velocities are random but capped at
    ``MAX_PLAYER_SPEED``. The carrier is the attacker nearest the ball.
    """
    rng = np.random.default_rng(seed)

    attackers = _formation_positions(rng, attacking=True)
    defenders = _formation_positions(rng, attacking=False)
    positions = np.vstack([attackers, defenders])
    team_ids = np.array([TEAM_ATTACK] * 11 + [TEAM_DEFEND] * 11)

    # Ball somewhere in the middle third; pull everyone (except keepers) toward
    # it a little so the frame clusters around the action.
    ball_pos = rng.uniform([-20.0, -15.0], [20.0, 15.0])
    pull = 0.15
    for i in range(positions.shape[0]):
        if i in (0, 11):  # keepers stay home
            continue
        positions[i] += pull * (ball_pos - positions[i])

    positions = pitch.clip_to_pitch(positions)

    # Random velocities, mostly modest, capped at the sprint ceiling.
    velocities = rng.normal(0.0, 2.5, size=positions.shape)
    velocities = clip_speed(velocities, MAX_PLAYER_SPEED)
    velocities[0] = velocities[11] = 0.0  # keepers roughly static

    carrier = _nearest_attacker_to(positions, team_ids, ball_pos)
    ball_pos = positions[carrier].copy()  # ball sits at the carrier's feet
    return FrozenFrame(positions, velocities, team_ids, ball_pos,
                       ball_carrier=carrier)


ALL_SCENARIOS = {
    "counter_attack": scenario_counter_attack,
    "low_block": scenario_low_block,
    "wing_overload": scenario_wing_overload,
    "random": lambda: scenario_random(seed=0),
}


# --- Augmentation helpers ----------------------------------------------------
def mirror_lengthwise(frame: FrozenFrame) -> FrozenFrame:
    """Reflect across the halfway axis (y -> -y): swaps left/right wings.

    Attacking direction is unchanged, so team roles are preserved. This is the
    "obviously correct" augmentation — no relabeling needed.
    """
    out = frame.copy()
    out.positions[:, 1] *= -1.0
    out.velocities[:, 1] *= -1.0
    out.ball_pos[1] *= -1.0
    return out


def mirror_widthwise(frame: FrozenFrame) -> FrozenFrame:
    """Reflect across the width axis (x -> -x), preserving the attack->+x rule.

    Reflecting x alone would send the attackers toward -x, breaking the
    convention, so we also swap the team labels. The result is the same picture
    viewed from the other end: the former defenders now attack toward +x. Ball
    carrier stays the same physical player.
    """
    out = frame.copy()
    out.positions[:, 0] *= -1.0
    out.velocities[:, 0] *= -1.0
    out.ball_pos[0] *= -1.0
    out.team_ids = np.where(out.team_ids == TEAM_ATTACK, TEAM_DEFEND, TEAM_ATTACK)
    return out


def jitter_positions(frame: FrozenFrame, sigma: float = 0.5,
                     seed: int | None = None) -> FrozenFrame:
    """Add Gaussian noise (std ``sigma`` m) to every position and the ball.

    Emulates homography + detection error in a real tracking pipeline. Results
    are clipped back onto the pitch.
    """
    rng = np.random.default_rng(seed)
    out = frame.copy()
    out.positions = pitch.clip_to_pitch(
        out.positions + rng.normal(0.0, sigma, size=out.positions.shape))
    out.ball_pos = pitch.clip_to_pitch(
        out.ball_pos + rng.normal(0.0, sigma, size=out.ball_pos.shape))
    return out


def perturb_velocities(frame: FrozenFrame, sigma: float = 0.5,
                       seed: int | None = None) -> FrozenFrame:
    """Add Gaussian noise (std ``sigma`` m/s) to velocities, re-capping speed."""
    rng = np.random.default_rng(seed)
    out = frame.copy()
    out.velocities = clip_speed(
        out.velocities + rng.normal(0.0, sigma, size=out.velocities.shape),
        MAX_PLAYER_SPEED)
    return out


if __name__ == "__main__":
    for name, make in ALL_SCENARIOS.items():
        ff = make()
        print(f"{name:15s}: {ff.n_players:2d} players "
              f"({ff.attack_mask.sum()} atk / {ff.defend_mask.sum()} def), "
              f"max speed {ff.speeds.max():.1f} m/s, carrier idx {ff.ball_carrier}")
