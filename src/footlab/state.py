"""The :class:`FrozenFrame` dataclass — the single object every algorithm reads.

A "frozen frame" is one instant of play reduced to pure numbers: where every
player is, how fast they are moving, which team they belong to, where the ball
is, and who is carrying it. Every later phase (Voronoi, pitch control, cost
maps, planners) consumes a ``FrozenFrame`` and nothing else, which keeps the
whole pipeline decoupled from how the data was produced (simulated now, real
tracking later).

Team-id convention (see ``pitch.py`` for the coordinate convention):

    * ``TEAM_ATTACK = 0`` — attacks the goal at (+52.5, 0), i.e. moves toward +x.
    * ``TEAM_DEFEND = 1`` — defends that goal.

We store data as numpy arrays (not per-player objects) so downstream code can
vectorize cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Team ids.
TEAM_ATTACK: int = 0
TEAM_DEFEND: int = 1

# Kinematic modelling constant: sprinting footballers top out around 9 m/s.
# Used to sanity-check/cap generated and perturbed velocities. Later motion
# models (Fujimura & Sugihara 2005) refine this with acceleration limits.
MAX_PLAYER_SPEED: float = 9.0


def clip_speed(velocities: np.ndarray, max_speed: float = MAX_PLAYER_SPEED) -> np.ndarray:
    """Rescale any velocity vector whose speed exceeds ``max_speed`` down to it.

    Direction is preserved; only over-speed rows are touched. Accepts a single
    (2,) vector or an (N, 2) array.
    """
    v = np.asarray(velocities, dtype=float)
    single = v.ndim == 1
    v = np.atleast_2d(v).copy()
    speeds = np.linalg.norm(v, axis=1)
    over = speeds > max_speed
    # Avoid divide-by-zero: only touch the over-speed rows (which have speed>0).
    v[over] *= (max_speed / speeds[over])[:, None]
    return v[0] if single else v


@dataclass
class FrozenFrame:
    """One instant of play.

    Attributes:
        positions:   (N, 2) player positions in meters, origin at pitch center.
        velocities:  (N, 2) player velocities in m/s.
        team_ids:    (N,) ints; TEAM_ATTACK (0) or TEAM_DEFEND (1).
        ball_pos:    (2,) ball position in meters.
        ball_carrier: index into the player arrays of the carrier, or -1 if the
                      ball is loose. By convention the carrier is an attacker.

    Arrays are coerced to float/int and shape-validated on construction, so a
    malformed frame fails loudly at creation rather than deep inside a planner.
    """

    positions: np.ndarray
    velocities: np.ndarray
    team_ids: np.ndarray
    ball_pos: np.ndarray
    ball_carrier: int = -1
    # Optional stable player labels (e.g. jersey numbers); handy for debugging.
    player_ids: np.ndarray | None = field(default=None)

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float).reshape(-1, 2)
        self.velocities = np.asarray(self.velocities, dtype=float).reshape(-1, 2)
        self.team_ids = np.asarray(self.team_ids, dtype=int).reshape(-1)
        self.ball_pos = np.asarray(self.ball_pos, dtype=float).reshape(2)
        self.ball_carrier = int(self.ball_carrier)

        n = self.positions.shape[0]
        if self.velocities.shape[0] != n:
            raise ValueError(
                f"velocities has {self.velocities.shape[0]} rows, expected {n}")
        if self.team_ids.shape[0] != n:
            raise ValueError(
                f"team_ids has {self.team_ids.shape[0]} entries, expected {n}")
        if not set(np.unique(self.team_ids)).issubset({TEAM_ATTACK, TEAM_DEFEND}):
            raise ValueError("team_ids must contain only 0 (attack) or 1 (defend)")
        if not (-1 <= self.ball_carrier < n):
            raise ValueError(
                f"ball_carrier {self.ball_carrier} out of range for {n} players")
        if self.player_ids is not None:
            self.player_ids = np.asarray(self.player_ids).reshape(-1)
            if self.player_ids.shape[0] != n:
                raise ValueError("player_ids length must match number of players")

    # --- Convenience views ---------------------------------------------------
    @property
    def n_players(self) -> int:
        return self.positions.shape[0]

    @property
    def attack_mask(self) -> np.ndarray:
        return self.team_ids == TEAM_ATTACK

    @property
    def defend_mask(self) -> np.ndarray:
        return self.team_ids == TEAM_DEFEND

    @property
    def attacker_positions(self) -> np.ndarray:
        return self.positions[self.attack_mask]

    @property
    def defender_positions(self) -> np.ndarray:
        return self.positions[self.defend_mask]

    @property
    def speeds(self) -> np.ndarray:
        """(N,) per-player speed in m/s."""
        return np.linalg.norm(self.velocities, axis=1)

    @property
    def carrier_position(self) -> np.ndarray | None:
        """Position of the ball carrier, or None if the ball is loose."""
        if self.ball_carrier < 0:
            return None
        return self.positions[self.ball_carrier]

    def copy(self) -> "FrozenFrame":
        """Deep-ish copy (arrays duplicated) for non-destructive augmentation."""
        return FrozenFrame(
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            team_ids=self.team_ids.copy(),
            ball_pos=self.ball_pos.copy(),
            ball_carrier=self.ball_carrier,
            player_ids=None if self.player_ids is None else self.player_ids.copy(),
        )


if __name__ == "__main__":
    # Tiny 2v1 sanity frame.
    ff = FrozenFrame(
        positions=[[0, 0], [10, 5], [20, -3]],
        velocities=[[3, 0], [2, 1], [-4, 0]],
        team_ids=[TEAM_ATTACK, TEAM_ATTACK, TEAM_DEFEND],
        ball_pos=[0, 0],
        ball_carrier=0,
    )
    print(f"{ff.n_players} players, "
          f"{ff.attack_mask.sum()} attackers, {ff.defend_mask.sum()} defenders")
    print("speeds (m/s):", np.round(ff.speeds, 2))
    print("carrier at:", ff.carrier_position)
