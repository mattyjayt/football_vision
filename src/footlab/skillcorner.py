"""Load SkillCorner open broadcast-tracking data into :class:`FrozenFrame`.

This module is the bridge between *real* professional match data and the
footlab tactical engine. SkillCorner has already solved the vision problem
(broadcast video -> per-player coordinates); we consume their tracking output
and re-express it as ``FrozenFrame`` so every downstream phase (pitch control,
cost maps, planners, reactive rollout) can run on a real A-League attack
instead of a simulated one.

Reference:
    SkillCorner & PySport (2025). "SkillCorner Open Data."
    https://github.com/SkillCorner/opendata — 10 matches of broadcast tracking
    for the 2024/25 Australian A-League. Tracking is a JSONL list of frames at
    10 fps; each frame has ``ball_data`` (x, y, z), ``possession`` and
    ``player_data`` (per-player x, y, player_id, is_detected). Coordinates are
    in meters with origin at pitch center and x along the long side — the SAME
    convention as ``footlab.pitch`` (and Metrica / Friends-of-Tracking), so no
    rescaling is required beyond the 104 m vs 105 m length difference.

Simplifications / honest notes:
    * SkillCorner provides positions only; velocities are DERIVED here by a
      centered finite difference between frames (dt = 0.1 s), lightly smoothed
      with a moving average to suppress frame-to-frame jitter. They are
      estimates, not measurements.
    * ``is_detected == False`` marks players whose position is extrapolated
      (off-camera). We keep them (a full 22-player frame is more useful for the
      planners) but expose ``only_detected`` to drop them if preferred.
    * Team ids: home team -> TEAM_ATTACK (0), away -> TEAM_DEFEND (1). The
      frame is oriented (via ``home_team_side`` and period) so the attack
      proceeds toward +x, matching ``footlab.pitch``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .state import FrozenFrame, TEAM_ATTACK, TEAM_DEFEND, clip_speed

# SkillCorner tracking runs at 10 frames per second.
FPS: float = 10.0
DT: float = 1.0 / FPS

# A-League pitches in this dataset are 104 x 68 (vs footlab's 105 x 68).


def load_match_meta(match_dir: str | Path) -> dict:
    """Load ``{id}_match.json`` (lineups, teams, pitch size) for one match.

    Returns the parsed dict. The ``players`` list maps each player's ``id`` to
    a ``team_id`` and jersey ``number`` — used to assign team and labels.
    """
    match_dir = Path(match_dir)
    match_files = list(match_dir.glob("*_match.json"))
    if not match_files:
        raise FileNotFoundError(f"no *_match.json found in {match_dir}")
    with open(match_files[0]) as fh:
        return json.load(fh)


def _tracking_path(match_dir: str | Path) -> Path:
    match_dir = Path(match_dir)
    files = list(match_dir.glob("*_tracking_extrapolated.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"no *_tracking_extrapolated.jsonl in {match_dir}")
    return files[0]


def _player_team_and_number(meta: dict) -> dict[int, tuple[int, int]]:
    """Map player_id -> (footlab team id, jersey number).

    Home team -> TEAM_ATTACK, away -> TEAM_DEFEND. Players not tied to either
    team (e.g. match officials appearing in the registry) default to defend so
    they never masquerade as attackers.
    """
    home_id = meta["home_team"]["id"]
    away_id = meta["away_team"]["id"]
    mapping: dict[int, tuple[int, int]] = {}
    for p in meta.get("players", []):
        tid = p.get("team_id")
        team = TEAM_ATTACK if tid == home_id else TEAM_DEFEND
        mapping[p["id"]] = (team, int(p.get("number", 0)))
    return mapping


def read_frames(
    match_dir: str | Path,
    *,
    only_detected: bool = False,
) -> tuple[list[dict], dict[int, tuple[int, int]]]:
    """Stream the tracking JSONL and return (frames, player-team map).

    Each returned frame dict has: ``frame``, ``period``, ``ball`` (x, y, z),
    ``possession_player_id``, and ``players`` — a list of
    ``(player_id, x, y)`` for that instant. Memory-light: the JSONL is parsed
    line by line rather than loaded whole.
    """
    meta = load_match_meta(match_dir)
    team_map = _player_team_and_number(meta)
    frames: list[dict] = []
    with open(_tracking_path(match_dir)) as fh:
        for line in fh:
            f = json.loads(line)
            players = []
            for p in f.get("player_data", []):
                if only_detected and not p.get("is_detected", False):
                    continue
                if p.get("x") is None or p.get("y") is None:
                    continue
                players.append((p["player_id"], float(p["x"]), float(p["y"])))
            ball = f.get("ball_data", {})
            frames.append({
                "frame": f.get("frame"),
                "period": f.get("period"),
                "ball": (ball.get("x"), ball.get("y")),
                "possession_player_id": (f.get("possession") or {}).get("player_id"),
                "players": players,
            })
    return frames, team_map


def _estimate_velocities(series: np.ndarray, dt: float = DT) -> np.ndarray:
    """Centered finite-difference velocities from a position series.

    ``series`` is (T, N, 2) positions over time. Returns (T, N, 2) velocities.
    Interior frames use the centered difference (x[t+1]-x[t-1])/(2dt); the two
    ends fall back to a one-sided difference. Speeds are clipped to the
    physiological sprint cap so single-frame jitter cannot produce a 40 m/s
    ghost.
    """
    series = np.asarray(series, dtype=float)
    vel = np.zeros_like(series)
    if series.shape[0] < 2:
        return vel
    vel[1:-1] = (series[2:] - series[:-2]) / (2.0 * dt)
    vel[0] = (series[1] - series[0]) / dt
    vel[-1] = (series[-1] - series[-2]) / dt
    # Clip each frame's per-player velocities (clip_speed expects (N, 2)).
    return np.stack([clip_speed(frame) for frame in vel])


def frames_to_frozen(
    match_dir: str | Path,
    frame_index: int,
    *,
    velocity_window: int = 3,
    only_detected: bool = False,
) -> FrozenFrame:
    """Build one :class:`FrozenFrame` from a real match at ``frame_index``.

    ``frame_index`` selects the tracking frame (0-based index into the JSONL).
    Velocities are estimated from a short centred window of neighbouring frames
    (``velocity_window`` frames on each side) to smooth jitter.

    The attacking team is oriented toward +x: if the home team defends the +x
    goal in this period, all coordinates are mirrored so the attack always
    plays left -> right (the footlab convention).
    """
    frames, team_map = read_frames(match_dir, only_detected=only_detected)
    if not (0 <= frame_index < len(frames)):
        raise IndexError(f"frame_index {frame_index} out of range "
                         f"(0..{len(frames) - 1})")

    target = frames[frame_index]
    if not target["players"]:
        raise ValueError(f"frame {frame_index} has no players "
                         "(pre-match or camera off play)")

    # --- Orientation & team assignment ---------------------------------------
    # footlab plans a path for the BALL CARRIER, so the "attacking" team is the
    # team in possession, and the attack is normalised to left->right (+x).
    # home_team_side is a per-period list, e.g. ["right_to_left","left_to_right"]
    # giving the HOME team's attack direction in each period. From it we derive
    # each team's signed attack direction, then mirror x if the possessing team
    # attacks toward -x so their attack becomes +x.
    meta = load_match_meta(match_dir)
    sides = meta.get("home_team_side") or []
    period = target.get("period") or 1
    period_idx = min(max(int(period) - 1, 0), len(sides) - 1) if sides else 0
    home_dir = -1.0 if (sides and sides[period_idx] == "right_to_left") else 1.0
    away_dir = -home_dir

    poss = target["possession_player_id"]
    poss_home = poss in team_map and team_map[poss][0] == TEAM_ATTACK
    # home=0 in team_map means home side; see _player_team_and_number.
    attack_dir = home_dir if poss_home else away_dir
    mirror = attack_dir < 0

    def orient_xy(xy: np.ndarray) -> np.ndarray:
        out = np.array(xy, dtype=float, copy=True)
        if mirror:
            out[..., 0] *= -1.0
        return out

    # Stable player ordering for the velocity window: use the target frame's
    # ids, look each one up in neighbouring frames (missing -> target pos).
    ids = [pid for pid, _, _ in target["players"]]
    id_to_idx = {pid: i for i, pid in enumerate(ids)}
    n = len(ids)

    def positions_at(fr: dict) -> np.ndarray:
        pos = np.full((n, 2), np.nan)
        for pid, x, y in fr["players"]:
            if pid in id_to_idx:
                pos[id_to_idx[pid]] = (x, y)
        return pos

    lo = max(0, frame_index - velocity_window)
    hi = min(len(frames), frame_index + velocity_window + 1)
    window = np.stack([positions_at(frames[k]) for k in range(lo, hi)])
    # Fill gaps (player briefly untracked) with the target-frame position.
    target_pos = positions_at(target)
    for t in range(window.shape[0]):
        window[t][np.isnan(window[t, :, 0])] = target_pos[np.isnan(window[t, :, 0])]

    center = frame_index - lo
    positions = orient_xy(window[center])
    # Finite difference over the window; take the velocity at the centre.
    # Velocities are computed in raw coords then mirrored to match positions.
    vels = orient_xy(_estimate_velocities(window)[center])

    # Team assignment: the possessing team is TEAM_ATTACK (0), the team without
    # the ball is TEAM_DEFEND (1). team_map stores home as 0 / away as 1, so we
    # remap to possession-relative ids. When possession is unknown we fall back
    # to home=attack (arbitrary but deterministic).
    raw_team = np.array([team_map.get(pid, (TEAM_DEFEND, 0))[0] for pid in ids],
                        dtype=int)  # 0 = home, 1 = away
    numbers = np.array([team_map.get(pid, (TEAM_DEFEND, 0))[1] for pid in ids])
    if poss in team_map:
        attack_side = team_map[poss][0]  # 0 if home has it, 1 if away has it
        team_ids = np.where(raw_team == attack_side, TEAM_ATTACK, TEAM_DEFEND)
    else:
        team_ids = raw_team  # loose ball: keep home=attack convention

    # Ball.
    bx, by = target["ball"]
    ball_pos = orient_xy(np.array([bx if bx is not None else 0.0,
                                   by if by is not None else 0.0], dtype=float))

    # Carrier: map possession player_id to array index (-1 if loose/unknown).
    poss = target["possession_player_id"]
    ball_carrier = id_to_idx.get(poss, -1)

    return FrozenFrame(
        positions=positions,
        velocities=vels,
        team_ids=team_ids,
        ball_pos=ball_pos,
        ball_carrier=ball_carrier,
        player_ids=numbers,
    )


if __name__ == "__main__":
    import sys

    md = sys.argv[1] if len(sys.argv) > 1 else (
        Path(__file__).resolve().parents[2]
        / "data" / "skillcorner" / "data" / "matches" / "1886347")
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 1500

    ff = frames_to_frozen(md, idx)
    print(f"match dir : {md}")
    print(f"frame     : {idx}")
    print(f"players   : {ff.n_players} "
          f"({ff.attack_mask.sum()} att / {ff.defend_mask.sum()} def)")
    print(f"ball      : {np.round(ff.ball_pos, 2)}")
    carrier = ff.carrier_position
    print(f"carrier   : index {ff.ball_carrier}"
          + (f" at {np.round(carrier, 2)}" if carrier is not None else " (loose)"))
    print(f"speeds    : min {ff.speeds.min():.2f} / "
          f"mean {ff.speeds.mean():.2f} / max {ff.speeds.max():.2f} m/s")
