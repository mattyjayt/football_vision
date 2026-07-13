"""Phase 2 — Spearman's potential pitch control field (PPCF).

For every point on the pitch we compute the probability that the attacking team
would control the ball *if it arrived at that point*. The result is a smooth
surface in [0, 1] (attacking control), the probabilistic successor to Phase 1's
hard dominant region and the raw material for the Phase 2b cost map.

Two entry points:
    * :func:`pitch_control_at_target` — readable single-point reference that
      loops over players. Returns ``(P_att, P_def)``.
    * :func:`pitch_control_surface`   — numpy-vectorized integration over a whole
      grid at once. Returns the attacking-control surface ``(ny, nx)``.

A test asserts the two agree; the vectorized version exists only for speed.

Reference:
    Spearman, W. (2018). "Beyond Expected Goals." MIT Sloan Sports Analytics
    Conference. Implements the potential pitch control field, Eq. (3)-(5):
    players race to the target under a time-to-intercept model, the probability
    a player has arrived by time T is a logistic ramp, and control accumulates
    via coupled ODEs

        dPPCF_j/dT = (1 - sum_k PPCF_k(T)) * f_j(T) * lambda_j ,

    where (1 - sum_k PPCF_k) is the probability the ball is still loose, f_j(T)
    is player j's arrival probability, and lambda_j is the control rate.

    Time-to-intercept groundwork: Spearman et al. (2017), "Physics-Based
    Modeling of Pass Probabilities in Soccer," MIT SSAC. Reference
    implementation: Laurie Shaw, ``LaurieOnTracking`` (Friends of Tracking).

Parameters follow Spearman (2018) / LaurieOnTracking defaults (see
:class:`PitchControlParams`).

Simplifications vs. the reference implementation:
    * No short-circuit: we integrate the ODE for every cell rather than
      analytically resolving cells that one team clearly wins.
    * ``int_dt = 0.04 s`` fixed Euler steps; convergence tolerance 0.01.
    * A single global ``max_speed`` (5 m/s) rather than per-player estimates.
    * Reaction phase is pure coasting at current velocity (see
      :func:`footlab.geometry.time_to_arrive`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from . import geometry
from .state import FrozenFrame


@dataclass(frozen=True)
class PitchControlParams:
    """Model parameters, Spearman (2018) / LaurieOnTracking defaults.

    reaction_time:     0.7 s   — coast at current velocity before redirecting.
    max_speed:         5.0 m/s — effective running speed toward a target.
    tti_sigma:         0.45 s  — uncertainty in time-to-intercept (logistic width).
    lambda_att:        4.3 1/s — attacking control-accrual rate.
    lambda_def:        4.3 1/s — defending control-accrual rate (kappa_def = 1;
                                 the paper's 1.72 gives defenders an edge).
    average_ball_speed:15.0 m/s— assumed ball speed to reach the target.
    int_dt:            0.04 s  — Euler integration step.
    max_int_time:      8.0 s   — control window after the ball/first player arrives.
    converge_tol:      0.01    — stop once <1% of control remains unassigned.
    """

    reaction_time: float = 0.7
    max_speed: float = 5.0
    tti_sigma: float = 0.45
    lambda_att: float = 4.3
    lambda_def: float = 4.3
    average_ball_speed: float = 15.0
    int_dt: float = 0.04
    max_int_time: float = 8.0
    converge_tol: float = 0.01

    @property
    def sigmoid_scale(self) -> float:
        """The logistic steepness pi / (sqrt(3) * tti_sigma) from Spearman (2018)."""
        return np.pi / (np.sqrt(3.0) * self.tti_sigma)


DEFAULT_PARAMS = PitchControlParams()


def _tti(frame: FrozenFrame, targets: np.ndarray, params: PitchControlParams
         ) -> np.ndarray:
    """Time-to-intercept of every player to every target, shape (N, M)."""
    return geometry.time_to_arrive(
        frame.positions, frame.velocities, targets,
        reaction_time=params.reaction_time, max_speed=params.max_speed)


def pitch_control_at_target(frame: FrozenFrame, target: np.ndarray, *,
                            params: PitchControlParams = DEFAULT_PARAMS
                            ) -> tuple[float, float]:
    """Probability each team controls the ball if it arrived at ``target``.

    Readable reference implementation (loops over players). Integrates the
    Spearman (2018) coupled ODEs forward in time from the moment the ball could
    arrive at the target, and returns ``(P_att, P_def)`` which sum to ~1 once
    converged. See module docstring for the model and citations.
    """
    target = np.asarray(target, dtype=float).reshape(1, 2)
    tti = _tti(frame, target, params).ravel()               # (N,)
    att = frame.attack_mask
    dfn = frame.defend_mask
    lam = np.where(att, params.lambda_att, params.lambda_def)  # (N,)
    scale = params.sigmoid_scale

    ball_time = float(np.linalg.norm(frame.ball_pos - target.ravel())
                      / params.average_ball_speed)
    t_max = max(ball_time, float(tti.min())) + params.max_int_time

    ppcf = np.zeros(frame.n_players)   # per-player accumulated control
    p_att = p_def = 0.0
    t = 0.0
    dt = params.int_dt
    while (1.0 - (p_att + p_def)) > params.converge_tol and t < t_max:
        t += dt
        if t < ball_time:
            continue  # ball has not arrived yet; nothing to control
        rem = max(0.0, 1.0 - p_att - p_def)
        f = expit(scale * (t - tti))                        # arrival prob (N,)
        ppcf += rem * f * lam * dt
        p_att = float(ppcf[att].sum())
        p_def = float(ppcf[dfn].sum())
    return p_att, p_def


def pitch_control_surface(frame: FrozenFrame, centers: np.ndarray, *,
                          params: PitchControlParams = DEFAULT_PARAMS
                          ) -> np.ndarray:
    """Attacking-team control probability over a whole grid, shape (ny, nx).

    Vectorized equivalent of :func:`pitch_control_at_target` run for every cell
    of ``centers`` (from :func:`footlab.pitch.make_grid`) simultaneously. The
    defending surface is ``1 - surface`` wherever the integration has converged.
    """
    ny, nx = centers.shape[:2]
    targets = centers.reshape(-1, 2)                        # (M, 2)
    tti = _tti(frame, targets, params)                      # (N, M)

    att_idx = np.where(frame.attack_mask)[0]
    def_idx = np.where(frame.defend_mask)[0]
    lam = np.where(frame.attack_mask, params.lambda_att,
                   params.lambda_def)[:, None]              # (N, 1)
    scale = params.sigmoid_scale

    ball_time = np.linalg.norm(targets - frame.ball_pos, axis=1) \
        / params.average_ball_speed                         # (M,)
    min_tti = tti.min(axis=0)                               # (M,)
    global_t_max = float(
        (np.maximum(ball_time, min_tti) + params.max_int_time).max())

    ppcf = np.zeros((frame.n_players, targets.shape[0]))    # (N, M)
    p_att = np.zeros(targets.shape[0])                      # (M,)
    p_def = np.zeros(targets.shape[0])
    t = 0.0
    dt = params.int_dt
    while t < global_t_max:
        t += dt
        rem = np.clip(1.0 - p_att - p_def, 0.0, 1.0)        # (M,)
        if rem.max() < params.converge_tol:
            break
        gate = (t >= ball_time)                             # (M,) ball arrived?
        f = expit(scale * (t - tti))                        # (N, M)
        dppcf = rem[None, :] * f * lam * dt                 # (N, M)
        dppcf *= gate[None, :]
        ppcf += dppcf
        p_att = ppcf[att_idx].sum(axis=0)
        p_def = ppcf[def_idx].sum(axis=0)
    return p_att.reshape(ny, nx)


if __name__ == "__main__":
    from . import pitch
    from .simulate import scenario_counter_attack

    frame = scenario_counter_attack()
    _, _, centers = pitch.make_grid(cell_size=1.0)
    surf = pitch_control_surface(frame, centers)
    print(f"surface shape {surf.shape}, range [{surf.min():.2f}, {surf.max():.2f}]")
    print(f"mean attacking control: {surf.mean():.1%}")
    p_att, p_def = pitch_control_at_target(frame, frame.carrier_position)
    print(f"control at carrier's feet: att {p_att:.2f} / def {p_def:.2f}")
