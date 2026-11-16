"""Touch detection and control-based possession evidence (Approach A).

Replaces "who is nearest the ball right now" with "who last controlled the ball", derived purely
from ball kinematics and player proximity. Every threshold is expressed in physical units (metres,
m/s, seconds, degrees) from football and measurement-noise physics, never fitted to labels.
"""

from dataclasses import dataclass

import numpy as np

import signals

FPS = 25
HOME = 0
AWAY = 1
NONE = -1


@dataclass
class TouchConfig:
    # --- A1: ball dynamics (smoothed velocity, acceleration, turn angle on observed frames) ---
    smooth_half_window: int = 1    # frames. [0.25, 0.5, 0.25] smoothing damps ~0.3 m ATD jitter
    dyn_window_frames: int = 5     # frames (0.20 s). Before/after velocity window; resolves >=0.4 s passes
    min_turn_speed: float = 2.2    # m/s. Min speed before AND after for turn angle to beat jitter

    # --- A2: touch candidate detection and attribution ---
    contact_radius: float = 2.2    # m. Foot reach (~0.7 m) + ball/player ATD tracking error (~1.5 m)
    min_touch_speed: float = 4.2   # m/s. Max(v_before, v_after) must reach at least a soft pass
    min_dv: float = 5.2            # m/s. Vector velocity jump |v_after - v_before| over 0.20 s (~26 m/s^2)
    min_speed_jump: float = 4.0    # m/s. Scalar speed jump ||v_after| - |v_before|| for kick/trap
    min_turn_deg: float = 38.0     # deg. Redirection angle threshold when both sides are moving
    dv_ref: float = 12.0           # m/s. Reference velocity jump giving full dynamic confidence
    speed_jump_ref: float = 10.0   # m/s. Reference scalar speed jump giving full dynamic confidence
    turn_ref_deg: float = 90.0     # deg. Right-angle redirect reference for dynamic confidence
    nms_window_s: float = 0.20     # s (5 frames). Non-maximum suppression radius around each contact
    contest_margin: float = 0.50   # m. Min |d_away - d_home| gap for an uncontested touch

    # --- A4/A5: frame-level control log-odds ---
    ctrl_gain: float = 1.3         # log-odds scale per unit of control confidence
    ctrl_cap: float = 1.2          # cap on |ctrl_logodds|, matching model.Config.gap_cap
    flight_decay_s: float = 12.0   # s. Exponential confidence decay while ball is loose/in flight
    oob_reset_s: float = 1.0       # s (25 frames). Sustained out-of-bounds resets active control


def _detect_filled_fallback(ball_clean, observed):
    """Unused unless caller already passed pre-cleaned ball without observed/filled masks."""
    return np.zeros(len(ball_clean), dtype=bool)


def _lagged_positions(pos_work, n, lag, direction):
    """Return shifted positions `pos_work[t + direction * k]` searching fallback offsets near `lag`.

    `direction` is -1 for backward (before t) or +1 for forward (after t). Returns (xy[n,2], dt_frames[n]).
    """
    xy = np.full((n, 2), np.nan, dtype=np.float32)
    dt = np.zeros(n, dtype=np.float32)
    offsets = (lag, lag + 1, max(2, lag - 1), lag + 2, max(2, lag - 2))
    # Deduplicate while preserving search priority
    seen = set()
    ordered = []
    for k in offsets:
        if k not in seen and k >= 2:
            seen.add(k)
            ordered.append(k)

    for k in ordered:
        if k >= n:
            continue
        if direction < 0:
            src = pos_work[:n - k]
            dst_slice = slice(k, n)
        else:
            src = pos_work[k:]
            dst_slice = slice(0, n - k)
        need = np.isnan(xy[dst_slice, 0]) & np.isfinite(src[:, 0]) & np.isfinite(src[:, 1])
        if np.any(need):
            sub_xy = xy[dst_slice]
            sub_dt = dt[dst_slice]
            sub_xy[need] = src[need]
            sub_dt[need] = float(k)
    return xy, dt


def ball_dynamics(ball, observed=None, filled=None, cfg=None):
    """Compute NaN-safe ball kinematics on observed frames, never creating evidence in `filled`.

    Returns a dict with:
        vel_before, vel_after, vel: (T, 2) float32 m/s
        speed_before, speed_after, dv, dspeed, accel, turn_angle: (T,) float32
        observed, filled, valid_obs, ball_ok: (T,) bool
    """
    cfg = cfg or TouchConfig()
    ball_arr = np.asarray(ball, dtype=np.float32)
    n = len(ball_arr)

    if observed is None or filled is None:
        cleaned, obs, fil = signals.clean_ball(ball_arr)
        ball_clean = cleaned
        observed_m = obs if observed is None else np.asarray(observed, dtype=bool)
        filled_m = fil if filled is None else np.asarray(filled, dtype=bool)
    else:
        ball_clean = ball_arr
        observed_m = np.asarray(observed, dtype=bool)
        filled_m = np.asarray(filled, dtype=bool)

    finite = np.isfinite(ball_clean[:, 0]) & np.isfinite(ball_clean[:, 1])
    valid_obs = observed_m & ~filled_m & finite
    ball_ok = (observed_m | filled_m) & finite

    # Smooth only over observed frames so interpolated gaps never bias observed endpoints.
    pos_obs = np.full((n, 2), np.nan, dtype=np.float32)
    if n > 0:
        w_sum = np.zeros(n, dtype=np.float32)
        xy_sum = np.zeros((n, 2), dtype=np.float32)
        xy_sum[valid_obs] += 0.5 * ball_clean[valid_obs]
        w_sum[valid_obs] += 0.5
        if n > 1 and cfg.smooth_half_window >= 1:
            m_l = valid_obs[:-1]
            xy_sum[1:][m_l] += 0.25 * ball_clean[:-1][m_l]
            w_sum[1:][m_l] += 0.25
            m_r = valid_obs[1:]
            xy_sum[:-1][m_r] += 0.25 * ball_clean[1:][m_r]
            w_sum[:-1][m_r] += 0.25
        keep = valid_obs & (w_sum > 0)
        pos_obs[keep] = xy_sum[keep] / w_sum[keep, None]

    # Use smoothed observed positions where observed, and clean linear fills across short gaps
    # so touches can still be evaluated right at the observed edges of a filled stretch.
    pos_work = np.where(valid_obs[:, None], pos_obs, np.where(ball_ok[:, None], ball_clean, np.nan))

    w = max(2, int(cfg.dyn_window_frames))
    xy_before, dt_before = _lagged_positions(pos_work, n, w, -1)
    xy_after, dt_after = _lagged_positions(pos_work, n, w, +1)

    vel_before = np.full((n, 2), np.nan, dtype=np.float32)
    vel_after = np.full((n, 2), np.nan, dtype=np.float32)
    vel = np.full((n, 2), np.nan, dtype=np.float32)

    ok_b = valid_obs & (dt_before > 0) & np.isfinite(xy_before[:, 0])
    ok_a = valid_obs & (dt_after > 0) & np.isfinite(xy_after[:, 0])
    vel_before[ok_b] = (pos_obs[ok_b] - xy_before[ok_b]) * (FPS / dt_before[ok_b, None])
    vel_after[ok_a] = (xy_after[ok_a] - pos_obs[ok_a]) * (FPS / dt_after[ok_a, None])

    ok_both = ok_b & ok_a
    vel[ok_both] = (xy_after[ok_both] - xy_before[ok_both]) * (
        FPS / (dt_before[ok_both, None] + dt_after[ok_both, None])
    )
    only_b = ok_b & ~ok_a
    only_a = ok_a & ~ok_b
    vel[only_b] = vel_before[only_b]
    vel[only_a] = vel_after[only_a]

    speed_before = np.zeros(n, dtype=np.float32)
    speed_after = np.zeros(n, dtype=np.float32)
    speed_before[ok_b] = np.hypot(vel_before[ok_b, 0], vel_before[ok_b, 1])
    speed_after[ok_a] = np.hypot(vel_after[ok_a, 0], vel_after[ok_a, 1])

    dv = np.zeros(n, dtype=np.float32)
    dspeed = np.zeros(n, dtype=np.float32)
    accel = np.zeros(n, dtype=np.float32)
    turn_angle = np.zeros(n, dtype=np.float32)

    if np.any(ok_both):
        dvec = vel_after[ok_both] - vel_before[ok_both]
        dv_val = np.hypot(dvec[:, 0], dvec[:, 1])
        dv[ok_both] = dv_val
        dspeed[ok_both] = np.abs(speed_after[ok_both] - speed_before[ok_both])
        dt_mid = 0.5 * (dt_before[ok_both] + dt_after[ok_both]) / float(FPS)
        accel[ok_both] = dv_val / np.maximum(dt_mid, 1.0 / FPS)

        sb = speed_before[ok_both]
        sa = speed_after[ok_both]
        moving = (sb >= cfg.min_turn_speed) & (sa >= cfg.min_turn_speed)
        if np.any(moving):
            vb_m = vel_before[ok_both][moving]
            va_m = vel_after[ok_both][moving]
            denom = sb[moving] * sa[moving]
            cos_t = np.clip(np.sum(vb_m * va_m, axis=1) / np.maximum(denom, 1e-6), -1.0, 1.0)
            ang = np.zeros(int(ok_both.sum()), dtype=np.float32)
            ang[moving] = np.arccos(cos_t)
            turn_angle[ok_both] = ang

    # Guarantee zero dynamics evidence on non-observed or filled frames
    vel_before[~valid_obs] = np.nan
    vel_after[~valid_obs] = np.nan
    vel[~valid_obs] = np.nan
    dv[~valid_obs] = 0.0
    dspeed[~valid_obs] = 0.0
    accel[~valid_obs] = 0.0
    turn_angle[~valid_obs] = 0.0

    return {
        "ball_clean": ball_clean,
        "pos_obs": pos_obs,
        "vel_before": vel_before,
        "vel_after": vel_after,
        "vel": vel,
        "speed_before": speed_before,
        "speed_after": speed_after,
        "dv": dv,
        "dspeed": dspeed,
        "accel": accel,
        "turn_angle": turn_angle,
        "observed": observed_m,
        "filled": filled_m,
        "valid_obs": valid_obs,
        "ball_ok": ball_ok,
    }


def detect_touches(sig_so_far, dyn, cfg=None):
    """Detect per-frame touch candidates with NMS and team attribution.

    Returns (touch[T] float32 in [0,1], touch_team[T] int8 in {-1, 0, 1}, touch_dist[T] float32).
    """
    cfg = cfg or TouchConfig()
    valid_obs = dyn["valid_obs"]
    n = len(valid_obs)

    touch = np.zeros(n, dtype=np.float32)
    touch_team = np.full(n, NONE, dtype=np.int8)
    touch_dist = np.full(n, np.nan, dtype=np.float32)
    if n == 0:
        return touch, touch_team, touch_dist

    dh = np.asarray(sig_so_far["dist_home"], dtype=np.float32)
    da = np.asarray(sig_so_far["dist_away"], dtype=np.float32)
    nearest = np.minimum(dh, da)
    margin = np.abs(np.nan_to_num(da, nan=np.inf, posinf=1e6) - np.nan_to_num(dh, nan=np.inf, posinf=1e6))

    sb = dyn["speed_before"]
    sa = dyn["speed_after"]
    dv = dyn["dv"]
    dspeed = dyn["dspeed"]
    turn = dyn["turn_angle"]

    max_s = np.maximum(sb, sa)
    both_moving = (sb >= cfg.min_turn_speed) & (sa >= cfg.min_turn_speed)
    turn_thresh = np.deg2rad(cfg.min_turn_deg)

    abrupt = (dv >= cfg.min_dv) & ((dspeed >= cfg.min_speed_jump) | (both_moving & (turn >= turn_thresh)))
    in_range = valid_obs & np.isfinite(nearest) & (nearest <= cfg.contact_radius)
    if "oob" in sig_so_far:
        in_range = in_range & ~np.asarray(sig_so_far["oob"], dtype=bool)
    cand = in_range & (max_s >= cfg.min_touch_speed) & abrupt

    if not np.any(cand):
        return touch, touch_team, touch_dist

    turn_norm = np.where(both_moving, turn / np.deg2rad(cfg.turn_ref_deg), 0.0)
    dyn_strength = np.clip(
        np.maximum.reduce([dv / cfg.dv_ref, dspeed / cfg.speed_jump_ref, turn_norm]),
        0.0,
        1.0,
    )
    prox_strength = np.clip(1.0 - 0.75 * (nearest / cfg.contact_radius) ** 2, 0.25, 1.0)
    uncontested = margin >= cfg.contest_margin
    contest_scale = np.where(uncontested, 1.0, 0.55)
    raw = np.where(cand, np.clip(np.sqrt(dyn_strength * prox_strength) * contest_scale, 0.15, 1.0), 0.0)

    # Greedy NMS over nms_window_s radius
    nms_r = max(1, int(round(cfg.nms_window_s * FPS)))
    cand_idx = np.flatnonzero(raw > 0)
    order = cand_idx[np.argsort(-raw[cand_idx], kind="mergesort")]
    suppressed = np.zeros(n, dtype=bool)
    for idx in order:
        if suppressed[idx]:
            continue
        score = float(raw[idx])
        touch[idx] = score
        touch_team[idx] = HOME if dh[idx] <= da[idx] else AWAY
        touch_dist[idx] = nearest[idx]
        lo = max(0, idx - nms_r)
        hi = min(n, idx + nms_r + 1)
        suppressed[lo:hi] = True

    return touch, touch_team, touch_dist


def _compute_mvp_control(n, touch, touch_team, sig_so_far, dyn, cfg):
    """MVP A4: last toucher holds possession while ball is live/in-flight; reset on oob."""
    ctrl_logodds = np.zeros(n, dtype=np.float32)
    ctrl_conf = np.zeros(n, dtype=np.float32)
    if n == 0:
        return ctrl_logodds, ctrl_conf

    ball_ok = dyn["ball_ok"]
    oob = np.asarray(sig_so_far["oob"], dtype=bool) if "oob" in sig_so_far else np.zeros(n, dtype=bool)
    decay = float(np.exp(-1.0 / (max(cfg.flight_decay_s, 0.1) * FPS)))
    oob_limit = max(1, int(round(cfg.oob_reset_s * FPS)))

    cur_team = NONE
    cur_conf = 0.0
    oob_streak = 0

    for t in range(n):
        if oob[t] and ball_ok[t]:
            oob_streak += 1
            if oob_streak >= oob_limit:
                cur_team = NONE
                cur_conf = 0.0
            continue
        oob_streak = 0

        if touch_team[t] != NONE:
            cur_team = int(touch_team[t])
            cur_conf = max(cur_conf * decay, float(touch[t]))
        else:
            cur_conf *= decay
            if cur_conf < 1e-3:
                cur_team = NONE
                cur_conf = 0.0

        if cur_team != NONE and ball_ok[t]:
            ctrl_conf[t] = cur_conf
            sign = 1.0 if cur_team == HOME else -1.0
            ctrl_logodds[t] = sign * min(cfg.ctrl_cap, cfg.ctrl_gain * cur_conf)

    return np.nan_to_num(ctrl_logodds, nan=0.0).astype(np.float32), np.nan_to_num(ctrl_conf, nan=0.0).astype(np.float32)


def build_touch_signals(ball, players, team, sig_so_far, observed=None, filled=None, cfg=None):
    """Compute touch & control signals and add them to `sig_so_far` (additive, in-place)."""
    cfg = cfg or TouchConfig()
    dyn = ball_dynamics(ball, observed=observed, filled=filled, cfg=cfg)
    touch, touch_team, _ = detect_touches(sig_so_far, dyn, cfg=cfg)
    ctrl_logodds, ctrl_conf = _compute_mvp_control(len(touch), touch, touch_team, sig_so_far, dyn, cfg)

    sig_so_far["touch"] = touch.astype(np.float32)
    sig_so_far["touch_team"] = touch_team.astype(np.int8)
    sig_so_far["ctrl_logodds"] = ctrl_logodds.astype(np.float32)
    sig_so_far["ctrl_conf"] = ctrl_conf.astype(np.float32)
    return sig_so_far
