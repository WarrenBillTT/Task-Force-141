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
    contest_margin: float = 0.50   # m. Min |d_away - d_home| gap for an uncontested touch or carry

    # --- A3: control episodes (carry/dribble/hold and passing sequences) ---
    carry_radius: float = 1.8      # m. Max ball-to-holder distance during controlled carry/dribble/hold
    carry_max_dv: float = 4.5      # m/s. Max |v_ball - v_player| for ball moving with holder
    carry_max_ball_speed: float = 8.5  # m/s. Upper bound on ball speed during a sprint dribble
    min_carry_s: float = 0.52      # s (13 frames). Min continuous carry/hold to establish control
    carry_bridge_frames: int = 3   # frames (0.12 s). Bridge tiny tracker dropouts inside a carry
    swap_link_radius: float = 2.2  # m. Spatial continuity radius to link same-team track ID swaps
    max_pass_interval_s: float = 2.8  # s (70 frames). Max gap between consecutive same-team touches
    duel_pair_s: float = 0.28      # s (7 frames). Opposing touches closer than this indicate a 50-50 duel

    # --- A4/A5: frame-level control log-odds ---
    ctrl_gain: float = 1.3         # log-odds scale per unit of control confidence
    ctrl_cap: float = 1.2          # cap on |ctrl_logodds|, matching model.Config.gap_cap
    flight_decay_s: float = 12.0   # s. Exponential confidence decay while ball is loose/in flight
    oob_reset_s: float = 1.0       # s (25 frames). Sustained out-of-bounds resets active control


def _lagged_positions(pos_work, n, lag, direction):
    """Return shifted positions `pos_work[t + direction * k]` searching fallback offsets near `lag`.

    `direction` is -1 for backward (before t) or +1 for forward (after t). Returns (xy[n,2], dt_frames[n]).
    """
    xy = np.full((n, 2), np.nan, dtype=np.float32)
    dt = np.zeros(n, dtype=np.float32)
    offsets = (lag, lag + 1, max(2, lag - 1), lag + 2, max(2, lag - 2))
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

    Returns (touch[T] float32 in [0,1], touch_team[T] int8 in {-1, 0, 1}, uncontested[T] bool).
    """
    cfg = cfg or TouchConfig()
    valid_obs = dyn["valid_obs"]
    n = len(valid_obs)

    touch = np.zeros(n, dtype=np.float32)
    touch_team = np.full(n, NONE, dtype=np.int8)
    uncontested_touch = np.zeros(n, dtype=bool)
    if n == 0:
        return touch, touch_team, uncontested_touch

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
        return touch, touch_team, uncontested_touch

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
        touch[idx] = float(raw[idx])
        touch_team[idx] = HOME if dh[idx] <= da[idx] else AWAY
        uncontested_touch[idx] = bool(uncontested[idx])
        lo = max(0, idx - nms_r)
        hi = min(n, idx + nms_r + 1)
        suppressed[lo:hi] = True

    return touch, touch_team, uncontested_touch


def _nearest_track_velocity(players, nearest_slot, max_player_speed=12.5):
    """Vectorised velocity (T, 2) m/s of `nearest_slot[t]` at each frame t.

    Rejects candidate windows whose implied player speed exceeds `max_player_speed` (12.5 m/s)
    so track-ID swaps or slot teleports never corrupt carrier velocity, falling back to one-sided
    windows on either side of a swap.
    """
    n = len(nearest_slot)
    v_player = np.full((n, 2), np.nan, dtype=np.float32)
    if n == 0 or players.shape[1] == 0:
        return v_player

    t_idx = np.arange(n)
    s_idx = nearest_slot
    p_cur = players[t_idx, s_idx]

    def _try_window(lo_off, hi_off):
        if n <= lo_off + hi_off:
            return
        lo = np.clip(t_idx - lo_off, 0, n - 1)
        hi = np.clip(t_idx + hi_off, 0, n - 1)
        p_lo = players[lo, s_idx]
        p_hi = players[hi, s_idx]
        dt = float(lo_off + hi_off) / float(FPS)
        in_bounds = (t_idx >= lo_off) & (t_idx + hi_off < n)
        finite = np.isfinite(p_lo[:, 0]) & np.isfinite(p_hi[:, 0]) & np.isfinite(p_cur[:, 0])
        vel_cand = (p_hi - p_lo) / dt
        spd = np.hypot(vel_cand[:, 0], vel_cand[:, 1])
        # Also verify both halves against p_cur when centered so a symmetric jump doesn't slip by
        ok = np.isnan(v_player[:, 0]) & in_bounds & finite & (spd <= max_player_speed)
        if lo_off > 0 and hi_off > 0:
            s_l = np.hypot(p_cur[:, 0] - p_lo[:, 0], p_cur[:, 1] - p_lo[:, 1]) * (FPS / float(lo_off))
            s_r = np.hypot(p_hi[:, 0] - p_cur[:, 0], p_hi[:, 1] - p_cur[:, 1]) * (FPS / float(hi_off))
            ok = ok & (s_l <= max_player_speed) & (s_r <= max_player_speed)
        v_player[ok] = vel_cand[ok]

    _try_window(3, 3)   # centered +/- 3 frames (0.24 s)
    _try_window(0, 4)   # forward 0.16 s (right after track spawn / ID swap)
    _try_window(4, 0)   # backward 0.16 s (right before track dropout / ID swap)
    _try_window(0, 2)   # tight forward 0.08 s for rapid swaps
    _try_window(2, 0)   # tight backward 0.08 s for rapid swaps
    return v_player


def _bridge_same_team_gaps(mask, max_gap, blocked):
    """Fill gaps <= `max_gap` inside a single-team boolean mask if `blocked` is False across the gap."""
    out = mask.copy()
    idx = np.flatnonzero(mask)
    if len(idx) < 2:
        return out
    gaps = np.diff(idx) - 1
    for k in np.flatnonzero((gaps > 0) & (gaps <= max_gap)):
        a, b = int(idx[k]), int(idx[k + 1])
        if not np.any(blocked[a + 1:b]):
            out[a + 1:b] = True
    return out


def _mask_runs(mask):
    """Return list of (start, end) half-open intervals where boolean `mask` is True."""
    if len(mask) == 0 or not np.any(mask):
        return []
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def detect_control_episodes(players, team, sig_so_far, touch, touch_team, uncontested_touch, dyn, cfg=None):
    """Detect (a) carry/dribble/hold episodes and (b) passing-sequence episodes.

    Returns a list of dicts sorted by `(confirm, start)`:
        {"start": int, "end": int, "team": int, "confidence": float, "kind": str, "confirm": int}
    """
    cfg = cfg or TouchConfig()
    valid_obs = dyn["valid_obs"]
    n = len(valid_obs)
    if n == 0:
        return []

    players_arr = np.asarray(players, dtype=np.float32)
    team_arr = np.asarray(team, dtype=np.int8)
    ball_clean = dyn["ball_clean"]
    oob = np.asarray(sig_so_far["oob"], dtype=bool) if "oob" in sig_so_far else np.zeros(n, dtype=bool)
    dh = np.asarray(sig_so_far["dist_home"], dtype=np.float32)
    da = np.asarray(sig_so_far["dist_away"], dtype=np.float32)
    margin = np.abs(np.nan_to_num(da, nan=np.inf, posinf=1e6) - np.nan_to_num(dh, nan=np.inf, posinf=1e6))

    # Mark uncontested opposing touches (excluding tight 50-50 aerial duels)
    duel_r = max(1, int(round(cfg.duel_pair_s * FPS)))
    clean_touch = touch_team.copy()
    clean_touch[~uncontested_touch] = NONE
    touch_idx = np.flatnonzero(clean_touch != NONE)
    if len(touch_idx) > 1:
        is_duel = np.zeros(len(touch_idx), dtype=bool)
        for j in range(len(touch_idx) - 1):
            if (touch_idx[j + 1] - touch_idx[j] <= duel_r) and (
                clean_touch[touch_idx[j + 1]] != clean_touch[touch_idx[j]]
            ):
                is_duel[j] = True
                is_duel[j + 1] = True
        clean_touch[touch_idx[is_duel]] = NONE

    episodes = []
    min_carry_frames = max(2, int(round(cfg.min_carry_s * FPS)))

    # --- A3(a): carry / dribble / hold episodes ---
    carry_team_raw = np.full(n, NONE, dtype=np.int8)
    nearest_slot = np.zeros(n, dtype=np.int32)
    if players_arr.shape[1] > 0:
        diff = players_arr - ball_clean[:, None, :]
        d_all = np.hypot(diff[:, :, 0], diff[:, :, 1])
        d_all = np.where(np.isnan(d_all), np.inf, d_all)
        nearest_slot = np.argmin(d_all, axis=1)
        d_min = d_all[np.arange(n), nearest_slot]
        slot_team = np.where(np.isfinite(d_min), team_arr[nearest_slot], NONE)

        v_player = _nearest_track_velocity(players_arr, nearest_slot)
        have_vp = np.isfinite(v_player[:, 0])

        # Velocity match checks centered velocity, or incoming/outgoing edges so traps start at frame 0
        # while frames immediately after a kick (recent_kick) do not trail into free flight.
        vc = dyn["vel"]
        va = dyn["vel_after"]
        vb = dyn["vel_before"]
        sc = np.where(np.isfinite(vc[:, 0]), np.hypot(vc[:, 0], vc[:, 1]), np.inf)
        sa = np.where(np.isfinite(va[:, 0]), np.hypot(va[:, 0], va[:, 1]), np.inf)
        sb = np.where(np.isfinite(vb[:, 0]), np.hypot(vb[:, 0], vb[:, 1]), np.inf)

        dc = np.where(have_vp & np.isfinite(vc[:, 0]), np.hypot(vc[:, 0] - v_player[:, 0], vc[:, 1] - v_player[:, 1]), np.inf)
        da_v = np.where(have_vp & np.isfinite(va[:, 0]), np.hypot(va[:, 0] - v_player[:, 0], va[:, 1] - v_player[:, 1]), np.inf)
        db_v = np.where(have_vp & np.isfinite(vb[:, 0]), np.hypot(vb[:, 0] - v_player[:, 0], vb[:, 1] - v_player[:, 1]), np.inf)

        recent_kick = np.zeros(n, dtype=bool)
        all_touch_idx = np.flatnonzero(touch_team != NONE)
        for ti in all_touch_idx:
            # Look ahead only 3 frames to stop trailing carry after a kick
            recent_kick[ti + 1:min(n, ti + 4)] = True

        match_c = (dc <= cfg.carry_max_dv) & (sc <= cfg.carry_max_ball_speed)
        match_a = (da_v <= cfg.carry_max_dv) & (sa <= cfg.carry_max_ball_speed)
        match_b = (db_v <= cfg.carry_max_dv) & (sb <= cfg.carry_max_ball_speed) & ~recent_kick
        fallback_slow = ~have_vp & (np.minimum(sa, sb) <= 5.5)
        vel_ok = match_c | match_a | match_b | fallback_slow

        uncontested_carry = valid_obs & ~oob & np.isfinite(d_min) & (d_min <= cfg.carry_radius) & (
            margin >= cfg.contest_margin
        ) & vel_ok
        carry_team_raw = np.where(uncontested_carry, slot_team, NONE)

        for side in (HOME, AWAY):
            side_mask = carry_team_raw == side
            opp_touch = (clean_touch == (1 - side))
            blocked = oob | (carry_team_raw == (1 - side)) | opp_touch
            bridged = _bridge_same_team_gaps(side_mask, cfg.carry_bridge_frames, blocked)

            for s, e in _mask_runs(bridged):
                if e - s < min_carry_frames:
                    continue
                # Verify single-player / spatially continuous track segments within [s, e)
                raw_idx = np.flatnonzero(side_mask[s:e]) + s
                if len(raw_idx) == 0:
                    continue
                # Split raw carry frames by slot changes; merge slots that are spatially continuous (ID swap)
                cur_start = int(raw_idx[0])
                cur_prev = int(raw_idx[0])
                cur_slot = int(nearest_slot[cur_start])
                sub_runs = []
                for idx_t in raw_idx[1:]:
                    idx_t = int(idx_t)
                    sl = int(nearest_slot[idx_t])
                    same_slot = sl == cur_slot
                    if not same_slot:
                        p_old = players_arr[cur_prev, cur_slot]
                        p_new = players_arr[idx_t, sl]
                        if np.isfinite(p_old[0]) and np.isfinite(p_new[0]):
                            swap_ok = np.hypot(*(p_new - p_old)) <= cfg.swap_link_radius
                        else:
                            swap_ok = False
                        ball_slow_here = float(min(sc[idx_t], sa[idx_t], sb[idx_t])) <= 4.5
                        if not (swap_ok or ball_slow_here):
                            sub_runs.append((cur_start, cur_prev + 1))
                            cur_start = idx_t
                    cur_prev = idx_t
                    cur_slot = sl
                sub_runs.append((cur_start, cur_prev + 1))

                for ss, ee in sub_runs:
                    dur = ee - ss
                    if dur >= min_carry_frames:
                        conf = float(np.clip(0.72 + 0.28 * min(1.0, (dur - min_carry_frames) / float(FPS)), 0.72, 1.0))
                        confirm = min(ee - 1, ss + min_carry_frames - 1)
                        episodes.append({
                            "start": ss,
                            "end": ee,
                            "team": int(side),
                            "confidence": conf,
                            "kind": "carry",
                            "confirm": int(confirm),
                        })

    # Pre-index opposing carry intervals to block false passing sequences across opposing control
    carry_opp_mask = {HOME: np.zeros(n, dtype=bool), AWAY: np.zeros(n, dtype=bool)}
    for ep in episodes:
        carry_opp_mask[ep["team"]][ep["start"]:ep["end"]] = True

    # --- A3(b): consecutive uncontested same-team touches within max_pass_interval_s ---
    max_pass_frames = max(2, int(round(cfg.max_pass_interval_s * FPS)))
    clean_idx = np.flatnonzero(clean_touch != NONE)
    for j in range(len(clean_idx) - 1):
        t1 = int(clean_idx[j])
        t2 = int(clean_idx[j + 1])
        side1 = int(clean_touch[t1])
        side2 = int(clean_touch[t2])
        if side1 != side2 or t2 - t1 > max_pass_frames or t2 <= t1:
            continue
        opp = 1 - side1
        if np.any(oob[t1:t2 + 1]) or np.any(carry_opp_mask[opp][t1:t2 + 1]):
            continue
        conf = float(np.clip(0.5 * (float(touch[t1]) + float(touch[t2])) + 0.25, 0.65, 1.0))
        episodes.append({
            "start": t1,
            "end": t2 + 1,
            "team": side1,
            "confidence": conf,
            "kind": "pass",
            "confirm": t2,
        })

    episodes.sort(key=lambda ep: (ep["confirm"], ep["start"]))
    return episodes


def compute_control_evidence(n, episodes, touch, touch_team, uncontested_touch, sig_so_far, dyn, cfg=None):
    """A4/A5: Compute per-frame `ctrl_logodds` and `ctrl_conf` from control episodes and touches.

    Rules:
    - Possession follows the most recent established control episode.
    - A lone opposing touch (deflection, duel, brief poke) NEVER switches possession; a switch
      only occurs at `ep["confirm"]` of an opposing control episode.
    - While the ball is in flight (no one in range), the last controlling team is retained with
      gentle exponential confidence decay (`flight_decay_s`).
    - Where nothing is known (before control or on `~ball_ok` missing stretches or sustained `oob`),
      output evidence is 0.0 so the state decoder persists or enters D cleanly.
    """
    cfg = cfg or TouchConfig()
    ctrl_logodds = np.zeros(n, dtype=np.float32)
    ctrl_conf = np.zeros(n, dtype=np.float32)
    if n == 0:
        return ctrl_logodds, ctrl_conf

    ball_ok = dyn["ball_ok"]
    oob = np.asarray(sig_so_far["oob"], dtype=bool) if "oob" in sig_so_far else np.zeros(n, dtype=bool)
    decay = float(np.exp(-1.0 / (max(cfg.flight_decay_s, 0.1) * FPS)))
    oob_limit = max(1, int(round(cfg.oob_reset_s * FPS)))

    # Pre-build per-frame episode triggers so frame loop is purely O(T)
    confirm_ep = {}
    for ep in episodes:
        confirm_ep.setdefault(ep["confirm"], []).append(ep)

    # Active carry span mask per team: refreshes confidence throughout an active carry of current team
    active_carry = np.full(n, NONE, dtype=np.int8)
    active_carry_conf = np.zeros(n, dtype=np.float32)
    for ep in episodes:
        if ep["kind"] == "carry":
            s, e = ep["start"], ep["end"]
            active_carry[s:e] = ep["team"]
            active_carry_conf[s:e] = np.maximum(active_carry_conf[s:e], ep["confidence"])

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

        # Decay smoothly across gaps and flight
        cur_conf *= decay

        # First check confirmed control episodes ending/confirming at frame t
        if t in confirm_ep:
            for ep in confirm_ep[t]:
                ep_team = ep["team"]
                if cur_team == NONE or ep_team != cur_team:
                    cur_team = ep_team
                    cur_conf = max(cur_conf, ep["confidence"])
                else:
                    cur_conf = max(cur_conf, ep["confidence"])

        # Same-team active carry sustains confidence while held
        if cur_team != NONE and active_carry[t] == cur_team:
            cur_conf = max(cur_conf, float(active_carry_conf[t]))

        # Same-team touch refreshes confidence; initial uncontested touch only when cur_team == NONE
        if touch_team[t] != NONE:
            tt = int(touch_team[t])
            if cur_team == tt:
                cur_conf = max(cur_conf, float(touch[t]))
            elif cur_team == NONE and uncontested_touch[t]:
                cur_team = tt
                cur_conf = max(cur_conf, 0.55 * float(touch[t]))

        if cur_conf < 1e-3:
            cur_team = NONE
            cur_conf = 0.0

        if cur_team != NONE and ball_ok[t]:
            ctrl_conf[t] = cur_conf
            sign = 1.0 if cur_team == HOME else -1.0
            ctrl_logodds[t] = sign * min(cfg.ctrl_cap, cfg.ctrl_gain * cur_conf)

    return (
        np.nan_to_num(ctrl_logodds, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        np.nan_to_num(ctrl_conf, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
    )


def build_touch_signals(ball, players, team, sig_so_far, observed=None, filled=None, cfg=None,
                        return_details=False):
    """Compute touch & control signals and add them to `sig_so_far` (additive, in-place)."""
    cfg = cfg or TouchConfig()
    dyn = ball_dynamics(ball, observed=observed, filled=filled, cfg=cfg)
    touch, touch_team, uncontested_touch = detect_touches(sig_so_far, dyn, cfg=cfg)
    episodes = detect_control_episodes(
        players, team, sig_so_far, touch, touch_team, uncontested_touch, dyn, cfg=cfg
    )
    ctrl_logodds, ctrl_conf = compute_control_evidence(
        len(touch), episodes, touch, touch_team, uncontested_touch, sig_so_far, dyn, cfg=cfg
    )

    sig_so_far["touch"] = touch.astype(np.float32)
    sig_so_far["touch_team"] = touch_team.astype(np.int8)
    sig_so_far["ctrl_logodds"] = ctrl_logodds.astype(np.float32)
    sig_so_far["ctrl_conf"] = ctrl_conf.astype(np.float32)

    if return_details:
        return sig_so_far, {"episodes": episodes, "dyn": dyn, "uncontested_touch": uncontested_touch}
    return sig_so_far
