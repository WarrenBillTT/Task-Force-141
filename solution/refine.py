"""Boundary refinement: slide each decoded boundary to where the local signals say the change happened.

Why: the decoder works with smooth, delayed evidence (a possession change needs ~1-2 s of consistent
evidence before it is "paid for"), so its boundaries sit a bit late. spellF1 matches spells by temporal
IoU, so a boundary that is off by a second costs on every short spell. Refinement fixes the *position*
only: labels never change, no segment is created or removed, and no segment may end up shorter than the
configured minimum (it may stay as short as it already was).

For each boundary, within +/- `refine_window_s`, pick the frame b maximising

    sum_{t<b} cue_left(t) + sum_{t>=b} cue_right(t)          (a step fit of the local evidence)
    + dv_weight * (ball velocity change near b)              (kick / trap / deflection)
    + touch_weight * (touch score near b)                    (only if sig["touch"] exists)
    - drift * |b - decoder position|                         (ties stay where the decoder put them)

Cues (all from `sig`, all in physical units):
* H <-> A:       which team is nearer the ball (only while someone is within `refine_radius`: a ball in
                 flight says nothing). The boundary is the first frame where the *new* team is clearly the
                 nearest, and the ball's velocity changes there.
* alive -> D:    ball beyond a line, or stopped (slow). Evidence for D is the ball leaving play / stopping.
* D -> alive:    ball moving fast again. The boundary is where it starts moving (kick), with the same
                 velocity-change credit.
`sig["touch"]` is an optional per-frame touch score in [0, 1] from a future touch detector; it is used
when present and ignored otherwise.
"""

import numpy as np

import model

FPS = 25
H, A, D = model.H, model.A, model.D
DV_SPREAD = 3  # frames: the velocity change is a spike a frame or two wide; credit it within this reach


def _near_max(x, reach):
    """Max of x over +/- reach frames (x has no NaN)."""
    out = x.copy()
    for k in range(1, reach + 1):
        out[k:] = np.maximum(out[k:], x[:-k])
        out[:-k] = np.maximum(out[:-k], x[k:])
    return out


def team_cue(sig, cfg):
    """Per frame, > 0 favours home, < 0 favours away; 0 when the ball is in flight or unobserved."""
    dh, da = sig["dist_home"], sig["dist_away"]
    nearest = np.minimum(dh, da)
    with np.errstate(invalid="ignore"):
        gap = np.clip(np.nan_to_num(da - dh, nan=0.0, posinf=15.0, neginf=-15.0), -15.0, 15.0)
        close = np.isfinite(nearest) & (nearest < cfg.refine_radius)
    return np.where(close, np.tanh(gap / cfg.refine_margin), 0.0)


def dead_cue(sig, cfg):
    """Per frame, > 0 favours a dead ball, < 0 favours live play, 0 when unknown.

    A slow ball is only weak evidence of a dead ball (a player shielding or trapping it is slow too), so
    it gets `refine_slow_weight`; leaving the pitch (+2) and a clearly fast ball (-1) are the strong cues.
    """
    ok = sig["ball_ok"]
    speed = sig["speed"]
    with np.errstate(invalid="ignore"):
        slow = ok & (speed < cfg.slow_speed)
        fast = ok & (speed > cfg.live_speed)
    oob = sig["oob"] & ok
    return np.where(oob, 2.0, np.where(slow, cfg.refine_slow_weight, np.where(fast, -1.0, 0.0)))


def event_bonus(sig, cfg):
    """Per frame credit for 'something happened to the ball here' (velocity change and/or touch)."""
    n = len(sig["speed"])
    bonus = np.zeros(n)
    dv = sig.get("ball_dv")
    if dv is not None:
        unit = np.clip(np.nan_to_num(np.asarray(dv, dtype=np.float64), nan=0.0) / cfg.refine_dv_ref, 0, 1)
        bonus += cfg.refine_dv_weight * _near_max(unit, DV_SPREAD)
    touch = sig.get("touch")
    if touch is not None:
        unit = np.clip(np.nan_to_num(np.asarray(touch, dtype=np.float64), nan=0.0), 0, 1)
        bonus += cfg.refine_touch_weight * _near_max(unit, DV_SPREAD)
    return bonus


def _cue_pair(la, lb, team, dead):
    """(cue for the left label, cue for the right label) as per-frame arrays."""
    if la != D and lb != D:                    # H <-> A
        sign = 1.0 if la == H else -1.0
        return sign * team, -sign * team
    if la == D:                                # D -> alive: dead evidence supports the left side
        return dead, -dead
    return -dead, dead                         # alive -> D


def refine(path, sig, cfg):
    """Refined copy of a label path (int8). Same labels, same number of segments, same minimum lengths."""
    path = np.asarray(path).astype(np.int8)
    n = len(path)
    change = np.flatnonzero(np.diff(path)) + 1
    if len(change) == 0:
        return path.copy()
    starts = np.concatenate(([0], change))
    labels = path[starts]
    bounds = list(change.astype(int))          # bounds[k] = first frame of segment k + 1
    orig_bounds = list(bounds)

    team = team_cue(sig, cfg)
    dead = dead_cue(sig, cfg)
    bonus = event_bonus(sig, cfg)
    reach = int(round(cfg.refine_window_s * FPS))

    def limit(label):
        return cfg.min_dead if label == D else cfg.min_alive

    seg_start = 0
    for k in range(len(bounds)):
        la, lb = int(labels[k]), int(labels[k + 1])
        b = orig_bounds[k]
        right_end = orig_bounds[k + 1] if k + 1 < len(orig_bounds) else n
        # keep each side at least as long as the configured minimum (or as long as it already was)
        min_left = min(limit(la), orig_bounds[k] - (orig_bounds[k - 1] if k else 0))
        min_right = min(limit(lb), right_end - b)
        lo = max(b - reach, seg_start + min_left)
        hi = min(b + reach, right_end - min_right)
        if hi < lo:
            seg_start = bounds[k]
            continue
        cand = np.arange(lo, hi + 1)
        cue_l, cue_r = _cue_pair(la, lb, team, dead)
        diff = np.concatenate(([0.0], np.cumsum((cue_l - cue_r)[lo:hi])))   # sum over [lo, cand)
        score = diff - cfg.refine_drift * np.abs(cand - b) + bonus[cand]
        top = np.flatnonzero(score >= score.max() - 1e-12)
        bounds[k] = int(cand[top[np.argmin(np.abs(cand[top] - b))]])
        seg_start = bounds[k]

    out = np.empty(n, dtype=np.int8)
    edges = [0] + bounds + [n]
    for k, label in enumerate(labels):
        out[edges[k]:edges[k + 1]] = label
    return out
