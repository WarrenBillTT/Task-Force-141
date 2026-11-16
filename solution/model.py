"""Possession as a state sequence: per-frame evidence -> Viterbi over {H, A, D} -> cleanup.

Why a state model and not a vote: spellF1 scores whole possessions, so the one thing that matters
is not splitting a possession and not merging two. A switch has to be *paid for* with sustained
evidence; Viterbi with a switching penalty does exactly that, and where the ball is unobserved the
evidence is zero, so the current state simply persists (the scorer forward-fills abstentions the
same way, so this costs nothing inside a possession).
"""

from dataclasses import dataclass

import numpy as np

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])


@dataclass
class Config:
    # --- team evidence (who is nearer the ball) ---
    gap_gain: float = 0.4        # log-odds per metre of (away - home) nearest-distance gap
    gap_cap: float = 1.2         # cap on that log-odds, so one frame never decides alone
    control_radius: float = 6.0  # m. Ball further than this from everyone is in flight: weak evidence
    # --- dead-ball evidence (score added to D) ---
    oob_bonus: float = 2.0       # ball beyond a line
    restart_bonus: float = 0.6   # slow ball sitting ON a line (throw-in / goal-kick / corner set-up)
    idle_bonus: float = 0.6      # slow ball with nobody within idle_distance
    setpiece_bonus: float = 0.6  # slow ball inside a Smart Tagging set-piece window
    live_bonus: float = 1.0      # subtracted from D when the ball is moving fast in bounds
    slow_speed: float = 1.0      # m/s
    live_speed: float = 4.0      # m/s
    idle_distance: float = 3.0   # m
    # --- transitions (nats) ---
    switch_penalty: float = 10.0   # H <-> A: must be earned with ~1-2 s of consistent evidence
    dead_penalty: float = 6.0      # H/A <-> D
    # --- cleanup (frames) ---
    min_alive: int = 38            # a team run shorter than ~1.5 s is a touch/deflection, not control
    min_dead: int = 25             # a dead-ball run shorter than 1 s is noise
    # --- touch / control evidence (solution/touch.py, Approach A) ---
    use_touch: int = 1             # 1 (--touch): use ctrl_logodds; 0 (--no-touch): proximity only
    touch_weight: float = 0.85     # blend weight on ctrl_logodds vs nearest-player proximity llr


def emissions(sig, cfg):
    """(T, 3) log-score of each state per frame, up to a constant. NaN ball -> all zeros."""
    dh, da = sig["dist_home"], sig["dist_away"]
    nearest = np.minimum(dh, da)
    known = np.isfinite(nearest)

    with np.errstate(invalid="ignore"):  # inf - inf: neither team tracked, no evidence
        gap = np.clip(np.nan_to_num(da - dh, nan=0.0, posinf=15.0, neginf=-15.0), -15.0, 15.0)
    llr = np.clip(cfg.gap_gain * gap, -cfg.gap_cap, cfg.gap_cap)
    weight = np.where(known, np.exp(-(np.where(known, nearest, 0.0) / cfg.control_radius) ** 2), 0.0)
    prox_team = 0.5 * weight * llr

    speed = sig["speed"]
    have_speed = ~np.isnan(speed)
    slow = have_speed & (speed < cfg.slow_speed)
    fast = have_speed & (speed > cfg.live_speed)
    ball_ok = sig["ball_ok"]

    if cfg.use_touch and "ctrl_logodds" in sig:
        ctrl = np.nan_to_num(sig["ctrl_logodds"], nan=0.0)
        has_ctrl = np.abs(ctrl) > 1e-6
        dead_cand = slow & (sig["near_line"] | sig["set_piece"] | (nearest > cfg.idle_distance))
        ctrl_scale = np.where(dead_cand, np.where(known, 0.5 * weight, 0.0), 1.0)
        ctrl_team = np.where(ball_ok & ~sig["oob"], 0.5 * ctrl_scale * ctrl, 0.0)
        team = np.where(has_ctrl, cfg.touch_weight * ctrl_team + (1.0 - cfg.touch_weight) * prox_team, prox_team)
    else:
        team = prox_team

    dead = np.zeros(len(team), dtype=np.float32)
    dead += cfg.oob_bonus * (sig["oob"] & ball_ok)
    dead += cfg.restart_bonus * (slow & sig["near_line"] & ball_ok)
    dead += cfg.idle_bonus * (slow & (nearest > cfg.idle_distance) & ball_ok)
    dead += cfg.setpiece_bonus * (slow & sig["set_piece"] & ball_ok)
    dead -= cfg.live_bonus * (fast & ~sig["oob"] & ball_ok)

    emit = np.zeros((len(team), 3), dtype=np.float32)
    emit[:, H] = team
    emit[:, A] = -team
    emit[:, D] = dead
    return emit


def transition_matrix(cfg):
    t = np.full((3, 3), -cfg.dead_penalty, dtype=np.float32)
    t[H, A] = t[A, H] = -cfg.switch_penalty
    np.fill_diagonal(t, 0.0)
    return t


def viterbi(emit, trans):
    steps = len(emit)
    back = np.zeros((steps, 3), dtype=np.int8)
    score = emit[0].astype(np.float64)
    for t in range(1, steps):
        cand = score[:, None] + trans
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + emit[t]
    path = np.zeros(steps, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(steps - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _runs(path):
    change = np.flatnonzero(np.diff(path)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(path)]))
    return [[int(s), int(e), int(path[s])] for s, e in zip(starts, ends)]


def merge_short_runs(path, cfg):
    """Absorb runs too short to be real: touches/deflections (H/A) and flicker (D).

    The shortest offending run goes to its neighbour: if both neighbours agree, to that label
    (so H A H becomes one H possession, which is what a deflection is); otherwise to the longer one.
    """
    path = path.copy()
    while True:
        runs = _runs(path)
        if len(runs) < 2:
            return path
        worst = None
        for k, (s, e, label) in enumerate(runs):
            limit = cfg.min_dead if label == D else cfg.min_alive
            if e - s < limit and (worst is None or e - s < runs[worst][1] - runs[worst][0]):
                worst = k
        if worst is None:
            return path
        s, e, _ = runs[worst]
        left = runs[worst - 1] if worst > 0 else None
        right = runs[worst + 1] if worst + 1 < len(runs) else None
        if left and right:
            if left[2] == right[2]:
                new = left[2]
            else:
                new = left[2] if (left[1] - left[0]) >= (right[1] - right[0]) else right[2]
        else:
            new = (left or right)[2]
        path[s:e] = new


def decode(sig, cfg):
    """Evidence for one contiguous stretch of play -> label path (0/1/2 per frame)."""
    path = viterbi(emissions(sig, cfg), transition_matrix(cfg))
    return merge_short_runs(path, cfg)
