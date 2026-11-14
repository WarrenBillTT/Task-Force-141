"""Possession as a state sequence: per-frame evidence -> Viterbi over {H, A, D} -> cleanup.

Why a state model and not a vote: spellF1 scores whole possessions, so the one thing that matters
is not splitting a possession and not merging two. A switch has to be *paid for* with sustained
evidence; Viterbi with a switching penalty does exactly that, and where the ball is unobserved the
evidence is zero, so the current state simply persists (the scorer forward-fills abstentions the
same way, so this costs nothing inside a possession).

Team evidence (who is near the ball, which side the Smart Tagging phase belongs to) is mostly about
H versus A. Against D it is capped at log 2 a frame, because before a restart somebody is standing
at the ball too. The first version had no cap, so a player near the ball always outvoted the
dead-ball cues and the model almost never said D. Dead-ball evidence (ball over a line, players
walking, a set piece on) is what decides between D and play.
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
    tag_gain: float = 0.1        # per frame inside a Smart Tagging phase that names the side in
                                 # possession. Small next to clear ball control (up to 0.6), but
                                 # it adds up over a phase that lasts many seconds
    # --- dead-ball evidence (score added to D) ---
    oob_bonus: float = 2.0       # ball beyond a line
    restart_bonus: float = 0.6   # slow ball sitting ON a line (throw-in / goal-kick / corner set-up)
    idle_bonus: float = 0.6      # slow ball with nobody within idle_distance
    setpiece_bonus: float = 0.1  # inside a Smart Tagging set-piece window. Weak: the window also
                                 # covers the play after the restart
    live_bonus: float = 0.3      # subtracted from D when the ball is moving fast in bounds. Not
                                 # stronger: the ball is also thrown or kicked back during stoppages
    motion_gain: float = 1.0     # per frame, per m/s of team speed below walk_run_speed (capped at
                                 # 1 m/s). Everyone walking (~1 m/s) beats the log 2 a controlled
                                 # ball can say against D; a lull at 1.5 m/s does not
    walk_run_speed: float = 1.8  # m/s. Walking is ~1.4; a match averages ~1.9 (about 11 km in 95 min)
    slow_speed: float = 1.0      # m/s
    live_speed: float = 4.0      # m/s
    idle_distance: float = 3.0   # m
    # --- transitions (nats) ---
    switch_penalty: float = 10.0   # H <-> A: must be earned with ~1-2 s of consistent evidence
    dead_penalty: float = 6.0      # H/A <-> D
    # --- cleanup (frames) ---
    min_alive: int = 38            # a team run shorter than ~1.5 s is a touch/deflection, not control
    min_dead: int = 50             # even a quick throw-in takes longer than 2 s to restart


def emissions(sig, cfg):
    """(T, 3) log-score of each state per frame, up to a constant. NaN ball -> all zeros."""
    dh, da = sig["dist_home"], sig["dist_away"]
    nearest = np.minimum(dh, da)
    known = np.isfinite(nearest)

    with np.errstate(invalid="ignore"):  # inf - inf: neither team tracked, no evidence
        gap = np.clip(np.nan_to_num(da - dh, nan=0.0, posinf=15.0, neginf=-15.0), -15.0, 15.0)
    llr = np.clip(cfg.gap_gain * gap, -cfg.gap_cap, cfg.gap_cap)
    weight = np.where(known, np.exp(-(np.where(known, nearest, 0.0) / cfg.control_radius) ** 2), 0.0)
    team = 0.5 * weight * llr + cfg.tag_gain * sig["tag_side"]

    speed = sig["speed"]
    have_speed = ~np.isnan(speed)
    slow = have_speed & (speed < cfg.slow_speed)
    fast = have_speed & (speed > cfg.live_speed)
    ball_ok = sig["ball_ok"]

    dead = np.zeros(len(team), dtype=np.float32)
    dead += cfg.oob_bonus * (sig["oob"] & ball_ok)
    dead += cfg.restart_bonus * (slow & sig["near_line"] & ball_ok)
    dead += cfg.idle_bonus * (slow & (nearest > cfg.idle_distance) & ball_ok)
    dead += cfg.setpiece_bonus * sig["set_piece"]
    dead -= cfg.live_bonus * (fast & ~sig["oob"] & ball_ok)
    # No ball needed for this one; NaN (nobody tracked) gives nothing either way.
    walking = np.clip(cfg.walk_run_speed - sig["team_speed"], -1.0, 1.0)
    dead += cfg.motion_gain * np.nan_to_num(walking)

    # log(2 * sigmoid(odds)): H minus A is still 2 * team, no evidence still scores 0 everywhere,
    # and the leading side never gains more than log 2 over D.
    odds = 2 * team
    emit = np.zeros((len(team), 3), dtype=np.float32)
    emit[:, H] = np.log(2) - np.logaddexp(0, -odds)
    emit[:, A] = np.log(2) - np.logaddexp(0, odds)
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
