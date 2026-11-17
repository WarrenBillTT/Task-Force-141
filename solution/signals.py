"""Turn raw tracking into per-frame evidence: a cleaned ball track, who is near it, is it live.

Nothing here looks at labels. Every threshold is a physical quantity (metres, m/s, seconds), chosen
from football knowledge, not fitted to this match.
"""

import numpy as np

FPS = 25


def clean_ball(ball, vmax=45.0, max_gap=int(1.5 * FPS)):
    """Reject impossible jumps, then linearly bridge short gaps.

    vmax    m/s. A real ball tops out near 35 m/s, so anything faster than this between two
            observations is a detection error (a spike on a line marking, a boot, a corner flag).
    max_gap frames. Gaps up to this long are interpolated; longer ones stay NaN and carry no
            evidence (the state model simply persists through them).

    Returns (ball, observed, filled): the cleaned track and two boolean masks.
    """
    out = ball.copy()
    n = len(out)

    last = None
    for i in np.flatnonzero(~np.isnan(out[:, 0])):
        if last is not None:
            seconds = (i - last) / float(FPS)
            if np.hypot(*(out[i] - out[last])) / seconds > vmax:
                out[i] = np.nan  # reject; `last` stays, so the allowed jump grows with the gap
                continue
        last = i

    observed = ~np.isnan(out[:, 0])
    valid = np.flatnonzero(observed)
    filled = np.zeros(n, dtype=bool)
    for a, b in zip(valid[:-1], valid[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            t = np.linspace(0.0, 1.0, gap + 2)[1:-1, None]
            out[a + 1:b] = out[a] * (1 - t) + out[b] * t
            filled[a + 1:b] = True
    return out, observed, filled


def ball_speed(ball, half_window=5):
    """Ball speed in m/s from a centred difference over +/- half_window frames. NaN if unknown."""
    padded = np.full((len(ball) + 2 * half_window, 2), np.nan, dtype=np.float32)
    padded[half_window:-half_window] = ball
    ahead = padded[2 * half_window:]
    behind = padded[:-2 * half_window]
    return np.hypot(*(ahead - behind).T) * FPS / (2.0 * half_window)


def team_distances(ball, players, team):
    """Distance from the ball to the nearest home and nearest away player. inf if none tracked."""
    out = []
    for side in (0, 1):
        d = np.linalg.norm(players[:, team == side, :] - ball[:, None, :], axis=2)
        d = np.where(np.isnan(d), np.inf, d)
        out.append(d.min(axis=1) if d.shape[1] else np.full(len(ball), np.inf))
    return out[0].astype(np.float32), out[1].astype(np.float32)


def team_speed(speed, window=FPS):
    """Average speed of every tracked player, over a centred window of `window` frames.

    During a stoppage nearly everyone walks; during play the average sits well above walking pace.
    It needs no ball, which matters because a third of dead-ball frames have none.
    """
    seen = ~np.isnan(speed)
    total = np.where(seen, speed, 0.0).sum(axis=1)
    count = seen.sum(axis=1).astype(np.float64)
    kernel = np.ones(window)
    total = np.convolve(total, kernel, "same")
    count = np.convolve(count, kernel, "same")
    return np.where(count > 0, total / np.maximum(count, 1.0), np.nan).astype(np.float32)


def out_of_play(ball, pitch, margin=0.5):
    """True where the ball is beyond a touchline or goal line by more than `margin` metres."""
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    return (x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin)


def near_boundary(ball, pitch, distance=0.6):
    """True where the ball is within `distance` metres of a touchline or goal line."""
    length, width = pitch
    x, y = ball[:, 0], ball[:, 1]
    return (x < distance) | (x > length - distance) | (y < distance) | (y > width - distance)


def window_mask(n, intervals):
    """Boolean mask of length n that is True inside any [start, end) frame interval."""
    mask = np.zeros(n, dtype=bool)
    for start, end in intervals:
        mask[max(0, start):max(0, min(n, end))] = True
    return mask
