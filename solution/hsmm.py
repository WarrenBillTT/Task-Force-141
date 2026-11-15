"""Explicit-duration (hidden semi-Markov) decoder over {H, A, D}.

Why: the 3-state Viterbi charges a flat price per switch and nothing for *how long* a state lasts, so
a short burst of misleading evidence can still split a possession, and a missed short stoppage can
merge two. spellF1 punishes exactly those errors. A semi-Markov decoder scores whole segments:

    segment score = sum of emission scores over its frames
                    + dur_scale * log P(duration | label)      (log-normal, from football knowledge)
                    + transition score from the previous segment's label

The duration priors live in `model.Config` (every number has its physical reasoning there); nothing
here is derived from match data or labels.

Efficiency: cumulative sums make a segment's emission O(1); the DP runs on a coarse grid (every
`hsmm_step` frames) and vectorises over segment durations with numpy; boundaries are then re-polished
at full resolution against the emissions. With step=1 the decoder is exact and is tested against a
brute-force enumeration (`brute_force`).

Conventions
-----------
* Frames with no ball evidence have emission 0, so they add nothing to any segment: the label simply
  persists through them, as in the Viterbi.
* The first and last segment of the stretch are censored (their true extent is unknown), so they
  carry no duration prior.
* Segments longer than `hsmm_max_seconds` are not scored as one; the decoder may chain two segments of
  the same label at the price of one switch penalty (keeps the problem always feasible).
"""

import itertools
import math

import numpy as np

import model

FPS = 25
H, A, D = model.H, model.A, model.D


def _max_frames(cfg):
    return max(int(cfg.hsmm_max_seconds * FPS), 1)


def duration_table(cfg, max_frames=None):
    """(max_frames + 1, 3) weighted log-duration prior in nats, indexed by duration in frames.

    Log-normal density per second, evaluated at the segment length. H and A share one prior.
    Row 0 is unused (-inf).
    """
    max_frames = max_frames or _max_frames(cfg)
    d = np.maximum(np.arange(max_frames + 1), 1) / float(FPS)
    out = np.empty((max_frames + 1, 3))
    for label, median, sigma in ((H, cfg.alive_median_s, cfg.alive_sigma),
                                 (A, cfg.alive_median_s, cfg.alive_sigma),
                                 (D, cfg.dead_median_s, cfg.dead_sigma)):
        z = (np.log(d) - math.log(median)) / sigma
        out[:, label] = -np.log(d * sigma * math.sqrt(2 * math.pi)) - 0.5 * z * z
    out *= cfg.dur_scale
    out[0] = -np.inf
    return out


def transition_scores(cfg):
    """(3, 3) segment-to-segment score. The diagonal is the cost of chaining two same-label segments."""
    t = np.full((3, 3), -cfg.dead_penalty * cfg.hsmm_penalty_scale)
    t[H, A] = t[A, H] = -cfg.switch_penalty * cfg.hsmm_penalty_scale
    np.fill_diagonal(t, -cfg.switch_penalty * cfg.hsmm_penalty_scale)
    return t


def segmentation_score(emit, segments, cfg, table=None):
    """Score of an explicit segmentation [(start, end, label), ...] covering the whole stretch.

    The definition the DP maximises; also used by `brute_force` and the tests.
    """
    table = duration_table(cfg) if table is None else table
    trans = transition_scores(cfg)
    n = len(emit)
    csum = np.vstack([np.zeros(3), np.cumsum(emit.astype(np.float64), axis=0)])
    total, prev = 0.0, None
    for s, e, label in segments:
        if e - s >= len(table):
            return -np.inf
        total += csum[e, label] - csum[s, label]
        if s != 0 and e != n:
            total += table[e - s, label]
        if prev is not None:
            total += trans[prev, label]
        prev = label
    return total


def brute_force(emit, cfg):
    """Exhaustive best segmentation for tiny T. Returns (best_score, path). Reference only."""
    n = len(emit)
    table = duration_table(cfg)
    best_score, best_segments = -np.inf, None
    for mask in range(1 << (n - 1)):
        cuts = [0] + [i + 1 for i in range(n - 1) if mask >> i & 1] + [n]
        spans = list(zip(cuts[:-1], cuts[1:]))
        for labels in itertools.product(range(3), repeat=len(spans)):
            segs = [(s, e, l) for (s, e), l in zip(spans, labels)]
            score = segmentation_score(emit, segs, cfg, table)
            if score > best_score + 1e-12:
                best_score, best_segments = score, segs
    return best_score, segments_to_path(best_segments, n)


def segments_to_path(segments, n):
    path = np.zeros(n, dtype=np.int8)
    for s, e, label in segments:
        path[s:e] = label
    return path


def path_to_segments(path):
    path = np.asarray(path)
    change = np.flatnonzero(np.diff(path)) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(path)]))
    return [(int(s), int(e), int(path[s])) for s, e in zip(starts, ends)]


def decode_segments(emit, cfg, step=None):
    """Best segmentation [(start, end, label), ...] of a (T, 3) emission array."""
    emit = np.asarray(emit, dtype=np.float64)
    n = len(emit)
    step = max(int(cfg.hsmm_step if step is None else step), 1)
    if n == 0:
        return []
    cap = max(_max_frames(cfg), step)
    table = duration_table(cfg, cap)
    trans = transition_scores(cfg)

    m = -(-n // step)
    pos = np.minimum(np.arange(m + 1) * step, n)
    csum = np.vstack([np.zeros(3), np.cumsum(emit, axis=0)])
    cg = csum[pos]

    best = np.full((m + 1, 3), -np.inf)
    entry = np.full((m + 1, 3), -np.inf)      # best score of a prefix ending at i, plus the switch into l
    entry[0] = 0.0
    back_i = np.zeros((m + 1, 3), dtype=np.int64)
    back_prev = np.zeros((m + 1, 3), dtype=np.int64)
    lanes = np.arange(3)

    for j in range(1, m + 1):
        lo = int(np.searchsorted(pos, pos[j] - cap, side="left"))
        lo = min(lo, j - 1)
        dur = table[pos[j] - pos[lo:j]]
        if j == m:
            dur = np.zeros_like(dur)           # last segment is censored
        elif lo == 0:
            dur[0] = 0.0                       # first segment is censored
        cand = entry[lo:j] - cg[lo:j] + dur
        k = cand.argmax(axis=0)
        best[j] = cand[k, lanes] + cg[j]
        back_i[j] = lo + k
        if j < m:
            into = best[j][:, None] + trans    # [from, to]
            back_prev[j] = into.argmax(axis=0)
            entry[j] = into.max(axis=0)

    label = int(best[m].argmax())
    j, bounds = m, []
    while j > 0:
        i = int(back_i[j, label])
        bounds.append((int(pos[i]), int(pos[j]), label))
        j, label = i, int(back_prev[i, label])
    segments = bounds[::-1]
    return _polish(segments, csum, step) if step > 1 else segments


def _polish(segments, csum, step):
    """Slide each boundary within the coarse cell to the full-resolution best emission split."""
    segs = [list(s) for s in segments]
    for k in range(len(segs) - 1):
        left, right = segs[k], segs[k + 1]
        b = left[1]
        lo = max(left[0] + 1, b - step + 1)
        hi = min(right[1] - 1, b + step - 1)
        if hi < lo:
            continue
        cand = np.arange(lo, hi + 1)
        gain = csum[cand, left[2]] - csum[cand, right[2]]
        top = np.flatnonzero(gain >= gain.max() - 1e-12)
        pick = cand[top[np.argmin(np.abs(cand[top] - b))]]
        left[1] = right[0] = int(pick)
    return [tuple(s) for s in segs]


def decode(emit, cfg, step=None):
    """Label path (int8 per frame) of the best explicit-duration segmentation."""
    return segments_to_path(decode_segments(emit, cfg, step), len(emit))
