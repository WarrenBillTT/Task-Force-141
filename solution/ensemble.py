"""Consensus ensemble: decode under K deterministic parameter variants, keep the boundaries most agree on.

Why: any single decoder setting is a bet on the switching price and on the duration priors, and a
boundary that only exists under one setting is more likely noise than a real change of possession.
Boundaries found by at least half of the variants (clustered within ~1 s) are kept, the label path is
rebuilt from them, and the usual cleanup is re-applied. The fraction of variants that agree with each
final segment is returned as a stability score for diagnostics.

Deterministic: the variants are a fixed grid (no randomness), ordered by distance from the base
setting, so `--ensemble N` always means the same N variants.
"""

import dataclasses

import numpy as np

import hsmm
import model
import refine as refine_mod

FPS = 25
SWITCH_MULTS = (0.6, 0.8, 1.0, 1.25, 1.6)     # x switch_penalty and dead_penalty
DURATION_MULTS = (0.4, 0.7, 1.0, 1.4, 2.0)    # x dur_scale (HSMM only; ignored by the Viterbi)
MAX_VARIANTS = len(SWITCH_MULTS) * len(DURATION_MULTS)
TOLERANCE_S = 1.0                             # boundaries closer than this are "the same boundary"


def variants(cfg, k):
    """First k configs of the grid, nearest to the base setting first (the base itself is first)."""
    if not 1 <= k <= MAX_VARIANTS:
        raise ValueError("ensemble size must be 1..{}".format(MAX_VARIANTS))
    grid = [(abs(np.log(a)) + abs(np.log(b)), i, a, b)
            for i, (a, b) in enumerate((a, b) for a in SWITCH_MULTS for b in DURATION_MULTS)]
    out = []
    for _, _, a, b in sorted(grid)[:k]:
        out.append(dataclasses.replace(
            cfg, switch_penalty=cfg.switch_penalty * a, dead_penalty=cfg.dead_penalty * a,
            dur_scale=cfg.dur_scale * b, ensemble=1))
    return out


def decode_single(sig, cfg, emit=None):
    """One decoder run: emissions -> (viterbi | hsmm) -> cleanup -> optional boundary refinement."""
    emit = model.emissions(sig, cfg) if emit is None else emit
    if cfg.decoder == "hsmm":
        path = hsmm.decode(emit, cfg)
    elif cfg.decoder == "viterbi":
        path = model.viterbi(emit, model.transition_matrix(cfg))
    else:
        raise ValueError("unknown decoder {!r}".format(cfg.decoder))
    path = model.merge_short_runs(path, cfg)
    if cfg.refine:
        path = refine_mod.refine(path, sig, cfg)
    return path


def _boundaries(path):
    return np.flatnonzero(np.diff(path)) + 1


def consensus(paths, cfg, tolerance=None):
    """Combine label paths (K, T) into one: boundaries supported by >= half of the paths survive."""
    paths = np.asarray(paths)
    k, n = paths.shape
    tol = int(round(TOLERANCE_S * FPS)) if tolerance is None else tolerance

    events = sorted((int(f), v) for v in range(k) for f in _boundaries(paths[v]))
    clusters, current = [], []
    for f, v in events:
        if current and f - current[0][0] > tol:
            clusters.append(current)
            current = []
        current.append((f, v))
    if current:
        clusters.append(current)
    kept = []
    for cluster in clusters:
        if 2 * len({v for _, v in cluster}) >= k:
            frames = sorted(f for f, _ in cluster)
            kept.append(frames[(len(frames) - 1) // 2])
    kept = sorted(set(kept))

    votes = np.zeros((n + 1, 3), dtype=np.int32)
    for v in range(k):
        votes[1:] += np.cumsum(paths[v][:, None] == np.arange(3), axis=0)
    edges = [0] + kept + [n]
    out = np.empty(n, dtype=np.int8)
    for s, e in zip(edges[:-1], edges[1:]):
        out[s:e] = int(np.argmax(votes[e] - votes[s]))
    return model.merge_short_runs(out, cfg)


def stability(path, paths, tolerance=None):
    """Per segment of `path`: (start, end, label, fraction of paths with a matching segment).

    A path agrees with a segment if it has a segment of the same label whose start and end are both
    within the tolerance (a start at frame 0 / end at the last frame always matches its edge).
    """
    paths = np.asarray(paths)
    n = paths.shape[1]
    tol = int(round(TOLERANCE_S * FPS)) if tolerance is None else tolerance
    runs = [hsmm.path_to_segments(p) for p in paths]
    out = []
    for s, e, label in hsmm.path_to_segments(path):
        agree = 0
        for segs in runs:
            for s2, e2, l2 in segs:
                if l2 == label and abs(s2 - s) <= tol and abs(e2 - e) <= tol:
                    agree += 1
                    break
        out.append((s, e, label, agree / float(len(runs))))
    return out


def decode_ensemble(sig, cfg, return_stability=False):
    """Consensus label path over cfg.ensemble variants (and optionally per-segment stability)."""
    base_emit = model.emissions(sig, cfg)       # penalties/priors change between variants, evidence does not
    paths = np.array([decode_single(sig, c, base_emit) for c in variants(cfg, cfg.ensemble)])
    path = consensus(paths, cfg)
    if return_stability:
        return path, stability(path, paths)
    return path


def decode_full(sig, cfg):
    """Entry point used by model.decode when any of hsmm / refine / ensemble is requested."""
    if cfg.ensemble > 1:
        return decode_ensemble(sig, cfg)
    return decode_single(sig, cfg)

