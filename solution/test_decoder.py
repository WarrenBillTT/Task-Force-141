"""Tests for the upgraded decoder (HSMM, boundary refinement, consensus ensemble). No match data needed.

    python solution/test_decoder.py            # unit tests + invariants
    python solution/test_decoder.py --table    # also the synthetic comparison table (slower)

The comparison table is a regression guard and a sanity check, NOT evidence about the real match,
and nothing may be tuned on it.
"""

import dataclasses
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (HERE, ROOT, os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import hsmm  # noqa: E402
import model  # noqa: E402


def _tiny_cfg(**kw):
    """Tiny durations so that a 7-8 frame stretch exercises priors, the cap and chaining."""
    base = dict(alive_median_s=0.12, alive_sigma=0.8, dead_median_s=0.2, dead_sigma=0.6,
                hsmm_max_seconds=0.2, hsmm_step=1, hsmm_penalty_scale=0.5, switch_penalty=2.0,
                dead_penalty=1.5)
    base.update(kw)
    return dataclasses.replace(model.Config(), **base)


# ---------------------------------------------------------------- E1: HSMM


def test_hsmm_matches_brute_force():
    """The vectorised DP (step 1) reaches the same optimum score as exhaustive enumeration."""
    for seed in range(12):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(5, 9))
        emit = rng.normal(0, 1.5, size=(n, 3))
        emit[rng.random(n) < 0.3] = 0.0           # frames without ball evidence
        cfg = _tiny_cfg(dur_scale=float(rng.choice([0.5, 1.0, 2.0])),
                        hsmm_max_seconds=float(rng.choice([0.12, 0.2, 5.0])))
        want, _ = hsmm.brute_force(emit, cfg)
        segs = hsmm.decode_segments(emit, cfg, step=1)
        got = hsmm.segmentation_score(emit, segs, cfg)
        assert abs(got - want) < 1e-9, (seed, got, want)
        assert segs[0][0] == 0 and segs[-1][1] == n
        assert all(a[1] == b[0] for a, b in zip(segs, segs[1:]))


def test_hsmm_persists_through_missing_evidence():
    cfg = dataclasses.replace(model.Config(), hsmm_step=5)
    emit = np.zeros((600, 3))
    emit[:200, model.H] = 0.5
    emit[400:, model.H] = 0.5
    emit[200:400] = 0.0                            # ball missing for 8 s inside a home possession
    path = hsmm.decode(emit, cfg)
    assert (path == model.H).all(), "state must persist through a gap with no evidence"


def test_hsmm_duration_prior_keeps_long_possession_whole():
    """A 0.3 s burst of away evidence in a long home possession is not allowed to split it."""
    cfg = dataclasses.replace(model.Config(), hsmm_step=5)
    emit = np.zeros((1500, 3))
    emit[:, model.H] = 0.3
    emit[:, model.A] = -0.3
    emit[700:708, model.H], emit[700:708, model.A] = -0.5, 0.5
    path = hsmm.decode(emit, cfg)
    assert (path == model.H).all()


def test_hsmm_coarse_close_to_exact():
    """Coarse grid + polish gives (almost) the same segmentation as step 1 on a clean signal."""
    rng = np.random.default_rng(3)
    emit = rng.normal(0, 0.3, size=(1200, 3))
    for s, e, l in ((0, 300, 0), (300, 520, 2), (520, 900, 1), (900, 1200, 0)):
        emit[s:e, l] += 0.6
    cfg = dataclasses.replace(model.Config(), hsmm_step=5)
    a = hsmm.decode(emit, cfg, step=1)
    b = hsmm.decode(emit, cfg, step=5)
    assert (a != b).mean() < 0.02


def run_all():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print("ok  {}".format(name))
    print("{} tests passed".format(len(tests)))


if __name__ == "__main__":
    run_all()
