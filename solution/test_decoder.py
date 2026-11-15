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


# ---------------------------------------------------------------- synthetic data helpers

PITCH = (105.3, 68.0)
N = 20000
BLOCKS = [(1, 500, 9500), (2, 10500, 19500)]
KICKOFF2 = 10800
LEVELS = {"clean": (0.0, 0.0), "noisy": (0.5, 0.15), "harsh": (1.0, 0.30)}  # (extra ball noise m, extra dropout)


def make_signals(seed, level="clean"):
    """Synthetic match -> (sim, signals). Extra noise / dropout bursts are layered on the ball only."""
    import run
    import simulate
    sim = simulate.simulate(N, BLOCKS, seed=seed, pitch=PITCH)
    noise, dropout = LEVELS[level]
    ball = sim["ball"].copy()
    rng = np.random.default_rng(1000 + seed)
    ball += rng.normal(0, noise, size=ball.shape).astype(np.float32)
    if dropout:
        import stability
        ball[stability.drop_mask(N, dropout, 2000 + seed)] = np.nan
    return sim, run.build_signals(ball, sim["players"], sim["team"], PITCH)


def make_cfg(decoder="viterbi", refine=0, ensemble=1, **kw):
    return dataclasses.replace(model.Config(), decoder=decoder, refine=refine, ensemble=ensemble, **kw)


METHODS = [("viterbi", dict(decoder="viterbi")),
           ("hsmm", dict(decoder="hsmm")),
           ("hsmm+refine", dict(decoder="hsmm", refine=1)),
           ("hsmm+refine+ens5", dict(decoder="hsmm", refine=1, ensemble=5))]


def spell_f1(sim, labels):
    import test_pipeline
    gt = sim["gt"].copy()
    return test_pipeline.score(gt, labels)[0]


# ---------------------------------------------------------------- E2: signals + refinement


def _flat_sig(n, near_home_until=None, slow_from=None):
    sig = dict(dist_home=np.full(n, 1.0, np.float32), dist_away=np.full(n, 8.0, np.float32),
               speed=np.full(n, 6.0, np.float32), ball_dv=np.zeros(n, np.float32),
               ball_ok=np.ones(n, bool), oob=np.zeros(n, bool), near_line=np.zeros(n, bool),
               set_piece=np.zeros(n, bool))
    if near_home_until is not None:
        sig["dist_home"][near_home_until:], sig["dist_away"][near_home_until:] = 8.0, 1.0
    if slow_from is not None:
        sig["speed"][slow_from:] = 0.2
    return sig


def _two_run_path(n, cut, first, second):
    path = np.full(n, first, dtype=np.int8)
    path[cut:] = second
    return path


def test_ball_velocity_change():
    import signals
    t = np.arange(200)[:, None] / 25.0
    straight = np.hstack([10.0 * t, 0 * t]).astype(np.float32)           # 10 m/s along x
    dv = signals.ball_velocity_change(straight)
    assert np.isnan(dv[:3]).all() and np.isnan(dv[-3:]).all()
    assert np.nanmax(dv) < 1e-3
    kinked = straight.copy()
    kinked[100:, 0] = kinked[100, 0] - 10.0 * (t[100:, 0] - t[100, 0])      # reverses at frame 100
    dv = signals.ball_velocity_change(kinked)
    assert abs(np.nanmax(dv) - 20.0) < 1.0 and 97 <= int(np.nanargmax(dv)) <= 103
    gap = straight.copy()
    gap[50:60] = np.nan
    assert np.isnan(signals.ball_velocity_change(gap)[48:63]).any()


def test_refine_snaps_team_change():
    import refine
    cfg = make_cfg(refine=1)
    sig = _flat_sig(2000, near_home_until=1000)
    out = refine.refine(_two_run_path(2000, 1030, model.H, model.A), sig, cfg)
    b = int(np.flatnonzero(np.diff(out))[0]) + 1
    assert 996 <= b <= 1004, b


def test_refine_alive_to_dead_and_back():
    import refine
    cfg = make_cfg(refine=1)
    sig = _flat_sig(2000)
    sig["oob"][1000:] = True                                   # ball leaves the pitch at 1000
    out = refine.refine(_two_run_path(2000, 1040, model.H, model.D), sig, cfg)
    assert 996 <= int(np.flatnonzero(np.diff(out))[0]) + 1 <= 1004
    sig = _flat_sig(2000, slow_from=1000)                      # ball merely stops at 1000 (weak cue)
    out = refine.refine(_two_run_path(2000, 1010, model.H, model.D), sig, cfg)
    assert int(np.flatnonzero(np.diff(out))[0]) + 1 >= 1000    # never before the last clearly live frame
    sig = _flat_sig(2000)
    sig["speed"][:1000] = 0.2                                  # ball starts moving at 1000
    sig["ball_dv"][998:1003] = 10.0                            # the kick
    out = refine.refine(_two_run_path(2000, 960, model.D, model.A), sig, cfg)
    assert 993 <= int(np.flatnonzero(np.diff(out))[0]) + 1 <= 1004


def test_refine_is_bounded_by_window():
    import refine
    cfg = make_cfg(refine=1)
    sig = _flat_sig(2000, near_home_until=1000)
    out = refine.refine(_two_run_path(2000, 1100, model.H, model.A), sig, cfg)   # cue is 100 frames away
    b = int(np.flatnonzero(np.diff(out))[0]) + 1
    assert b == 1100 - int(cfg.refine_window_s * 25), b


def test_refine_touch_optional_and_used():
    import refine
    cfg = make_cfg(refine=1)
    sig = _flat_sig(2000)
    sig["dist_home"][:], sig["dist_away"][:] = 20.0, 20.0                  # no team cue at all
    path = _two_run_path(2000, 1020, model.H, model.A)
    plain = refine.refine(path, sig, cfg)
    assert int(np.flatnonzero(np.diff(plain))[0]) + 1 == 1020                # nothing to go on: stays put
    sig["touch"] = np.zeros(2000, np.float32)
    sig["touch"][1005] = 1.0
    out = refine.refine(path, sig, cfg)
    assert 1002 <= int(np.flatnonzero(np.diff(out))[0]) + 1 <= 1008


def test_refine_keeps_labels_count_and_minimum():
    import refine
    for seed in (0, 1):
        _, sig = make_signals(seed, "noisy")
        cfg = make_cfg(decoder="hsmm", refine=1)
        part = {k: v[500:9501] for k, v in sig.items()}
        base = model.merge_short_runs(hsmm.decode(model.emissions(part, cfg), cfg), cfg)
        out = refine.refine(base, part, cfg)
        a, b = hsmm.path_to_segments(base), hsmm.path_to_segments(out)
        assert [s[2] for s in a] == [s[2] for s in b], "labels / segment order changed"
        for (s, e, l), (s0, e0, _) in zip(b, a):
            limit = cfg.min_dead if l == model.D else cfg.min_alive
            assert e - s >= min(limit, e0 - s0), "refinement shrank a segment below the minimum"
            assert abs(s - s0) <= int(cfg.refine_window_s * 25)


# ---------------------------------------------------------------- E3: ensemble


def test_ensemble_variants_deterministic():
    import ensemble
    cfg = make_cfg(decoder="hsmm")
    a, b = ensemble.variants(cfg, 7), ensemble.variants(cfg, 7)
    assert a == b
    assert a[0].switch_penalty == cfg.switch_penalty and a[0].dur_scale == cfg.dur_scale
    assert len({(c.switch_penalty, c.dur_scale) for c in ensemble.variants(cfg, 25)}) == 25
    assert ensemble.variants(cfg, 3) == a[:3]


def test_consensus_keeps_supported_boundaries():
    import ensemble
    cfg = make_cfg()
    n = 4000
    paths = []
    for k in range(5):
        p = np.full(n, model.H, dtype=np.int8)
        p[1000 + 8 * k:] = model.A                    # all five agree within 1 s
        if k < 2:
            p[3000:3100] = model.D                    # only 2 of 5 see a stoppage
        paths.append(p)
    out = ensemble.consensus(np.array(paths), cfg)
    segs = hsmm.path_to_segments(out)
    assert [s[2] for s in segs] == [model.H, model.A], segs
    assert 1000 <= segs[0][1] <= 1032
    # at least half is enough to keep a boundary: 3 of 5 see the stoppage
    for p in paths[2:3]:
        p[3000:3100] = model.D
    out3 = ensemble.consensus(np.array(paths), cfg)
    assert model.D in set(out3.tolist())


def test_stability_scores():
    import ensemble
    cfg = make_cfg()
    p = np.full(3000, model.H, dtype=np.int8)
    p[1500:] = model.A
    same = ensemble.stability(p, np.array([p, p, p]))
    assert all(abs(s[3] - 1.0) < 1e-9 for s in same)
    other = p.copy()
    other[1500:] = model.H                            # one variant never sees the change
    mixed = ensemble.stability(p, np.array([p, p, other]))
    assert all(0.0 <= s[3] <= 1.0 for s in mixed) and abs(mixed[1][3] - 2 / 3) < 1e-9


def test_ensemble_deterministic_on_synthetic():
    import ensemble
    _, sig = make_signals(2, "noisy")
    cfg = make_cfg(decoder="hsmm", refine=1, ensemble=3)
    part = {k: v[500:4501] for k, v in sig.items()}
    a = ensemble.decode_ensemble(part, cfg)
    b, stab = ensemble.decode_ensemble(part, cfg, return_stability=True)
    assert a.tobytes() == b.tobytes()
    assert all(0.0 <= s[3] <= 1.0 for s in stab) and len(stab) == len(hsmm.path_to_segments(a))


# ---------------------------------------------------------------- E4: invariants through run.predict


def _label_runs(labels):
    runs, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            runs.append((start, i, labels[start]))
            start = i
    return runs


def test_invariants_all_decoders():
    import pandas as pd
    import possession_metrics as pm
    import run
    _, sig = make_signals(3, "noisy")
    for name, kw in METHODS:
        cfg = make_cfg(**kw)
        a = run.predict(sig, BLOCKS, cfg, KICKOFF2, N)
        b = run.predict(sig, BLOCKS, cfg, KICKOFF2, N)
        assert a.tobytes() == b.tobytes(), name + ": not deterministic"
        assert set(np.unique(a)) <= {"H", "A", "D", "X"}
        for _, first, last in BLOCKS:
            assert not (a[first:last + 1] == "X").any(), name + ": abstained inside a graded half"
        assert (a[10500:KICKOFF2] == "D").all(), name + ": kickoff2 handling"
        assert (a[:500] == "X").all() and (a[9501:10500] == "X").all()
        for s, e, l in _label_runs(a):
            if l in "HA" and s > 0 and e < N and a[s - 1] != "X" and a[e] != "X":
                assert e - s >= cfg.min_alive, (name, s, e)
            if l == "D" and s > KICKOFF2 and e < N and a[s - 1] != "X" and a[e] != "X":
                assert e - s >= cfg.min_dead, (name, s, e)
        table = pd.DataFrame({"frame": np.arange(N), "label": a})
        assert pm.validate_submission(table, N) is None, name


def test_run_cli_flags_parse():
    import subprocess
    out = subprocess.run([sys.executable, os.path.join(HERE, "run.py"), "--help"],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0
    for flag in ("--decoder", "--refine", "--ensemble", "--switch-penalty", "--hsmm-step"):
        assert flag in out.stdout, flag


# ---------------------------------------------------------------- label-free stability


def test_stability_drop_ball():
    import stability
    m1, m2 = stability.drop_mask(5000, 0.25, 0), stability.drop_mask(5000, 0.25, 0)
    assert (m1 == m2).all() and m1.mean() >= 0.25
    assert (stability.drop_mask(5000, 0.25, 1) != m1).any()
    _, sig = make_signals(1)
    out = stability.drop_ball(sig, 0.25, 0)
    gone = ~out["ball_ok"] & sig["ball_ok"]
    assert gone.any() and np.isinf(out["dist_home"][gone]).all() and np.isnan(out["speed"][gone]).all()
    assert (out["set_piece"] == sig["set_piece"]).all()
    assert sig["ball_ok"].sum() > out["ball_ok"].sum() and np.isfinite(sig["dist_home"]).any()  # input untouched


def test_stability_report_runs():
    import stability
    _, sig = make_signals(1, "noisy")
    for kw in (dict(decoder="viterbi"), dict(decoder="hsmm")):
        rep = stability.stability_report(sig, BLOCKS, make_cfg(**kw), KICKOFF2, seeds=(0, 1))
        assert 0.0 <= rep["spell_f1"] <= 1.0 and 0.0 < rep["count_ratio"] <= 1.0


# ---------------------------------------------------------------- synthetic comparison (regression guard)


def comparison(seeds=(1, 2, 3, 4, 5), levels=("clean", "noisy", "harsh"), methods=METHODS):
    """{level: {method: [spellF1 per seed]}}. Sanity / regression guard only: never tune on this."""
    import run
    table = {lv: {name: [] for name, _ in methods} for lv in levels}
    for lv in levels:
        for seed in seeds:
            sim, sig = make_signals(seed, lv)
            for name, kw in methods:
                labels = run.predict(sig, BLOCKS, make_cfg(**kw), None, N)
                table[lv][name].append(spell_f1(sim, labels))
    return table


def print_table(table):
    print("{:<8} {:<20} {:>8} {:>8} {:>8} {:>8}".format("level", "method", "mean", "std", "min", "max"))
    for lv, rows in table.items():
        for name, vals in rows.items():
            v = np.array(vals)
            print("{:<8} {:<20} {:8.4f} {:8.4f} {:8.4f} {:8.4f}".format(lv, name, v.mean(), v.std(), v.min(), v.max()))


def test_new_decoders_not_worse_on_synthetic():
    """Regression guard: the new path is not materially worse than Viterbi on the synthetic match."""
    table = comparison(seeds=(1, 2), levels=("clean", "harsh"),
                       methods=[METHODS[0], METHODS[1], METHODS[2]])
    for lv, rows in table.items():
        base = np.mean(rows["viterbi"])
        for name in ("hsmm", "hsmm+refine"):
            assert np.mean(rows[name]) >= base - 0.05, (lv, name, np.mean(rows[name]), base)


def test_hsmm_full_match_timing():
    """About 80,000 frames per half decodes in seconds (not minutes)."""
    import time
    rng = np.random.default_rng(0)
    emit = rng.normal(0, 0.5, size=(80000, 3)).astype(np.float32)
    cfg = make_cfg(decoder="hsmm")
    t0 = time.time()
    hsmm.decode(emit, cfg)
    elapsed = time.time() - t0
    print("   hsmm 80,000 frames: {:.1f} s".format(elapsed))
    assert elapsed < 120


def run_all():
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        fn()
        print("ok  {}".format(name))
    print("{} tests passed".format(len(tests)))
    if "--table" in sys.argv:
        print_table(comparison())


if __name__ == "__main__":
    run_all()
