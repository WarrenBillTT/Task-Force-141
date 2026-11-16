"""Unit tests, invariants, synthetic benchmark, stability, and timing for Approach A (touch.py).

    python -I solution/test_touch.py

Tests run purely on hand-made synthetic scenes and `simulate.py` (no real match data).
"""

import os
import site
import sys
import time

_user_site = site.getusersitepackages()
if _user_site and os.path.isdir(_user_site) and _user_site not in sys.path:
    sys.path.append(_user_site)

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (HERE, ROOT, os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import model  # noqa: E402
import possession_metrics as pm  # noqa: E402
import run  # noqa: E402
import signals  # noqa: E402
import simulate  # noqa: E402
import touch  # noqa: E402

PITCH = (105.3, 68.0)


def _blank_scene(n_frames=140):
    """4 track slots (slots 0,1 = Home, slots 2,3 = Away) parked far from midfield."""
    team = np.array([0, 0, 1, 1], dtype=np.int8)
    players = np.zeros((n_frames, 4, 2), dtype=np.float32)
    players[:, 0] = [5.0, 5.0]
    players[:, 1] = [10.0, 5.0]
    players[:, 2] = [95.0, 62.0]
    players[:, 3] = [90.0, 62.0]
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    return ball, players, team


def _eval_scene(ball_raw, players, team, pitch=PITCH):
    ball, observed, filled = signals.clean_ball(ball_raw)
    dh, da = signals.team_distances(ball, players, team)
    sig = {
        "dist_home": dh,
        "dist_away": da,
        "speed": signals.ball_speed(ball),
        "ball_ok": observed | filled,
        "oob": signals.out_of_play(ball, pitch),
        "near_line": signals.near_boundary(ball, pitch),
        "set_piece": np.zeros(len(ball), dtype=bool),
    }
    _, details = touch.build_touch_signals(
        ball, players, team, sig, observed=observed, filled=filled, return_details=True
    )
    return sig, details


# ---------------------------------------------------------------- 8 hand-made scenes


def test_pass_between_teammates():
    """Scene 1: Home holds, passes through open pitch to Home teammate; no switch."""
    n = 120
    ball, players, team = _blank_scene(n)
    players[:, 0] = [20.0, 34.0]
    players[:, 1] = [45.0, 34.0]
    ball[:31] = [20.1, 34.0]
    for t in range(31, 56):
        alpha = (t - 30) / 25.0
        ball[t] = [20.1 + 24.8 * alpha, 34.0]
    ball[56:] = [44.9, 34.0]

    sig, details = _eval_scene(ball, players, team)
    assert np.any(sig["touch"][28:33] > 0), "kick touch should be detected"
    assert np.all(sig["touch_team"][sig["touch"] > 0] == touch.HOME), "all touches belong to Home"
    assert np.all(sig["ctrl_logodds"][15:] > 0), "Home retains control through pass and reception"
    path = model.decode(sig, model.Config(use_touch=1))
    assert np.all(path == model.H), "decoded labels stay H with no switch"


def test_interception_followed_by_control():
    """Scene 2: Home kicks, Away intercepts and dribbles; switch happens AFTER control, not at 1st contact."""
    n = 120
    ball, players, team = _blank_scene(n)
    players[:, 0] = [20.0, 34.0]
    players[0:45, 2] = [45.0, 34.0]
    for t in range(45, n):
        players[t, 2] = [45.0 + 0.15 * (t - 45), 34.0]

    ball[:26] = [20.1, 34.0]
    for t in range(26, 46):
        alpha = (t - 25) / 20.0
        ball[t] = [20.1 + 24.9 * alpha, 34.0]
    for t in range(46, n):
        ball[t] = players[t, 2] + [0.2, 0.0]

    sig, details = _eval_scene(ball, players, team)
    away_eps = [ep for ep in details["episodes"] if ep["team"] == touch.AWAY]
    assert len(away_eps) >= 1, "Away carry episode should be detected after interception"
    assert np.any(sig["touch_team"][43:48] == touch.AWAY), "intercept contact touch attributed to Away"
    assert np.all(sig["ctrl_logodds"][15:50] > 0), "must NOT switch at first contact (frames 45..49)"
    assert np.all(sig["ctrl_logodds"][62:] < 0), "must switch to Away once ~0.5 s carry control is met"
    path = model.decode(sig, model.Config(use_touch=1))
    assert path[20] == model.H and path[-1] == model.A, "possession transitions cleanly H -> A"


def test_deflection_off_opponent():
    """Scene 3: Home passes, ball deflects off Away player into open space; NO possession switch."""
    n = 110
    ball, players, team = _blank_scene(n)
    players[:, 0] = [20.0, 34.0]
    players[:, 2] = [45.0, 34.0]
    ball[:26] = [20.1, 34.0]
    for t in range(26, 46):
        ball[t] = [20.1 + (t - 25) * 1.245, 34.0]
    for t in range(46, n):
        ball[t] = [45.0 + (t - 45) * 0.5, 34.0 + (t - 45) * 0.4]  # fast deflection inbounds away from Away player

    sig, details = _eval_scene(ball, players, team)
    assert np.any(sig["touch_team"][43:48] == touch.AWAY), "deflection touch is detected for Away"
    assert not any(ep["team"] == touch.AWAY for ep in details["episodes"]), "deflection is NOT a control episode"
    assert np.all(sig["ctrl_logodds"][15:90] > 0), "lone opposing deflection must NOT switch control"
    path = model.decode(sig, model.Config(use_touch=1))
    assert model.A not in path, "decoded labels must not switch to A on a deflection"


def test_dribble_no_flicker():
    """Scene 4: Home player dribbles with mild tracking jitter while an opponent crosses near 2.3 m."""
    rng = np.random.default_rng(7)
    n = 130
    ball, players, team = _blank_scene(n)
    for t in range(n):
        players[t, 0] = [20.0 + 0.20 * t, 34.0] + rng.normal(0, 0.08, 2)
        ball[t] = [20.0 + 0.20 * t + 0.3, 34.0] + rng.normal(0, 0.12, 2)
    # Opponent briefly approaches to 2.3 m around t=60
    players[:, 2] = [32.0, 36.3]

    sig, details = _eval_scene(ball, players, team)
    assert any(ep["team"] == touch.HOME and ep["kind"] == "carry" for ep in details["episodes"])
    assert np.all(sig["ctrl_logodds"][15:] > 0), "continuous Home control during dribble"
    path = model.decode(sig, model.Config(use_touch=1))
    assert np.all(path == model.H), "zero flicker during dribble"


def test_long_ball_in_flight():
    """Scene 5: Home boots a 3.0 s long ball passing near an Away defender; Home retained."""
    n = 140
    ball, players, team = _blank_scene(n)
    players[:, 0] = [15.0, 34.0]
    players[:, 1] = [90.0, 34.0]
    players[:, 2] = [52.0, 37.0]  # 3.0 m off the ball flight line, well outside contact radius
    ball[:31] = [15.1, 34.0]
    for t in range(31, 106):
        alpha = (t - 30) / 75.0
        ball[t] = [15.1 + 74.8 * alpha, 34.0]
    ball[106:] = [89.9, 34.0]

    sig, _ = _eval_scene(ball, players, team)
    assert np.all(sig["ctrl_logodds"][15:] > 0), "Home retained through 3.0 s flight past defender"
    emit_nt = model.emissions(sig, model.Config(use_touch=0))
    assert emit_nt[68, model.A] > emit_nt[68, model.H], "sanity: old proximity model flips Away mid-flight"
    path = model.decode(sig, model.Config(use_touch=1))
    assert np.all(path == model.H), "touch model keeps H throughout long flight"


def test_ball_missing_mid_flight():
    """Scene 6: Short gap (filled) creates zero touch evidence; long gap (>1.5 s) outputs 0 logodds then resumes."""
    # Subcase A: 20-frame filled gap mid-flight
    n = 130
    ball, players, team = _blank_scene(n)
    players[:, 0] = [15.0, 34.0]
    players[:, 1] = [75.0, 34.0]
    ball[:31] = [15.1, 34.0]
    for t in range(31, 91):
        ball[t] = [15.1 + (t - 30) * 1.0, 34.0]
    ball[91:] = [74.9, 34.0]
    ball[42:62] = np.nan  # 20 frames <= 37 (filled by clean_ball)

    sig_a, _ = _eval_scene(ball, players, team)
    assert np.all(sig_a["touch"][42:62] == 0.0), "NEVER create touch evidence inside filled stretch"
    assert np.all(sig_a["touch_team"][42:62] == touch.NONE)
    assert np.all(sig_a["ctrl_logodds"][42:62] > 0), "Home retained across interpolated flight"

    # Subcase B: 55-frame un-filled missing gap (> 37 frames)
    ball_b = ball.copy()
    ball_b[36:91] = np.nan
    for t in range(91, 105):
        ball_b[t] = [75.0 + (t - 91) * 0.6, 34.0]
    ball_b[105:] = [83.4, 34.0]
    players[:, 1] = [83.4, 34.0]

    sig_b, _ = _eval_scene(ball_b, players, team)
    assert np.all(sig_b["touch"][36:91] == 0.0)
    assert np.all(sig_b["ctrl_logodds"][36:91] == 0.0), "where ball is missing (>1.5s), evidence is 0"
    emit_b = model.emissions(sig_b, model.Config(use_touch=1))
    assert np.all(emit_b[36:91] == 0.0), "NaN ball -> all zero emissions"
    assert sig_b["ctrl_logodds"][92] > 0, "resumes Home once ball reappears mid-flight"


def test_track_identity_swap_near_ball():
    """Scene 7: Same-team track slot switches mid-dribble; carry episode survives across swaps."""
    n = 100
    ball, players, team = _blank_scene(n)
    for t in range(n):
        pos = np.array([20.0 + 0.16 * t, 34.0], dtype=np.float32)
        ball[t] = pos + [0.25, 0.0]
        # Swap carrier between Home slot 0 and Home slot 1 every 10 frames (< 13 frames alone)
        active_slot = 0 if (t // 10) % 2 == 0 else 1
        players[t, active_slot] = pos

    sig, details = _eval_scene(ball, players, team)
    home_carries = [ep for ep in details["episodes"] if ep["team"] == touch.HOME and ep["kind"] == "carry"]
    assert len(home_carries) >= 1 and (home_carries[0]["end"] - home_carries[0]["start"]) >= 40, (
        "spatially continuous same-team track swaps must link into a continuous carry episode"
    )
    assert np.all(sig["ctrl_logodds"][15:] > 0)
    assert np.all(model.decode(sig, model.Config(use_touch=1)) == model.H)


def test_aerial_duel_contested():
    """Scene 8: Two opposing players contest an aerial ball; Home retains last control."""
    n = 110
    ball, players, team = _blank_scene(n)
    players[:, 0] = [20.0, 34.0]
    players[:, 1] = [50.0, 34.30]  # 0.30 m from ball
    players[:, 2] = [50.0, 33.85]  # 0.15 m from ball -> margin = 0.15 m < contest_margin (0.50 m)

    ball[:31] = [20.1, 34.0]
    for t in range(31, 51):
        ball[t] = [20.1 + (t - 30) * 1.495, 34.0]
    # Ball hovers/bounces between the two jumping players for 25 frames, then flies loose
    for t in range(51, 76):
        ball[t] = [50.0, 34.0 + 0.05 * np.sin(0.6 * (t - 51))]
    for t in range(76, n):
        ball[t] = [50.0 + (t - 75) * 0.8, 34.0]

    sig, details = _eval_scene(ball, players, team)
    assert not any(ep["team"] == touch.AWAY for ep in details["episodes"]), (
        "contested aerial duel must NOT create an Away control episode"
    )
    assert np.all(sig["ctrl_logodds"][15:95] > 0), "possession stays with Home during contested duel"
    assert model.A not in model.decode(sig, model.Config(use_touch=1))


# ---------------------------------------------------------------- invariants, benchmark & timing


def test_invariants_and_no_touch():
    sim = simulate.simulate(20000, [(1, 500, 9500), (2, 10500, 19500)], seed=2, pitch=PITCH)
    sig1 = run.build_signals(sim["ball"], sim["players"], sim["team"], PITCH)
    sig2 = run.build_signals(sim["ball"], sim["players"], sim["team"], PITCH)

    for key in ("touch", "touch_team", "ctrl_logodds", "ctrl_conf"):
        assert key in sig1
        assert np.array_equal(sig1[key], sig2[key]), f"{key} must be deterministic"
        assert np.all(np.isfinite(sig1[key])), f"{key} must have no NaN/inf"

    assert sig1["touch"].dtype == np.float32
    assert sig1["touch_team"].dtype == np.int8
    assert sig1["ctrl_logodds"].dtype == np.float32
    assert sig1["ctrl_conf"].dtype == np.float32
    assert np.all((sig1["touch"] >= 0.0) & (sig1["touch"] <= 1.0))
    assert set(np.unique(sig1["touch_team"]).tolist()).issubset({-1, 0, 1})
    assert np.all((sig1["ctrl_conf"] >= 0.0) & (sig1["ctrl_conf"] <= 1.0))
    assert np.all(np.abs(sig1["ctrl_logodds"]) <= touch.TouchConfig().ctrl_cap + 1e-6)

    emit = model.emissions(sig1, model.Config(use_touch=1))
    assert np.all(np.isfinite(emit)), "emissions must be finite"
    assert np.all(emit[~sig1["ball_ok"]] == 0.0), "missing ball (~ball_ok) must emit zeros"

    # --no-touch must match legacy proximity-only emissions and labels byte-for-byte
    legacy_sig = {k: sig1[k] for k in ("dist_home", "dist_away", "speed", "ball_ok", "oob", "near_line", "set_piece")}
    emit_nt = model.emissions(sig1, model.Config(use_touch=0))
    emit_leg = model.emissions(legacy_sig, model.Config(use_touch=0))
    assert np.array_equal(emit_nt, emit_leg), "--no-touch must exactly reproduce legacy emissions"

    blocks = [(1, 500, 9500), (2, 10500, 19500)]
    lab_t = run.predict(sig1, blocks, model.Config(use_touch=1), kickoff2=10800, n_frames=20000)
    lab_nt = run.predict(sig1, blocks, model.Config(use_touch=0), kickoff2=10800, n_frames=20000)
    lab_leg = run.predict(legacy_sig, blocks, model.Config(use_touch=0), kickoff2=10800, n_frames=20000)
    assert np.array_equal(lab_nt, lab_leg), "--no-touch must exactly reproduce legacy labels"
    assert set(np.unique(lab_t[500:9501]).tolist()).issubset({"H", "A", "D"})
    assert np.all(lab_t[10500:10800] == "D"), "kickoff2 handling must remain D"


def test_timing_full_half():
    """Full-match-half input (80,000 frames) must build touch signals well under a few minutes."""
    n = 80000
    blocks = [(1, 0, n - 1)]
    sim = simulate.simulate(n, blocks, seed=10, pitch=PITCH)
    t0 = time.perf_counter()
    sig = run.build_signals(sim["ball"], sim["players"], sim["team"], PITCH)
    dt = time.perf_counter() - t0
    assert "ctrl_logodds" in sig and len(sig["ctrl_logodds"]) == n
    assert dt < 30.0, f"80k frames took too long: {dt:.2f}s"
    print(f"timing (80,000 frames, 1 full half): {dt:.2f} s")


def _match_touches(pred_frames, gt_frames, tol=3):
    """Greedy 1-to-1 matching of predicted touches to ground-truth touches within +/- tol frames."""
    used = set()
    tp = 0
    for p in pred_frames:
        best_i = None
        best_d = tol + 1
        for i, g in enumerate(gt_frames):
            if i not in used:
                d = abs(p - g)
                if d <= tol and d < best_d:
                    best_d = d
                    best_i = i
        if best_i is not None:
            used.add(best_i)
            tp += 1
    prec = tp / max(len(pred_frames), 1)
    rec = tp / max(len(gt_frames), 1)
    return prec, rec


def _score_labels(gt, labels):
    res = pm.evaluate(pm.label_codes(gt), pm.label_codes(labels))
    return res["spell_f1_mean"], res["macro_f1"]


def test_synthetic_benchmark(seeds=(0, 1, 2, 3, 4, 5)):
    """Multi-seed regression guard: touch prec/rec (+/-3 frames) & spellF1 (--touch vs --no-touch)."""
    scenarios = [
        ("standard", {}),
        ("stress40", dict(missing_rate=0.40, jitter_std=0.25, spike_rate=0.005,
                          id_swap_rate=0.01, deflection_rate=0.35)),
    ]
    print("--- synthetic benchmark (seeds={}) ---".format(len(seeds)))
    for name, kw in scenarios:
        precs, recs = [], []
        sf1_t, mf1_t = [], []
        sf1_nt, mf1_nt = [], []
        for s in seeds:
            sim = simulate.simulate(seed=s, pitch=PITCH, **kw)
            _, obs, _ = signals.clean_ball(sim["ball"])
            sig = run.build_signals(sim["ball"], sim["players"], sim["team"], PITCH)

            gt_t = [t for t, _, _, _ in sim["touches"] if np.any(obs[max(0, t - 3):t + 4])]
            pred_t = [int(p) for p in np.flatnonzero(sig["touch"] > 0) if sim["gt"][p] in ("H", "A")]
            p, r = _match_touches(pred_t, gt_t, tol=3)
            precs.append(p)
            recs.append(r)

            lab_t = run.predict(sig, sim["blocks"], model.Config(use_touch=1), None, len(sim["ball"]))
            lab_nt = run.predict(sig, sim["blocks"], model.Config(use_touch=0), None, len(sim["ball"]))
            st, mt = _score_labels(sim["gt"], lab_t)
            snt, mnt = _score_labels(sim["gt"], lab_nt)
            sf1_t.append(st)
            mf1_t.append(mt)
            sf1_nt.append(snt)
            mf1_nt.append(mnt)

        print(
            "{:<9s} | touch P={:.3f}+/-{:.3f} R={:.3f}+/-{:.3f} | "
            "spellF1 touch={:.4f}+/-{:.4f} (macro {:.4f}) | "
            "no-touch={:.4f}+/-{:.4f} (macro {:.4f})".format(
                name,
                np.mean(precs), np.std(precs),
                np.mean(recs), np.std(recs),
                np.mean(sf1_t), np.std(sf1_t), np.mean(mf1_t),
                np.mean(sf1_nt), np.std(sf1_nt), np.mean(mf1_nt),
            )
        )
        assert np.mean(recs) >= 0.70, f"{name}: touch recall unexpectedly low ({np.mean(recs):.3f})"
        assert np.mean(sf1_t) >= np.mean(sf1_nt), (
            f"{name}: --touch ({np.mean(sf1_t):.4f}) regressed vs --no-touch ({np.mean(sf1_nt):.4f})"
        )


def test_perturbation_stability(seeds=(0, 1, 2)):
    """Label-free stability: drop extra ball bursts (~20%) + add 0.15m jitter, measure spellF1 agreement."""
    agreements = []
    macro_agreements = []
    for s in seeds:
        sim = simulate.simulate(seed=s, pitch=PITCH)
        sig_clean = run.build_signals(sim["ball"], sim["players"], sim["team"], PITCH)
        lab_clean = run.predict(sig_clean, sim["blocks"], model.Config(use_touch=1), None, len(sim["ball"]))

        rng = np.random.default_rng(1000 + s)
        ball_p = sim["ball"].copy()
        ball_p = ball_p + rng.normal(0, 0.15, size=ball_p.shape).astype(np.float32)
        graded_idx = np.concatenate([np.arange(f, l + 1) for _, f, l in sim["blocks"]])
        base_obs = float(np.mean(~np.isnan(ball_p[graded_idx, 0])))
        target_obs = max(0.35, base_obs - 0.20)
        tries = 0
        while float(np.mean(~np.isnan(ball_p[graded_idx, 0]))) > target_obs and tries < 1000:
            tries += 1
            start = int(rng.choice(graded_idx))
            gap = int(rng.integers(40, 90)) if rng.random() < 0.25 else int(rng.geometric(1 / 25.0))
            ball_p[start:min(len(ball_p), start + gap)] = np.nan

        sig_pert = run.build_signals(ball_p, sim["players"], sim["team"], PITCH)
        lab_pert = run.predict(sig_pert, sim["blocks"], model.Config(use_touch=1), None, len(sim["ball"]))
        sf1, mf1 = _score_labels(lab_clean, lab_pert)
        agreements.append(sf1)
        macro_agreements.append(mf1)

    print("perturbation stability (3 seeds, -20% ball + 0.15m jitter): spellF1={:.4f}+/-{:.4f} macroF1={:.4f}+/-{:.4f}".format(
        np.mean(agreements), np.std(agreements), np.mean(macro_agreements), np.std(macro_agreements)
    ))
    assert np.mean(agreements) >= 0.80, f"perturbation stability too low: {np.mean(agreements):.4f}"


def main():
    tests = [
        test_pass_between_teammates,
        test_interception_followed_by_control,
        test_deflection_off_opponent,
        test_dribble_no_flicker,
        test_long_ball_in_flight,
        test_ball_missing_mid_flight,
        test_track_identity_swap_near_ball,
        test_aerial_duel_contested,
        test_invariants_and_no_touch,
        test_timing_full_half,
        test_synthetic_benchmark,
        test_perturbation_stability,
    ]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
