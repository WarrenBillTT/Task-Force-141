"""Label-free stability check: does the decoder give the same possessions when more ball frames vanish?

    python solution/stability.py                                  # synthetic match, needs no data
    python solution/stability.py --decoder hsmm --refine          # compare decoders by stability
    python solution/stability.py --real --decoder hsmm            # real tracking: run it locally only

No labels are involved. We decode the signals, drop additional ball frames (seeded, in long bursts as
well as short flickers, to mimic ~40% missing ball on top of what is already missing), decode again, and
compare the two outputs with the competition's own spellF1 (one output playing "truth") and with the
number of possessions. A decoder that is right for the right reasons should be stable under this
perturbation; one that flickers or merges on thin evidence is not. Stability is necessary, not
sufficient: a decoder that always says "H" is perfectly stable.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (HERE, ROOT, os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import model  # noqa: E402
import possession_metrics as pm  # noqa: E402

FPS = 25
BALL_KEYS = ("dist_home", "dist_away", "speed", "ball_dv", "ball_ok", "oob", "near_line", "touch")


def drop_mask(n, extra, seed, burst_seconds=3.0, flicker_seconds=0.2, burst_share=0.7):
    """Boolean mask of frames to remove: bursts (mean `burst_seconds`) and flickers, ~`extra` of n."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    target = int(extra * n)
    while mask.sum() < target:
        mean = burst_seconds if rng.random() < burst_share else flicker_seconds
        length = int(rng.geometric(1.0 / max(mean * FPS, 1.0)))
        start = int(rng.integers(0, n))
        mask[start:start + length] = True
    return mask


def drop_ball(sig, extra=0.25, seed=0):
    """Copy of `sig` with extra ball frames made unobserved (same encoding as a real tracking gap)."""
    mask = drop_mask(len(sig["speed"]), extra, seed)
    out = {k: v.copy() for k, v in sig.items()}
    out["dist_home"][mask] = np.inf
    out["dist_away"][mask] = np.inf
    out["speed"][mask] = np.nan
    if "ball_dv" in out:
        out["ball_dv"][mask] = np.nan
    out["ball_ok"][mask] = False
    out["oob"][mask] = False
    out["near_line"][mask] = False
    if "touch" in out:
        out["touch"][mask] = 0.0
    return out


def agreement(a, b):
    """spellF1 of output b scored against output a, and the ratio of possession counts (<= 1)."""
    result = pm.evaluate(pm.label_codes(a), pm.label_codes(b), bootstrap_n=0)
    n_a, n_b = result["n_gt_spells"], result["n_pred_spells"]
    ratio = min(n_a, n_b) / float(max(n_a, n_b)) if max(n_a, n_b) else 1.0
    return result["spell_f1_mean"], ratio, n_a, n_b


def stability_report(sig, blocks, cfg, kickoff2=None, seeds=(0, 1, 2), extra=0.25):
    """Mean spellF1 / run-count agreement between the clean decode and decodes with dropped ball frames."""
    import run
    base = run.predict(sig, blocks, cfg, kickoff2)
    rows = [agreement(base, run.predict(drop_ball(sig, extra, s), blocks, cfg, kickoff2)) for s in seeds]
    rows = np.array([r[:2] for r in rows])
    return {"spell_f1": float(rows[:, 0].mean()), "spell_f1_min": float(rows[:, 0].min()),
            "count_ratio": float(rows[:, 1].mean()), "n_base_spells": int(agreement(base, base)[2])}


def _config(args):
    cfg = model.Config()
    cfg.decoder, cfg.refine, cfg.ensemble = args.decoder, int(args.refine), args.ensemble
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--decoder", choices=("viterbi", "hsmm"), default="viterbi")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--ensemble", type=int, default=1)
    parser.add_argument("--extra", type=float, default=0.25, help="extra fraction of frames to drop")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--real", action="store_true", help="use the real tracking bundle (run locally)")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    cfg = _config(args)
    seeds = tuple(range(args.seeds))

    if args.real:
        import atd
        import run
        from games import GAME, GAMES, half_video_blocks, require, source_frames
        meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, args.data_dir)
        n_frames = source_frames("tactical", GAME)
        pitch = GAMES[GAME]["pitch"]
        ball, players, team = atd.load_atd(raw_path, meta_path, n_frames, pitch)
        sig = run.build_signals(ball, players, team, pitch, run.set_piece_intervals(args.data_dir))
        blocks, kickoff2 = half_video_blocks(GAME), run.SECOND_HALF_KICKOFF
    else:
        import run
        import simulate
        pitch = (105.3, 68.0)
        blocks = [(1, 500, 9500), (2, 10500, 19500)]
        sim = simulate.simulate(20000, blocks, seed=1, pitch=pitch)
        sig = run.build_signals(sim["ball"], sim["players"], sim["team"], pitch)
        kickoff2 = 10800

    report = stability_report(sig, blocks, cfg, kickoff2, seeds, args.extra)
    print("decoder={} refine={} ensemble={}  extra drop={:.0%}".format(
        cfg.decoder, bool(cfg.refine), cfg.ensemble, args.extra))
    print("spellF1 agreement mean {spell_f1:.4f} (worst {spell_f1_min:.4f})  "
          "possession-count ratio {count_ratio:.3f}  (base has {n_base_spells} spells)".format(**report))


if __name__ == "__main__":
    main()
