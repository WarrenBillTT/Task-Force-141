"""Aggregate diagnostics for Approach A (touch & control signals).

Human usage on local match data:
    python solution/touch_report.py
    python solution/touch_report.py --data-dir /path/to/bundle

Synthetic check (no match data needed):
    python -I solution/touch_report.py --synthetic

Prints only aggregate statistics (no frame-level dumps or raw coordinates).
"""

import argparse
import os
import site
import sys

_user_site = site.getusersitepackages()
if _user_site and os.path.isdir(_user_site) and _user_site not in sys.path:
    sys.path.append(_user_site)

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (HERE, ROOT, os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import atd  # noqa: E402
import run  # noqa: E402
import signals  # noqa: E402
import simulate  # noqa: E402
import touch  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

FPS = float(touch.FPS)


def summarize_signals(ball_raw, players, team, pitch, blocks, set_piece_intervals=(), cfg=None):
    """Compute aggregate touch and control diagnostics over graded blocks."""
    cfg = cfg or touch.TouchConfig()
    ball, observed, filled = signals.clean_ball(ball_raw)
    dh, da = signals.team_distances(ball, players, team)
    sig = {
        "dist_home": dh,
        "dist_away": da,
        "speed": signals.ball_speed(ball),
        "ball_ok": observed | filled,
        "oob": signals.out_of_play(ball, pitch),
        "near_line": signals.near_boundary(ball, pitch),
        "set_piece": signals.window_mask(len(ball), set_piece_intervals),
    }
    _, details = touch.build_touch_signals(
        ball, players, team, sig, observed=observed, filled=filled, cfg=cfg, return_details=True
    )

    n = len(ball)
    episodes = details["episodes"]
    in_ep = np.zeros(n, dtype=bool)
    for ep in episodes:
        in_ep[ep["start"]:ep["end"]] = True

    graded_mask = np.zeros(n, dtype=bool)
    for _, first, last in blocks:
        graded_mask[first:last + 1] = True

    touch_mask = (sig["touch"] > 0) & graded_mask
    touch_team = sig["touch_team"]
    nearest = np.minimum(dh, da)
    touch_dists = nearest[touch_mask & np.isfinite(nearest)]

    graded_frames = int(graded_mask.sum())
    graded_mins = max(graded_frames / (FPS * 60.0), 1e-6)
    ctrl_nonzero = np.abs(sig["ctrl_logodds"]) > 1e-6

    in_ctrl_n = int((graded_mask & in_ep & ctrl_nonzero).sum())
    in_flight_n = int((graded_mask & ~in_ep & sig["ball_ok"] & ctrl_nonzero).sum())
    unknown_n = graded_frames - in_ctrl_n - in_flight_n

    dist_stats = {}
    if len(touch_dists) > 0:
        qs = np.percentile(touch_dists, [10, 25, 50, 75, 90])
        dist_stats = {
            "count": int(len(touch_dists)),
            "mean": float(np.mean(touch_dists)),
            "p10": float(qs[0]),
            "p25": float(qs[1]),
            "p50": float(qs[2]),
            "p75": float(qs[3]),
            "p90": float(qs[4]),
        }

    halves = []
    for half, first, last in blocks:
        h_mask = np.zeros(n, dtype=bool)
        h_mask[first:last + 1] = True
        h_mins = max((last + 1 - first) / (FPS * 60.0), 1e-6)
        h_touches = touch_mask & h_mask
        h_eps = [ep for ep in episodes if ep["start"] >= first and ep["end"] <= last + 1]
        durs_s = np.array([(ep["end"] - ep["start"]) / FPS for ep in h_eps], dtype=np.float64)
        halves.append({
            "half": int(half),
            "minutes": float(h_mins),
            "touches": int(h_touches.sum()),
            "touches_per_min": float(h_touches.sum() / h_mins),
            "touches_home": int((h_touches & (touch_team == touch.HOME)).sum()),
            "touches_away": int((h_touches & (touch_team == touch.AWAY)).sum()),
            "episodes": len(h_eps),
            "episodes_home": sum(1 for ep in h_eps if ep["team"] == touch.HOME),
            "episodes_away": sum(1 for ep in h_eps if ep["team"] == touch.AWAY),
            "episodes_carry": sum(1 for ep in h_eps if ep["kind"] == "carry"),
            "episodes_pass": sum(1 for ep in h_eps if ep["kind"] == "pass"),
            "mean_ep_s": float(durs_s.mean()) if len(durs_s) else 0.0,
            "median_ep_s": float(np.median(durs_s)) if len(durs_s) else 0.0,
        })

    all_durs_s = np.array(
        [(ep["end"] - ep["start"]) / FPS for ep in episodes
         if any(first <= ep["start"] and ep["end"] <= last + 1 for _, first, last in blocks)],
        dtype=np.float64,
    )

    return {
        "graded_frames": graded_frames,
        "graded_minutes": float(graded_mins),
        "ball_visible_pct": float(100.0 * np.mean(observed[graded_mask])) if graded_frames else 0.0,
        "ball_ok_pct": float(100.0 * np.mean(sig["ball_ok"][graded_mask])) if graded_frames else 0.0,
        "touches_total": int(touch_mask.sum()),
        "touches_per_min": float(touch_mask.sum() / graded_mins),
        "touches_home": int((touch_mask & (touch_team == touch.HOME)).sum()),
        "touches_away": int((touch_mask & (touch_team == touch.AWAY)).sum()),
        "touch_dist": dist_stats,
        "share_in_control_pct": 100.0 * in_ctrl_n / max(graded_frames, 1),
        "share_in_flight_pct": 100.0 * in_flight_n / max(graded_frames, 1),
        "share_unknown_pct": 100.0 * unknown_n / max(graded_frames, 1),
        "episodes_total": sum(h["episodes"] for h in halves),
        "mean_ep_s": float(all_durs_s.mean()) if len(all_durs_s) else 0.0,
        "median_ep_s": float(np.median(all_durs_s)) if len(all_durs_s) else 0.0,
        "halves": halves,
    }


def print_report(stats):
    print("=== Approach A Touch & Control Report (aggregates only) ===")
    print("graded: {} frames ({:.1f} min) | ball observed {:.1f}% | ball_ok {:.1f}%".format(
        stats["graded_frames"], stats["graded_minutes"], stats["ball_visible_pct"], stats["ball_ok_pct"]
    ))
    print("touches: {} ({:.2f}/min) | home {} | away {}".format(
        stats["touches_total"], stats["touches_per_min"], stats["touches_home"], stats["touches_away"]
    ))
    td = stats["touch_dist"]
    if td:
        print("toucher distance (m): mean {:.2f} | p10 {:.2f} | p25 {:.2f} | p50 {:.2f} | p75 {:.2f} | p90 {:.2f}".format(
            td["mean"], td["p10"], td["p25"], td["p50"], td["p75"], td["p90"]
        ))
    print("graded frame state share: in_control {:.1f}% | in_flight {:.1f}% | unknown {:.1f}%".format(
        stats["share_in_control_pct"], stats["share_in_flight_pct"], stats["share_unknown_pct"]
    ))
    print("control episodes (graded): {} | mean {:.2f} s | median {:.2f} s".format(
        stats["episodes_total"], stats["mean_ep_s"], stats["median_ep_s"]
    ))
    for h in stats["halves"]:
        print(
            "  half {}: {:.1f} min | touches {} ({:.2f}/min, H {}, A {}) | "
            "episodes {} (H {}, A {}; carry {}, pass {}) | mean {:.2f} s, median {:.2f} s".format(
                h["half"], h["minutes"], h["touches"], h["touches_per_min"],
                h["touches_home"], h["touches_away"], h["episodes"],
                h["episodes_home"], h["episodes_away"], h["episodes_carry"], h["episodes_pass"],
                h["mean_ep_s"], h["median_ep_s"],
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default=None, help="default: ./data or $MTAH_DATA_DIR")
    parser.add_argument("--no-tagging", action="store_true", help="ignore Smart Tagging")
    parser.add_argument("--synthetic", action="store_true", help="run on synthetic simulate() match")
    parser.add_argument("--seed", type=int, default=0, help="seed for --synthetic")
    args = parser.parse_args()

    if args.synthetic:
        pitch = (105.3, 68.0)
        sim = simulate.simulate(seed=args.seed, pitch=pitch)
        stats = summarize_signals(sim["ball"], sim["players"], sim["team"], pitch, sim["blocks"])
        print_report(stats)
        return

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, args.data_dir)
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]
    ball, players, team = atd.load_atd(raw_path, meta_path, n_frames, pitch)
    intervals = [] if args.no_tagging else run.set_piece_intervals(args.data_dir)
    stats = summarize_signals(ball, players, team, pitch, half_video_blocks(GAME), intervals)
    print_report(stats)


if __name__ == "__main__":
    main()
