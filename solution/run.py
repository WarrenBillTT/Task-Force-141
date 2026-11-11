"""Build the possession submission:  data bundle -> frame,label CSV.

    python solution/run.py --out evaluation/results/<team_name>.csv
    python evaluation/validate_submission.py evaluation/results/<team_name>.csv
    python evaluation/run_evaluation.py          # public 5 min: a correctness check, never a tuning set

Deterministic: same data + same code = same file.
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

import atd  # noqa: E402
import model  # noqa: E402
import signals  # noqa: E402
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402

SET_PIECE_CODE = "set pieces"
SECOND_HALF_KICKOFF = 113742  # published in DATA.md / FAQ.md and starter/load_smart_tagging.py


def build_signals(ball_raw, players, team, pitch, set_piece_intervals=()):
    """Everything the state model needs, for the whole video."""
    ball, observed, filled = signals.clean_ball(ball_raw)
    dist_home, dist_away = signals.team_distances(ball, players, team)
    return {
        "dist_home": dist_home,
        "dist_away": dist_away,
        "speed": signals.ball_speed(ball),
        "ball_ok": observed | filled,
        "oob": signals.out_of_play(ball, pitch),
        "near_line": signals.near_boundary(ball, pitch),
        "set_piece": signals.window_mask(len(ball), set_piece_intervals),
    }


def predict(sig, blocks, cfg, kickoff2=None, n_frames=None):
    """Label array (letters) over the whole video. Outside the graded halves: X (never scored)."""
    n_frames = n_frames or len(sig["speed"])
    labels = np.full(n_frames, "X", dtype="<U1")
    for half, first, last in blocks:
        start = first
        if half == 2 and kickoff2 is not None and first < kickoff2 <= last:
            labels[first:kickoff2] = "D"  # graded, but the ball is not in play until kickoff
            start = kickoff2
        part = {k: v[start:last + 1] for k, v in sig.items()}
        labels[start:last + 1] = model.LETTERS[model.decode(part, cfg)]
    return labels


def set_piece_intervals(data_dir):
    from starter.load_smart_tagging import load
    return [(e["start_frame"], e["end_frame"]) for e in load(GAME, data_dir)
            if (e["code"] or "").strip().lower() == SET_PIECE_CODE]


def write_csv(path, labels):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, lab) for i, lab in enumerate(labels))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--data-dir", default=None, help="default: ./data or $MTAH_DATA_DIR")
    parser.add_argument("--no-tagging", action="store_true", help="ignore Smart Tagging")
    for name, value in vars(model.Config()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    cfg = model.Config(**{k: getattr(args, k) for k in vars(model.Config())})

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, args.data_dir)
    n_frames = source_frames("tactical", GAME)
    pitch = GAMES[GAME]["pitch"]

    print("reading tracking ...")
    ball, players, team = atd.load_atd(raw_path, meta_path, n_frames, pitch)

    intervals = []
    if not args.no_tagging:
        intervals = set_piece_intervals(args.data_dir)
        print("smart tagging: {} set-piece windows".format(len(intervals)))

    sig = build_signals(ball, players, team, pitch, intervals)
    labels = predict(sig, half_video_blocks(GAME), cfg, SECOND_HALF_KICKOFF, n_frames)
    write_csv(args.out, labels)

    counts = {k: int((labels == k).sum()) for k in "HADX"}
    print("wrote {}  rows={}  {}".format(args.out, len(labels), counts))
    print("next: python evaluation/validate_submission.py {}".format(args.out))


if __name__ == "__main__":
    main()
