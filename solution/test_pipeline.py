"""End-to-end smoke/regression test on a synthetic match (no gated data needed).

    python -I solution/test_pipeline.py

Checks: the ATD parser round-trips, the output is a valid submission, and the state model beats the
repo's own 'nearest player + majority vote' baseline on the synthetic match. The last check is a
regression guard for the code, NOT an estimate of real-match accuracy.
"""

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (HERE, ROOT, os.path.join(ROOT, "evaluation")):
    sys.path.insert(0, path)

import atd  # noqa: E402
import model  # noqa: E402
import possession_metrics as pm  # noqa: E402
import run  # noqa: E402
import simulate  # noqa: E402
from starter.baseline import smooth  # noqa: E402

PITCH = (105.3, 68.0)
N = 20000
BLOCKS = [(1, 500, 9500), (2, 10500, 19500)]
KICKOFF2 = 10800


def score(gt, labels):
    result = pm.evaluate(pm.label_codes(gt), pm.label_codes(labels))
    return result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"], result["n_gt_spells"]


def nearest_baseline(sig, blocks, window=50):
    codes = np.where(sig["dist_home"] < sig["dist_away"], 0, 1).astype(np.int8)
    codes[~np.isfinite(np.minimum(sig["dist_home"], sig["dist_away"]))] = -1
    letters = np.full(len(codes), "X", dtype="<U1")
    for _, first, last in blocks:
        letters[first:last + 1] = smooth(codes[first:last + 1], window)
    return letters


def main():
    sim = simulate.simulate(N, BLOCKS, seed=1, pitch=PITCH)
    with tempfile.TemporaryDirectory() as tmp:
        raw, meta = os.path.join(tmp, "atd.txt"), os.path.join(tmp, "atd.xml")
        simulate.write_atd(sim, raw, meta, PITCH)
        ball, players, team = atd.load_atd(raw, meta, N, PITCH)

    both = ~np.isnan(ball[:, 0]) & ~np.isnan(sim["ball"][:, 0])
    assert np.abs(ball[both] - sim["ball"][both]).max() < 0.02, "ball round-trip"
    assert np.array_equal(team, sim["team"]), "team slots"
    print("parser round-trip OK; ball visible on {:.1f}% of frames".format(100 * np.mean(~np.isnan(ball[:, 0]))))

    sig = run.build_signals(ball, players, team, PITCH)
    labels = run.predict(sig, BLOCKS, model.Config(), kickoff2=KICKOFF2, n_frames=N)
    labels_nokick = run.predict(sig, BLOCKS, model.Config(), kickoff2=None, n_frames=N)
    assert (labels[10500:KICKOFF2] == "D").all(), "pre-kickoff must be D"

    frame_table = __import__("pandas").DataFrame({"frame": np.arange(N), "label": labels})
    assert pm.validate_submission(frame_table, N) is None, "invalid submission"

    gt = sim["gt"].copy()
    ours = score(gt, labels_nokick)
    base = score(gt, nearest_baseline(sig, BLOCKS))
    print("{:<22} spellF1 {:.4f}  macroF1 {:.4f}  spells {} (truth {})".format("state model", *ours))
    print("{:<22} spellF1 {:.4f}  macroF1 {:.4f}  spells {} (truth {})".format("nearest+smooth50", *base))
    assert ours[0] > base[0], "state model should beat the baseline on the synthetic match"
    print("OK")


if __name__ == "__main__":
    main()
