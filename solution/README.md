# Possession pipeline (More than a Hack 2027, Phase 1)

Tracking-only approach: no video, no training data, no learned weights. Everything is derived
from the provided automatic tracking (ATD) and, optionally, the Smart Tagging.

## Run

```bash
pip install -r requirements.txt                       # numpy + pandas
# put the downloaded bundle in ./data  (or: export MTAH_DATA_DIR=/path/to/bundle)
python solution/run.py --out evaluation/results/<team_name>.csv
python evaluation/validate_submission.py evaluation/results/<team_name>.csv
python evaluation/run_evaluation.py                   # public 5 min: correctness check only
python -I solution/test_pipeline.py                   # synthetic smoke test, needs no data
```

Deterministic and about 5 s of compute on top of reading the tracking file.

## How it works

1. `atd.py` reads the raw tracking into arrays in metres (`video_frame = frame_id - 1`).
2. `signals.py` cleans the ball (rejects jumps > 45 m/s, bridges gaps <= 1.5 s) and computes,
   per frame: nearest home/away distance to the ball, ball speed, out-of-bounds, near-line.
3. `model.py` scores each frame for H / A / D and decodes with **Viterbi** under a switching
   penalty, so a change of possession must be earned by ~1-2 s of consistent evidence. Where the
   ball is unobserved the evidence is zero and the state persists.
   - team evidence: which team is nearer the ball, weighted by how close anyone is
     (a ball in flight carries almost none)
   - dead-ball evidence: ball beyond a line (strong); slow ball on a line, slow ball nobody is
     near, slow ball inside a Smart Tagging set-piece window (weak); fast ball (live)
4. Cleanup merges runs too short to be real (team run < 1.5 s = touch/deflection, D run < 1 s).
5. Only the two graded halves are labelled; everything else is `X`. The second half is `D` until
   its published kickoff frame.

All thresholds are physical quantities set from football knowledge. None were fitted to the
public 5 minutes or to anything about this match.

## Status, honestly

Built and tested **without the match data** (it is gated). The synthetic test checks the code path
and guards against regressions; it says nothing about real-match accuracy. The first thing to do
with the real data is the correctness check above, and then look at what the model gets wrong:
how often the ball is missing during dead balls, whether `out_of_play` ever fires on the real
tracker, and what the Smart Tagging set-piece windows cover.

Do not choose parameters by the public 5-minute score (see EVALUATION.md: it picks the wrong
smoothing window even for the baseline).
