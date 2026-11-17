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

1. `atd.py` reads the raw tracking into arrays in metres (`video_frame = frame_id - 1`), plus
   each player's speed as the tracker reports it.
2. `signals.py` cleans the ball (rejects jumps > 45 m/s, bridges gaps <= 1.5 s) and computes,
   per frame: nearest home/away distance to the ball, ball speed, out-of-bounds, near-line, and
   the average speed of all tracked players over 1 s.
3. `model.py` scores each frame for H / A / D and decodes with **Viterbi** under a switching
   penalty, so a change of possession must be earned by ~1-2 s of consistent evidence. Where
   there is no evidence the state persists.
   - team evidence: which team is nearer the ball, weighted by how close anyone is (a ball in
     flight carries almost none), plus a weak prior from Smart Tagging phases that say who has
     the ball (Build Up, Progression, ...: the tagged team; Defending in ... thirds, Defensive
     Transition: the other one). It separates H from A; against D it is capped at log 2 a frame,
     because somebody is near the ball during a stoppage too.
   - dead-ball evidence: players walking (below 1.8 m/s on average, works without the ball);
     ball beyond a line (strong); slow ball on a line, slow ball nobody is near, inside a Smart
     Tagging set-piece window (weak); fast ball (live)
4. Cleanup merges runs too short to be real (team run < 1.5 s = touch/deflection, D run < 2 s).
5. Only the two graded halves are labelled; everything else is `X`. The second half is `D` until
   its published kickoff frame.

All thresholds are physical quantities set from football knowledge. None were fitted to the
public 5 minutes or to anything about this match. The Smart Tagging codes used for the team prior
were picked by what they mean, then checked against how often the tagged team is the one nearest
the ball over the full match (no ground truth needed).

`run.py` prints a label-free summary at the end (share of dead time, number of stoppages and
possessions). Compare it with what a match normally has, not with the public window.

## Status, honestly

First built without the match data. Once it arrived, the correctness check showed the model was
nearly blind to dead balls: on the real tracker the ball is missing for about a third of every
stoppage, `out_of_play` rarely fires, and a player near the ball outvoted every dead-ball cue. The
second version adds the walking cue, caps team evidence against D and uses the Smart Tagging
phases. On the full match it now calls about half the graded time dead, with ~146 stoppages
(was 11% and 65).

The synthetic test checks the code path and guards against regressions; it says nothing about
real-match accuracy. The simulated players move at ~1.7-1.8 m/s even in live play, close to the
walking threshold, so the new cue is not much help there and the synthetic score went down
(still above the baseline).

Do not choose parameters by the public 5-minute score (see EVALUATION.md: it picks the wrong
smoothing window even for the baseline).
