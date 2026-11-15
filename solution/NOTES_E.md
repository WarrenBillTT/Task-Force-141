# NOTES_E: possession decoder upgrade (HSMM + boundary refinement + consensus)

Everything here was built and tested on **synthetic data only** (`simulate.py`). No real match data,
ground truth or results were read, and nothing was tuned on any of them. The emission model
(`model.emissions`) is untouched and treated as a black box.

## Files

| File | Role |
|---|---|
| `hsmm.py` | E1. Explicit-duration decoder + brute-force reference |
| `refine.py` | E2. Bounded boundary refinement from `sig` |
| `ensemble.py` | E3. K-variant consensus + per-segment stability; `decode_single` / `decode_full` |
| `stability.py` | Label-free perturbation check (drop extra ball frames, compare two decodes) |
| `signals.py` | + `ball_velocity_change` (kick / trap / deflection spike), `run.build_signals` adds `ball_dv` |
| `model.py` | `Config` gains the parameters below; `decode()` dispatches (default path unchanged) |
| `run.py` | `--decoder {viterbi,hsmm}`, `--refine`, `--ensemble N`; output format unchanged |
| `test_decoder.py` | 20 tests + `--table` synthetic comparison |

`run.predict()` (graded halves, second-half kickoff handling) is unchanged.

## Design

**HSMM (E1).** Segments, not frames, are scored: `sum emissions + dur_scale * log P(duration | label)
+ transition`. Cumulative sums make segment emission O(1). The DP runs on a 5-frame grid and is
vectorised over durations; boundaries are then re-polished at 1-frame resolution against the
emissions. First/last segments of a stretch are censored (no duration prior). Segments longer than
`hsmm_max_seconds` can chain (one switch penalty) so the problem is always feasible. Frames with no
ball have emission 0, so the state persists exactly as in the Viterbi. `test_hsmm_matches_brute_force`
checks the DP (step 1) against exhaustive enumeration over all segmentations and labelings (same
optimum score on 12 random tiny cases, including cap/chaining and zero-evidence frames).

**Refinement (E2).** For each boundary, within +/- 2 s, choose the frame maximising a step fit of local
cues + event credit - drift. Cues: team nearness (H<->A, only while someone is within 3 m), ball left
the pitch / is clearly fast (alive<->D), plus credit for a ball velocity-change spike and, if
`sig["touch"]` exists, the touch score. Labels, segment count and minimum lengths are preserved
(tested). Boundaries are processed left to right and each side must stay >= the configured minimum
(or as long as it already was).

**Ensemble (E3).** `variants(cfg, K)`: first K of a fixed 5x5 grid (switch/dead penalty x
{0.6, 0.8, 1, 1.25, 1.6}, duration-prior weight x {0.4, 0.7, 1, 1.4, 2}) ordered by distance from the
base, so `--ensemble N` is always the same N variants (max 25). Each variant is decoded, cleaned up and
(optionally) refined. Boundaries are clustered (1 s), kept if >= half of the variants have one,
positioned at the cluster's lower median; each resulting segment takes the per-frame majority label of
the variants; `merge_short_runs` is re-applied. `decode_ensemble(..., return_stability=True)` also
returns per segment the fraction of variants having a same-label segment with both edges within 1 s.
No randomness anywhere.

## New parameters (`model.Config`)

| Parameter | Default | Physical reasoning |
|---|---|---|
| `decoder` / `refine` / `ensemble` | viterbi / 0 / 1 | Selection; defaults reproduce the original pipeline |
| `hsmm_step` | 5 frames | 0.2 s coarse grid; far below any boundary precision that matters; polished to 1 frame |
| `hsmm_max_seconds` | 150 s | Longest scored segment (a very long spell or stoppage); longer chains |
| `alive_median_s`, `alive_sigma` | 7 s, 1.0 | Possession spell: a few passes / one phase of play. Log-normal 5-95% = 1.4-36 s: sub-second to ~a minute with a heavy tail |
| `dead_median_s`, `dead_sigma` | 12 s, 0.7 | Throw-in ~8 s, free kick / goal kick / corner 15-25 s; 5-95% = 3.8-38 s, practically never under a couple of seconds; injuries give the tail |
| `dur_scale` | 1.0 | Weight of the log duration prior (varied by the ensemble) |
| `hsmm_penalty_scale` | 0.5 | The duration prior already charges short segments, so the Viterbi's switch/dead penalties are halved to avoid double-paying |
| `refine_window_s` | 2 s | Decoder lag is about 1-2 s (the evidence needs that long to pay the switch) |
| `refine_radius` | 3 m | A team cue means something only when a player is plausibly in control |
| `refine_margin` | 1 m | Distance gap at which the "who is nearer" cue saturates (tanh) |
| `refine_dv_ref` / `refine_dv_weight` | 8 m/s / 4 | A pass or trap changes the ball velocity by several m/s; 4 cue-frames of credit |
| `refine_touch_weight` | 6 | Touch score (optional) is a stronger, more specific cue than a velocity spike |
| `refine_slow_weight` | 0.0 | A slow ball is ambiguous (a held ball or restart is slow on both sides of the boundary and only drags the boundary to the window edge); the decisive cues are "left the pitch" and "clearly fast" |
| `refine_drift` | 0.02 /frame | Ties stay where the decoder put the boundary |

The duration and penalty values come from general football knowledge, not from any data. The
simulator's durations are different (e.g. uniform dead balls of 4-12 s), by design.

## Synthetic comparison (regression guard; do not tune on it)

20,000-frame synthetic matches, seeds 1-5, no second-half kickoff handling, spellF1 against the
simulator's truth. "noisy" adds 0.5 m ball noise and ~15% extra dropout bursts, "harsh" 1.0 m and ~30%.

| level | method | mean | std | min | max |
|---|---|---|---|---|---|
| clean | viterbi | 0.8961 | 0.0136 | 0.8818 | 0.9171 |
| clean | hsmm | 0.9221 | 0.0254 | 0.8933 | 0.9591 |
| clean | hsmm+refine | 0.9182 | 0.0219 | 0.8933 | 0.9500 |
| clean | hsmm+refine+ens5 | 0.9201 | 0.0201 | 0.8933 | 0.9500 |
| noisy | viterbi | 0.8350 | 0.0438 | 0.7680 | 0.8990 |
| noisy | hsmm | 0.8639 | 0.0421 | 0.8148 | 0.9196 |
| noisy | hsmm+refine | 0.8553 | 0.0342 | 0.8099 | 0.8957 |
| noisy | hsmm+refine+ens5 | 0.8572 | 0.0365 | 0.8099 | 0.9042 |
| harsh | viterbi | 0.6981 | 0.0507 | 0.6263 | 0.7551 |
| harsh | hsmm | 0.7111 | 0.0400 | 0.6582 | 0.7604 |
| harsh | hsmm+refine | 0.7079 | 0.0376 | 0.6633 | 0.7516 |
| harsh | hsmm+refine+ens5 | 0.7096 | 0.0396 | 0.6633 | 0.7600 |

Reproduce: `python solution/test_decoder.py --table`.

Reading: HSMM is >= Viterbi at every level (+0.03 / +0.03 / +0.01), but the seed-to-seed spread
(0.02-0.05) is as large as the gaps, so call it "not worse, probably a bit better". Refinement and the
ensemble do **not** help on the simulator (about -0.004 / -0.008 / -0.003 vs plain HSMM, inside the noise).
That is expected and says little: the simulator has no real touches, its restarts have no kick cue
(live play starts with a held ball) and its dead balls start while the ball is still moving, so the
cues refinement relies on are mostly absent. Per boundary type on a clean run, refinement left H<->A
timing unchanged (~5.5 frames mean error) and made D->alive slightly worse (9 -> 13 frames) until the
slow-ball cue was set to 0 (it was 25 frames with a 0.3-1.0 weight). That is a finding about the
simulator as much as about the cue, and I did not tune further.

Label-free stability (`python solution/stability.py --seeds 5`: 25% extra ball frames dropped in
bursts and flickers, clean decode vs perturbed decode, spellF1 of one against the other):

| decoder | spellF1 agreement | worst seed | possession-count ratio |
|---|---|---|---|
| viterbi | 0.860 | 0.828 | 0.864 |
| hsmm | 0.879 | 0.844 | 0.892 |
| hsmm + refine | 0.864 | 0.822 | 0.892 |
| hsmm + refine + ens5 | 0.859 | 0.818 | 0.894 |

## Choice of default

The default stays **`viterbi`, no refine, no ensemble**, so the existing output is byte-identical. HSMM
passed the "not worse" criterion on the simulator and is the recommended candidate, but it rests on
duration priors that only real data can vindicate, and I am not allowed to look at it. Flip the default
in `model.Config` after the label-free comparison below. Refinement and the ensemble showed no benefit
here and stay opt-in.

## Timing

`hsmm.decode` on 80,000 random frames (one half, full-match size): about 1.0 s (the whole 20,000-frame
synthetic comparison of four methods x 15 matches takes about a minute). An N-variant ensemble costs
about N decodes.

## Known limitations

* Duration priors are generic. A league or tracker with different spell lengths will want different
  medians; they are plain `Config` fields (`--alive-median-s 9` etc.).
* The HSMM decodes the emissions it is given: if the emission model is wrong (e.g. `oob` never fires),
  the decoder will be confidently wrong.
* Refinement can only use what is in `sig`. Without touches it relies on nearest-player distance and
  ball speed/velocity change. On the simulator it is a wash; its value on real data is unverified.
* The ensemble variants share the emissions; they probe decoder settings, not evidence errors.
* Ties in a per-segment variant vote go to the lowest label index (H < A < D).
* The consensus can drop a boundary that only a minority of variants found even if it is real (that is
  the point, but it favours merges over splits when variants disagree).
* The `-I` flag of `python -I solution/test_pipeline.py` ignores the per-user site-packages. On this
  machine `python-dateutil` (a pandas dependency) is only in the user site, so `-I` failed with an
  ImportError before I changed anything; I ran the tests without `-I`. In a clean environment
  (`pip install -r requirements.txt`) `-I` works.

## What to run on the real data (human, locally)

```bash
python -I solution/test_pipeline.py            # existing smoke test
python -I solution/test_decoder.py             # unit tests + invariants (needs no data)

# 1. build candidate submissions (never commit them)
python solution/run.py --out evaluation/results/vit.csv
python solution/run.py --decoder hsmm --out evaluation/results/hsmm.csv
python solution/run.py --decoder hsmm --refine --out evaluation/results/hsmm_ref.csv
python solution/run.py --decoder hsmm --refine --ensemble 5 --out evaluation/results/hsmm_ens.csv
for f in vit hsmm hsmm_ref hsmm_ens; do python evaluation/validate_submission.py evaluation/results/$f.csv; done

# 2. label-free stability: higher spellF1 agreement and count ratio close to 1 are better
python solution/stability.py --real --seeds 5
python solution/stability.py --real --decoder hsmm --seeds 5
python solution/stability.py --real --decoder hsmm --refine --seeds 5
python solution/stability.py --real --decoder hsmm --refine --ensemble 5 --seeds 5
```

To compare viterbi against hsmm using only label-free checks:

1. `validate_submission.py` passes for both.
2. Stability: hsmm should have at least the agreement of viterbi and a count ratio at least as close to 1.
3. Plausibility of the structure (counts only, no labels): number of possessions and stoppages per
   half, and the median / 5th percentile of H/A and D run lengths, against football expectations
   (roughly 6-12 s median possession, nothing live under 1.5 s, stoppages mostly over 3 s). Neither
   decoder should produce implausible values.
4. Agreement between the two outputs (`stability.agreement(a, b)` on the two label arrays) tells you how
   much the decoder choice matters at all. If it is tiny, keep viterbi.

Do not choose by the public 5-minute score (EVALUATION.md); use it only as the final correctness check.
