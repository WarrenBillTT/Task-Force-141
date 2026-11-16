# NOTES_A: Touch and Control-Based Possession Evidence (Approach A)

Everything in Approach A was designed from football kinematics and ATD noise physics, and tested
strictly on **synthetic scenes and `simulate.py`**. No real match assets (`./data`), ground truth
(`evaluation/gt/`), or submissions (`evaluation/results/`) were read or used for tuning.

## Files

| File | Role |
|---|---|
| `solution/touch.py` | A1–A5: ball dynamics, touch detection + NMS, carry/pass control episodes, per-frame `ctrl_logodds` & `ctrl_conf` |
| `solution/model.py` | A6: `use_touch` / `touch_weight` in `Config`; confidence-weighted blend in `emissions(sig, cfg)` (dead-ball & `decode()` untouched; `--no-touch` reproduces legacy behaviour exactly) |
| `solution/run.py` | Wires `touch.build_touch_signals()` into `build_signals()` and adds `--touch` / `--no-touch` CLI flags |
| `solution/simulate.py` | Backward-compatible touch event recording (`"touches"`) and optional separate-RNG stress perturbations |
| `solution/touch_report.py` | Aggregate-only diagnostics report for human local inspection (`--synthetic` mode available for testing) |
| `solution/test_touch.py` | 8 hand-made scene tests, invariants, `--no-touch` equivalence, 80k-frame timing, multi-seed benchmark, and perturbation stability |
| `solution/test_pipeline.py` | Updated for `python -I` isolated-mode user-site compatibility |

---

## Design Overview

1. **A1 — Ball dynamics (`ball_dynamics`)**:
   - Applies `[0.25, 0.50, 0.25]` spatial smoothing strictly across **observed** (`observed & ~filled`) frames so tracking jitter ($\sigma \approx 0.3\text{ m}$) is damped without smearing interpolated gap edges.
   - Evaluates incoming velocity (`vel_before`) and outgoing velocity (`vel_after`) over `dyn_window_frames = 5` frames (`0.20 s`), using short linear-fill positions only when measuring velocity across a short gap from its observed endpoints, and requiring a real observed frame in `[t - 7, t - 2]` before `vel_before` can drive a velocity-jump spike (avoiding late false receptions at the trailing edge of interpolated gaps).
   - Outputs `vel`, `vel_before`, `vel_after`, `speed_before`, `speed_after`, `dv`, `dspeed`, `accel`, and `turn_angle`.
   - Never creates touch evidence inside `filled` or unobserved frames (`~observed | filled`).

2. **A2 — Touch candidates (`detect_touches`)**:
   - Fires on observed, inbounds frames where a tracked player is within `contact_radius = 2.2 m`, `max(speed_before, speed_after) >= 6.5 m/s`, and `dv >= 6.5 m/s` paired with either a scalar speed jump `dspeed >= 4.8 m/s` or a directional turn `>= 38 deg` (with `min_turn_speed = 2.2 m/s`).
   - Suppresses longitudinal noise spikes along straight high-speed flights (`min(speed_before, speed_after) > 7.5 m/s`, `turn < 38 deg`, `dv < 0.35 * max_speed`).
   - Applies greedy Non-Maximum Suppression over `nms_window_s = 0.20 s` (`5` frames), scores confidence in `[0, 1]` from dynamic strength, proximity, and whether `|d_away - d_home| >= contest_margin (0.50 m)`, and writes `sig["touch"]` (`float32`) and `sig["touch_team"]` (`int8`).

3. **A3 — Control episodes (`detect_control_episodes`)**:
   - **Carry / dribble / hold (`"carry"`)**: vectorially identifies the nearest track slot and its velocity (rejecting implausible slot jumps `> 12.5 m/s` so ID swaps do not corrupt velocity). Frames where the ball is observed, inbounds, uncontested (`margin >= 0.50 m`), within `carry_radius = 1.8 m`, `speed <= 8.5 m/s`, and matching the carrier velocity (`|v_ball - v_player| <= 4.5 m/s`) are bridged across short dropouts (`<= 12` frames) unless blocked by `oob` or opposing control. Track-slot switches within `swap_link_radius = 2.2 m` (or slow holds `<= 4.5 m/s`) are merged so ID swaps mid-dribble stay continuous. Runs `>= min_carry_s (0.52 s = 13 frames)` emit carry episodes with `confirm = start + 12` (control established after `0.52 s`, not at first contact).
   - **Passing sequence (`"pass"`)**: pairs consecutive uncontested same-team touches (excluding 50-50 duels within `0.28 s` of an opposing touch) separated by `<= max_pass_interval_s (2.8 s = 70 frames)` with no `oob` and no opposing carry in between, confirming at the second touch `t2`.

4. **A4 & A5 — Per-frame control log-odds (`compute_control_evidence`)**:
   - Maintains the current controlling team `cur_team` and confidence `cur_conf`:
     - Possession follows the most recent established control episode.
     - **A lone opposing touch (deflection, interception attempt, aerial duel) never switches possession**; a switch occurs only at `confirm` of an opposing carry or passing episode.
     - While the ball is in flight or loose, `cur_team` is retained with exponential decay (`flight_decay_s = 3.0 s`).
     - Sustained out-of-bounds (`>= 1.0 s`) resets `cur_team = -1`.
     - Missing ball stretches (`~ball_ok`) output `ctrl_logodds = 0.0` and `ctrl_conf = 0.0` (so emissions are zero and the decoder persists), while linearly interpolated `filled` frames scale `ctrl_conf` by `filled_conf_scale = 0.35`.

5. **A6 — Emission integration (`model.emissions`)**:
   - Leaves dead-ball evidence, `transition_matrix`, `viterbi`, `merge_short_runs`, `decode()`, and `kickoff2` handling completely unchanged.
   - When `cfg.use_touch` is `1` (`--touch`, default), `ctrl_logodds` and nearest-player proximity `prox_team` are combined cleanly: when both agree on sign, the stronger signal is kept; when they disagree, they blend with weight `w = touch_weight * ctrl_conf^2 * flight_factor`, where `flight_factor = 1 - 0.40 * exp(-(nearest / 2.2)^2)` gives control logic dominant weight during flights/pass-bys while letting sustained close-range steals overcome stale control memory.
   - Slow dead-ball indicator frames (`oob`, `near_line`, or `slow & (set_piece | nearest > idle_distance)`) zero out `ctrl_team` so control memory never suppresses `D` stoppages.
   - `--no-touch` sets `use_touch = 0` and reproduces the original proximity-only emissions and labels bit-for-bit.

---

## Parameters and Physical Justification

### `TouchConfig` (`solution/touch.py`)

| Parameter | Default | Unit | Physical / football reasoning |
|---|---|---|---|
| `smooth_half_window` | `1` | frames | `[0.25, 0.50, 0.25]` smoothing damps ~0.3 m ATD jitter without blurring touch vertices |
| `dyn_window_frames` | `5` | frames (0.20 s) | Velocity window before/after `t`; long enough to stabilize differentiation, short enough for 0.4 s passes |
| `min_turn_speed` | `2.2` | m/s | Minimum speed on both sides before directional turn angle is physically meaningful |
| `contact_radius` | `2.2` | m | Player reach (~0.7 m) plus ball/player ATD tracking error (~1.5 m) |
| `min_touch_speed` | `6.5` | m/s | Ensures `max(v_before, v_after)` reaches at least a crisp pass, suppressing hold/walk jitter |
| `min_dv` | `6.5` | m/s | Vector velocity jump over 0.20 s (`~32.5 m/s^2`), well above grass drag and >3 sigma ATD noise |
| `min_speed_jump` | `4.8` | m/s | Scalar speed change threshold for kicks from rest or traps to rest |
| `min_turn_deg` | `38.0` | deg | Redirection angle threshold for volleys, one-touch passes, and deflections |
| `fast_flight_speed` | `7.5` | m/s | Longitudinal flight filter: straight fast motion (`turn < 38 deg`) requires `dv >= 0.35 * max_speed` |
| `dv_ref` / `speed_jump_ref` / `turn_ref_deg` | `12.0` / `10.0` / `90.0` | m/s, m/s, deg | Reference scales mapping kinematic abruptness onto `[0, 1]` confidence |
| `nms_window_s` | `0.20` | s (5 frames) | Single foot contact spans ~1 frame; human double-touches take > 0.2 s |
| `contest_margin` | `0.50` | m | Minimum `|d_away - d_home|` distance advantage for an uncontested touch or carry |
| `carry_radius` | `1.8` | m | Ball-to-holder proximity while dribbling, shielding, or holding |
| `carry_max_dv` | `4.5` | m/s | Max `|v_ball - v_player|` for ball moving with carrier under ATD noise |
| `carry_max_ball_speed` | `8.5` | m/s | Sprint dribble speed ceiling; rejects faster free passes flying near a player |
| `min_carry_s` | `0.52` | s (13 frames) | Minimum continuous carry/hold needed to establish control (~0.5 s rule) |
| `carry_bridge_frames` | `12` | frames (0.48 s) | Bridges brief player/ball dropouts inside an uninterrupted same-team carry |
| `swap_link_radius` | `2.2` | m | Spatial continuity radius to link same-team track-slot swaps mid-dribble |
| `max_pass_interval_s` | `2.8` | s (70 frames) | Max time between consecutive uncontested same-team touches in a passing sequence |
| `duel_pair_s` | `0.28` | s (7 frames) | Opposing uncontested touches closer than this indicate a contested duel |
| `ctrl_gain` / `ctrl_cap` | `1.3` / `1.2` | nats | Maps control confidence `[0, 1]` to capped log-odds matching `model.Config.gap_cap` |
| `flight_decay_s` | `3.0` | s | Physical ball-flight hang-time scale for confidence decay while loose |
| `filled_conf_scale` | `0.35` | ratio | Downweights control confidence on linearly interpolated `filled` frames |
| `oob_reset_s` | `1.0` | s (25 frames) | Sustained out-of-bounds resets prior possession state before restart |

### `model.Config` additions (`solution/model.py`)

| Parameter | Default | Reasoning |
|---|---|---|
| `use_touch` | `1` | `1` (`--touch`): enable control-based team evidence; `0` (`--no-touch`): exact legacy proximity model |
| `touch_weight` | `0.80` | Max weight on `ctrl_logodds` when control confidence is 1.0 and ball is away from opposing contact |

---

## Synthetic Results (No Real Data Used)

Measured with `python -I solution/test_touch.py` and `python -I solution/test_pipeline.py`:

### 1. Multi-seed benchmark (`simulate.py`, seeds 0..5, 20,000 frames per seed)

- `standard`: default `simulate()` (~18% ball missing, 0.3 m jitter, 0.3% spikes, 2% player dropouts).
- `stress40`: 40% burst-pattern ball loss, +0.25 m extra jitter, +0.5% extra spikes, `id_swap_rate=0.01`, `deflection_rate=0.35` (mid-flight opponent pass-bys & trajectory deflections).

| Scenario | Touch Precision (+/-3f) | Touch Recall (+/-3f) | `spellF1` (`--touch`) | `spellF1` (`--no-touch`) | `macroF1` (`--touch` vs `--no-touch`) |
|---|---|---|---|---|---|
| `standard` | 0.676 +/- 0.032 | 0.813 +/- 0.027 | **0.9238 +/- 0.0300** | 0.8964 +/- 0.0124 | **0.9312** vs 0.9237 |
| `stress40` | 0.465 +/- 0.038 | 0.757 +/- 0.021 | **0.8089 +/- 0.0346** | 0.8012 +/- 0.0396 | **0.8870** vs 0.8829 |

### 2. Smoke test (`python -I solution/test_pipeline.py`, seed 1)

- `state model` (`--touch`): `spellF1 0.8848`, `macroF1 0.9214`, `spells 50 (truth 49)`
- `nearest+smooth50` baseline: `spellF1 0.6218`, `macroF1 0.5777`, `spells 70 (truth 49)`

### 3. Label-free perturbation stability (`test_perturbation_stability`, seeds 0..2)

Dropping an extra 20% of graded ball frames in bursts and adding 0.15 m Gaussian jitter (`--touch` clean vs perturbed):
- `spellF1` agreement: **0.9176 +/- 0.0205**
- `macroF1` agreement: **0.9459 +/- 0.0008**

### 4. Timing (80,000 frames = 1 full match half)

- `run.build_signals` (including full `touch.build_touch_signals`): **1.18 s** (measured in `test_timing_full_half`).

---

## Known Failure Modes & Limitations

1. **Soft touches / one-touch passes below velocity threshold**: Very soft taps with `dv < 6.5 m/s` or `max_speed < 6.5 m/s` will not fire a discrete touch spike (they rely on carry detection if held `>= 0.52 s` or proximity fallback).
2. **Simultaneous ball occlusion at an interception**: Because `touch.py` never creates touch evidence inside `filled` gaps and scales down confidence there, an interception that begins inside a long ball dropout switches via the `prox_team` blend or waits until the ball reappears.
3. **Inbounds stoppages without `oob` / `set_piece`**: If play stops inside the pitch without `near_line` or a Smart Tagging `set_piece` window and a player stands directly over the ball (`nearest < 3.0 m`), `dead_cand` does not fire until someone steps away (`idle_distance = 3.0 m`).
4. **Short-gap chord noise**: Under heavy ball dropout (`stress40`), short linear interpolations between noisy endpoints can still generate occasional spurious velocity corners when the ball reappears (`touch P = 0.465` under `stress40` vs `0.676` under `standard`).
5. **Simulator assumptions**: `simulate.py` starts steals by teleporting the ball to a new holder without an initial kick spike and places players somewhat synthetically; real ATD failure modes (persistent shadows, multi-player box crowds, exact ball-loss correlation with aerial duels) can only be checked locally on real data.

---

## Commands for the Human to Run Locally on Real Data

```bash
# 0. Verify synthetic tests and invariants first (no data required)
python -I solution/test_pipeline.py
python -I solution/test_touch.py

# 1. Inspect aggregate touch & control statistics on the real match (prints aggregates only)
python solution/touch_report.py

# 2. Generate predictions with and without Approach A
python solution/run.py --touch --out evaluation/results/approach_a_touch.csv
python solution/run.py --no-touch --out evaluation/results/approach_a_notouch.csv

# 3. Validate submission format and run correctness check (public 5 min only)
python evaluation/validate_submission.py evaluation/results/approach_a_touch.csv
python evaluation/validate_submission.py evaluation/results/approach_a_notouch.csv
python evaluation/run_evaluation.py
```

If `solution/inspect_submission.py` is available on your local branch/tree, review tactical-video disagreements between `--touch` and `--no-touch`:

```bash
python solution/inspect_submission.py evaluation/results/approach_a_touch.csv --vs evaluation/results/approach_a_notouch.csv --list-diffs
```
