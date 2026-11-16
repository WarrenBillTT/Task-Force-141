"""A synthetic match in the real ATD file format, with a known possession ground truth.

It exists so the pipeline can be tested end to end WITHOUT the gated match data. It is a smoke test
and a regression test, not evidence about real-match accuracy: it knows nothing about how the real
tracker fails. Don't tune parameters on it.

Per segment: a team holds the ball, passing between its players; a steal teleports the ball to a
new team's player; a dead-ball stretch sends the ball over a line, leaves it still, then restarts.
The ball track gets noise, spikes and dropouts; players get dropouts.
"""

import numpy as np

SLOTS = 46
LIVE_PER_TEAM = 11


def _clip(xy, pitch):
    return np.clip(xy, [0.5, 0.5], [pitch[0] - 0.5, pitch[1] - 0.5])


def _team_positions(rng, centre, pitch, holder, spread, away_from=None):
    pos = centre + rng.normal(0, spread, size=(LIVE_PER_TEAM, 2))
    if holder is not None:
        pos[holder] = centre + rng.normal(0, 0.3, size=2)
    if away_from is not None:  # opponents never stand on the ball
        for i in range(LIVE_PER_TEAM):
            while np.hypot(*(pos[i] - away_from)) < 3.0:
                pos[i] = away_from + rng.normal(0, spread, size=2)
    return _clip(pos, pitch)


def simulate(n_frames=20000, blocks=((1, 500, 9500), (2, 10500, 19500)), seed=0,
             pitch=(105.3, 68.0), missing_rate=None, jitter_std=None, spike_rate=None,
             id_swap_rate=0.0, deflection_rate=0.0):
    """Return dict(ball[N,2], players[N,46,2], team[46], gt[N] letters, blocks, touches) in metres.

    Optional stress parameters (`missing_rate`, `jitter_std`, `spike_rate`, `id_swap_rate`,
    `deflection_rate`) default to None/0.0 and use a separate RNG stream so default outputs and
    seeds remain bit-for-bit identical.
    """
    rng = np.random.default_rng(seed)
    team = np.array([s % 2 for s in range(SLOTS)], dtype=np.int8)
    live = {0: [s for s in range(SLOTS) if team[s] == 0][:LIVE_PER_TEAM],
            1: [s for s in range(SLOTS) if team[s] == 1][:LIVE_PER_TEAM]}

    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, SLOTS, 2), np.nan)
    gt = np.full(n_frames, "X", dtype="<U1")
    touches = []
    pass_events = []

    for _, first, last in blocks:
        t = first
        ball_pos = np.array([pitch[0] / 2, pitch[1] / 2])
        owner = int(rng.integers(2))
        while t <= last:
            dead = rng.random() < 0.3 and t > first
            if dead:
                dur = int(rng.integers(100, 300))
                dur = min(dur, last + 1 - t)
                side_y = -1.0 if rng.random() < 0.5 else pitch[1] + 1.0
                spot = np.array([np.clip(ball_pos[0], 5, pitch[0] - 5), side_y])
                edge = np.array([spot[0], 0.3 if side_y < 0 else pitch[1] - 0.3])
                for k in range(dur):
                    if k < 20:
                        b = ball_pos + (spot - ball_pos) * (k / 20.0)
                    elif k < dur - 15:
                        b = spot
                    else:
                        b = edge
                    ball[t + k] = b
                    gt[t + k] = "D"
                near = np.array([b[0], np.clip(b[1], 0, pitch[1])])
                for side in (0, 1):
                    pos = _team_positions(rng, near, pitch, None, 10.0, away_from=None)
                    for j, s in enumerate(live[side]):
                        players[t:t + dur, s] = pos[j] + np.cumsum(
                            rng.normal(0, 0.03, size=(dur, 2)), axis=0)
                ball_pos = edge
                owner = int(rng.integers(2))
                t += dur
                continue

            dur = int(max(75, rng.exponential(300)))
            dur = min(dur, last + 1 - t)
            holder = int(rng.integers(LIVE_PER_TEAM))
            own = _team_positions(rng, ball_pos, pitch, holder, 12.0)
            opp = _team_positions(rng, ball_pos, pitch, None, 10.0, away_from=ball_pos)
            ramp = np.arange(dur)[:, None] / 25.0
            walk = {s: np.cumsum(rng.normal(0, 0.04, size=(dur, 2)), axis=0)
                    + rng.normal(0, 1.5, size=2) * ramp for s in range(SLOTS)}
            for j, s in enumerate(live[owner]):
                players[t:t + dur, s] = own[j] + walk[s]
            for j, s in enumerate(live[1 - owner]):
                players[t:t + dur, s] = opp[j] + walk[s]
            players[t:t + dur] = np.clip(players[t:t + dur], [0.5, 0.5],
                                         [pitch[0] - 0.5, pitch[1] - 0.5])
            gt[t:t + dur] = "HA"[owner]

            k = 0
            while k < dur:
                hold = int(rng.integers(50, 100))
                nxt = int(rng.integers(LIVE_PER_TEAM))
                while nxt == holder:
                    nxt = int(rng.integers(LIVE_PER_TEAM))
                fly = int(rng.integers(12, 20))
                for f in range(min(hold + fly, dur - k)):
                    a = players[t + k + f, live[owner][holder]]
                    if f < hold:
                        ball[t + k + f] = a + rng.normal(0, 0.15, size=2)
                    else:
                        c = players[t + k + f, live[owner][nxt]]
                        ball[t + k + f] = a + (c - a) * ((f - hold) / float(fly))
                if k + hold + 2 < dur:
                    touches.append((int(t + k + hold), int(owner), int(live[owner][holder]), "kick"))
                if k + hold + fly < dur:
                    touches.append((int(t + k + hold + fly), int(owner), int(live[owner][nxt]), "receive"))
                    pass_events.append((int(t + k + hold), int(t + k + hold + fly), int(owner)))
                k += hold + fly
                holder = nxt
            ball_pos = ball[t + dur - 1].copy()
            t += dur
            owner = 1 - owner  # next alive segment is a change of possession (a steal)

    # Optional pre-noise mid-flight opponent pass-bys / deflections (uses separate RNG)
    stress_rng = None
    if (missing_rate is not None or jitter_std is not None or spike_rate is not None
            or id_swap_rate > 0.0 or deflection_rate > 0.0):
        stress_rng = np.random.default_rng(int(seed) + 100000)

    if deflection_rate > 0.0 and pass_events:
        unused = {0: [s for s in range(SLOTS) if team[s] == 0 and s not in live[0]],
                  1: [s for s in range(SLOTS) if team[s] == 1 and s not in live[1]]}
        for ks, ke, p_owner in pass_events:
            if ke - ks < 10 or stress_rng.random() >= deflection_rate:
                continue
            opp = 1 - p_owner
            opp_pool = live[opp] if not unused[opp] else unused[opp]
            opp_slot = int(opp_pool[int(stress_rng.integers(len(opp_pool)))])
            mid = (ks + ke) // 2
            p0 = ball[ks].copy()
            p1 = ball[ke].copy()
            vec = p1 - p0
            dist = float(np.hypot(*vec))
            if dist < 4.0:
                continue
            perp = np.array([-vec[1], vec[0]]) / dist
            sign = -1.0 if stress_rng.random() < 0.5 else 1.0
            if stress_rng.random() < 0.5:
                # Close opponent pass-by (within ~1.2 m of flight path, no touch)
                mid_pos = _clip(0.5 * (p0 + p1) + sign * 1.2 * perp, pitch)
                span = slice(max(0, mid - 5), min(n_frames, mid + 6))
                players[span, opp_slot] = mid_pos
            else:
                # Sharp deflection off opponent (kinks trajectory, does NOT change gt)
                kink = _clip(0.5 * (p0 + p1) + sign * stress_rng.uniform(1.8, 3.0) * perp, pitch)
                players[max(0, mid - 3):min(n_frames, mid + 4), opp_slot] = kink
                for f in range(ks, mid + 1):
                    ball[f] = p0 + (kink - p0) * ((f - ks) / float(max(1, mid - ks)))
                for f in range(mid, ke + 1):
                    ball[f] = kink + (p1 - kink) * ((f - mid) / float(max(1, ke - mid)))

    # --- observation noise, spikes, dropouts ---
    ball = ball + rng.normal(0, 0.3, size=ball.shape)
    spikes = rng.random(n_frames) < 0.003
    ball[spikes] += rng.choice([-1, 1], size=(int(spikes.sum()), 2)) * rng.uniform(25, 40, (int(spikes.sum()), 2))
    t = 0
    while t < n_frames:
        if rng.random() < 0.002:
            gap = int(rng.geometric(1 / 30.0))
            ball[t:t + gap] = np.nan
            t += gap
        if rng.random() < 0.0003:
            gap = int(rng.integers(75, 200))
            ball[t:t + gap] = np.nan
            t += gap
        t += 1
    players = players + rng.normal(0, 0.15, size=players.shape)
    ball = np.clip(ball, [-3.0, -3.0], [pitch[0] + 3.0, pitch[1] + 3.0])
    players[rng.random(players.shape[:2]) < 0.02] = np.nan

    # --- optional stress perturbations (separate RNG, default leaves arrays untouched) ---
    if stress_rng is not None:
        if jitter_std is not None and jitter_std > 0.0:
            ball = ball + stress_rng.normal(0, float(jitter_std), size=ball.shape)
            players = players + stress_rng.normal(0, 0.5 * float(jitter_std), size=players.shape)
        if spike_rate is not None and spike_rate > 0.0:
            ex_spikes = stress_rng.random(n_frames) < float(spike_rate)
            n_sp = int(ex_spikes.sum())
            if n_sp:
                ball[ex_spikes] += stress_rng.choice([-1, 1], size=(n_sp, 2)) * stress_rng.uniform(20, 35, (n_sp, 2))
        ball = np.clip(ball, [-3.0, -3.0], [pitch[0] + 3.0, pitch[1] + 3.0])

        if id_swap_rate > 0.0:
            same_slots = {0: [s for s in range(SLOTS) if team[s] == 0],
                          1: [s for s in range(SLOTS) if team[s] == 1]}
            for _, first, last in blocks:
                tt = first + 25
                while tt <= last - 25:
                    if stress_rng.random() < float(id_swap_rate):
                        sd = int(stress_rng.integers(2))
                        s_a = int(live[sd][int(stress_rng.integers(len(live[sd])))])
                        s_b = int(same_slots[sd][int(stress_rng.integers(len(same_slots[sd])))])
                        if s_a != s_b:
                            players[tt:last + 1, [s_a, s_b]] = players[tt:last + 1, [s_b, s_a]]
                    tt += 1

        if missing_rate is not None:
            graded = np.zeros(n_frames, dtype=bool)
            for _, first, last in blocks:
                graded[first:last + 1] = True
            graded_idx = np.flatnonzero(graded)
            if len(graded_idx) > 0:
                cur_miss = float(np.mean(np.isnan(ball[graded_idx, 0])))
                tries = 0
                while cur_miss < float(missing_rate) and tries < 2000:
                    tries += 1
                    start = int(stress_rng.choice(graded_idx))
                    gap = int(stress_rng.integers(45, 110)) if stress_rng.random() < 0.25 else int(
                        stress_rng.geometric(1 / 25.0)
                    )
                    ball[start:min(n_frames, start + gap)] = np.nan
                    cur_miss = float(np.mean(np.isnan(ball[graded_idx, 0])))

    return {"ball": ball.astype(np.float32), "players": players.astype(np.float32),
            "team": team, "gt": gt, "blocks": list(blocks), "touches": touches}


def write_atd(sim, raw_path, meta_path, pitch=(105.3, 68.0)):
    """Write the simulation as ATD txt + metadata xml, in the real format (0-1 coordinates)."""
    scale = np.array(pitch)

    def fmt(xy):
        if np.isnan(xy[0]) or np.isnan(xy[1]):
            return "NaN,NaN,NaN"
        return "{:.5f},{:.5f},0.0".format(xy[0] / scale[0], xy[1] / scale[1])

    with open(raw_path, "w") as handle:
        for i in range(len(sim["ball"])):
            groups = ";".join(fmt(p) for p in sim["players"][i])
            handle.write("{}:{}:{}\n".format(i + 1, groups, fmt(sim["ball"][i])))

    players = "".join('<Player id="P{}" teamId="{}"/>'.format(s, "HOME" if t == 0 else "AWAY")
                      for s, t in enumerate(sim["team"]))
    with open(meta_path, "w") as handle:
        handle.write('<Metadata><Teams><Team id="HOME"><Name>Home</Name></Team>'
                     '<Team id="AWAY"><Name>Away</Name></Team></Teams>'
                     '<Players>{}</Players></Metadata>'.format(players))
