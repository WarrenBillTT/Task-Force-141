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
             pitch=(105.3, 68.0)):
    """Return dict(ball[N,2], players[N,46,2], team[46], gt[N] letters, blocks) in metres."""
    rng = np.random.default_rng(seed)
    team = np.array([s % 2 for s in range(SLOTS)], dtype=np.int8)
    live = {0: [s for s in range(SLOTS) if team[s] == 0][:LIVE_PER_TEAM],
            1: [s for s in range(SLOTS) if team[s] == 1][:LIVE_PER_TEAM]}

    ball = np.full((n_frames, 2), np.nan)
    players = np.full((n_frames, SLOTS, 2), np.nan)
    gt = np.full(n_frames, "X", dtype="<U1")

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
                k += hold + fly
                holder = nxt
            ball_pos = ball[t + dur - 1].copy()
            t += dur
            owner = 1 - owner  # next alive segment is a change of possession (a steal)

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
    return {"ball": ball.astype(np.float32), "players": players.astype(np.float32),
            "team": team, "gt": gt, "blocks": list(blocks)}


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
