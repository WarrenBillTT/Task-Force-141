"""Read the raw ATD (FIFA/EPTS text) into dense numpy arrays, in metres, on the video timeline.

Raw line:  "<frame_id>:<x,y,speed>;<x,y,speed>;...:<ball x,y,speed>"   (NaN where unobserved)
frame_id counts from 1, submissions from 0, so  video_frame = frame_id - 1  (see DATA.md).
"""

import xml.etree.ElementTree as ET

import numpy as np

HOME, AWAY = 0, 1


def team_of_each_track(meta_path):
    """Array over track slots: 0 home, 1 away. The first <Team> in the file is the home side."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(team_ids)))
    slots = [player.get("teamId") for player in root.iter("Player")]
    if not slots:
        raise ValueError("{}: no Player entries".format(meta_path))
    return np.array([HOME if t == team_ids[0] else AWAY for t in slots], dtype=np.int8)


def load_atd(raw_path, meta_path, n_frames, pitch):
    """Return (ball[N,2], players[N,S,2], speed[N,S], team[S]).

    Positions in metres, speed in m/s as the tracker reports it. Unobserved entries are NaN.
    """
    length, width = pitch
    team = team_of_each_track(meta_path)
    n_slots = len(team)
    scale = np.array([length, width], dtype=np.float32)

    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    players = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    speed = np.full((n_frames, n_slots), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            if not 0 <= i < n_frames:
                continue
            tokens = parts[1].replace(";", ",").strip(",").split(",")
            if tokens and tokens[0]:
                xys = np.array(tokens, dtype=np.float32).reshape(-1, 3)[:n_slots]
                players[i, :len(xys)] = xys[:, :2]
                speed[i, :len(xys)] = xys[:, 2]
            ball_tokens = parts[2].strip().split(",")
            if len(ball_tokens) >= 2:
                ball[i] = np.array(ball_tokens[:2], dtype=np.float32)

    return ball * scale, players * scale, speed, team
