"""The one play loop every script uses: decide -> act -> record -> round/match bookkeeping.

``choose(env, prev, cur, text, teacher_dist)`` returns (action to execute, extra meta). The teacher's distribution
is computed on every frame whoever is playing, so every recorded row already carries the teacher's gold: the
teacher's own rollouts are the seed set, and the student's rollouts are DAgger data before any relabelling.
"""
import json
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .dataset import Writer
from .env import FightEnv
from .rollout import annotate
from .teacher import argmax, teacher_policy

Choose = Callable[[FightEnv, np.ndarray, np.ndarray, str, Dict[str, float]], Tuple[str, Dict]]


def play(env: FightEnv, choose: Choose, matches: int, writer: Optional[Writer] = None,
         max_decisions: Optional[int] = None, log_every: int = 1) -> Tuple[List[Dict], List[Dict]]:
    rows: List[Dict] = []
    rounds: List[Dict] = []
    t0 = time.time()
    for _ in range(matches):
        env.reset()
        step, match_rows = 0, []
        rnd = {"episode": env.episode, "round": env.round, "dmg_for": 0, "dmg_against": 0,
               "opening": getattr(env, "opening", None)}
        while True:
            prev, cur = env.prev_frame.copy(), env.frame.copy()
            text, f, ctx = env.text(), env.f, env.context()
            t_dist = teacher_policy(f, ctx, env.me)
            action, extra = choose(env, prev, cur, text, t_dist)
            frame0, round0, controllable = env.frame_no, env.round, env.controllable()
            res = env.act(action)
            meta = dict(episode=env.episode, round=round0, frame=frame0, frames=res.frames, action=action,
                        teacher_action=argmax(t_dist), my_state=f.my_state, my_hp=f.my_hp, opp_hp=f.opp_hp, dx=f.dx,
                        dmg_for=res.dmg_for, dmg_against=res.dmg_against, controllable=controllable, **extra)
            rec = writer.record(env.episode, step, prev, cur, text, t_dist, meta) if writer else {
                "episode": env.episode, "meta": meta}
            match_rows.append(rec)
            step += 1
            rnd["dmg_for"] += res.dmg_for
            rnd["dmg_against"] += res.dmg_against
            if res.round_over:
                rnd["winner"], rnd["end_frame"] = res.winner, env.frame_no
                rounds.append(rnd)
                if log_every and len(rounds) % log_every == 0:
                    print("match %d round %d: %s  dealt %d taken %d  (%d decisions, %.0fs)"
                          % (env.episode, round0, res.winner, rnd["dmg_for"], rnd["dmg_against"],
                             len(rows) + len(match_rows), time.time() - t0), flush=True)
                if not env.next_round():
                    break
                rnd = {"episode": env.episode, "round": env.round, "dmg_for": 0, "dmg_against": 0,
                       "opening": getattr(env, "opening", None)}
            elif env.done:
                break
            if max_decisions and len(rows) + len(match_rows) >= max_decisions:
                break
        result = {r["round"]: r["winner"] for r in rounds if r["episode"] == env.episode}
        for r in annotate(match_rows):
            r["meta"]["round_result"] = result.get(r["meta"]["round"])  # None = unfinished round
            if writer:
                writer.add(r)
        rows.extend(match_rows)
        if max_decisions and len(rows) >= max_decisions:
            break
    return rows, rounds


def save_rounds(path: str, rounds: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        for r in rounds:
            f.write(json.dumps(r) + "\n")
