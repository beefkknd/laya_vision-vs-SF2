"""System 1: the stage-1 laya-vision checkpoint plays one character (player 1) against the arcade CPU.

Each decision, when the fighter can act (standing or crouching, on the ground): the model gets the two frames
(4 frames apart) and its prompt: the RAM note built exactly as in training (sf2.vs_sweep.note) followed by the short
memory against this opponent (sf2.memory.prompt_text; nothing when it is empty), and answers the outcome
question for every attack in one predict call (the images are encoded once). It plays the attack with the highest
P(hit) if that is at least ``threshold``; otherwise it walks forward. After an attack it waits until the fighter can
act again and reads what really happened from RAM (hit / whiff / blocked), so every attack is also a check of the
model's prediction against the live game.

The stage-1 model only knows what its own moves do to a still opponent: it has no notion of the CPU's attacks,
blocking or anti-air. This is the real-play test of that skill, not a finished player.
"""
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .config import PAD
from .dataset import save_png
from .game_log import action_entry, game_entry
from .memory import prompt_text
from .policy import make_state
from .vs import GROUND_Y, NAMES, physical, view
from .vs_sweep import MOVEMENT, actions, note, outcome, outcome_question

WAIT = 4              # idle frames per step while the fighter cannot act (the 4-frame prev/now gap)
MAX_RECOVER = 90      # frames to wait after an attack for the fighter to be able to act again
MAX_FRAMES = 12000    # a round cannot last longer (99 s clock plus the KO)
RESULT = {1: "win", 2: "loss", 0xFF: "draw"}


@dataclass
class Round:
    result: str = "unfinished"
    frames: int = 0
    log: List[Dict] = field(default_factory=list)      # sf2.game_log.action_entry per action
    summary: Dict = field(default_factory=dict)         # sf2.game_log.game_entry


class System1:
    def __init__(self, model: Optional[str], me: str, threshold: float = 0.5, device: Optional[str] = None,
                 seed: int = 0):
        """``model`` None: the baseline, a random attack every decision (same loop, no model)."""
        self.agent, self.rng = None, random.Random(seed)
        if model:
            import laya

            self.agent = laya.load_vlm(model, device=device)
            size = self.agent.model.prep.image_size
            if size != 256:
                raise SystemExit("%s sees %d px images; the stage-1 data is 256x256" % (model, size))
        self.me, self.threshold = me, threshold
        self.short = None          # the short memory vs the current opponent (sf2.memory), goes into laya's prompt
        self.attacks = [a for a in actions(me) if a not in MOVEMENT]
        self.questions = {a: outcome_question(a) for a in self.attacks}

    def decide(self, prev: np.ndarray, cur: np.ndarray, text: str) -> Dict:
        if self.agent is None:
            a = self.rng.choice(self.attacks)
            return {"action": a, "best": a, "p_hit": None, "predicted": "hit", "probs": {}}
        ans = self.agent.predict(make_state(prev, cur, text), self.questions)["answers"]
        probs = {a: ans[a]["probabilities"] for a in self.attacks}
        p_hit = {a: probs[a]["hit"] for a in self.attacks}
        best = max(self.attacks, key=lambda a: p_hit[a])
        action = best if p_hit[best] >= self.threshold else "forward"
        return {"action": action, "best": best, "p_hit": p_hit[best],
                "predicted": max(probs[best], key=probs[best].get) if action != "forward" else "none",
                "probs": p_hit}


def _can_act(r: Dict[str, int]) -> bool:
    return r["p1_state"] in (0, 2) and r["p1_y"] == GROUND_Y


def _run(bridge, frames: List[List[str]]):
    """Run the frames; return (rows, frame n-4, frame n) so the next decision has its two images."""
    n = len(frames)
    obs = bridge.run(frames, caps=[n - 4, n])
    return [dict(zip(NAMES, r)) for r in obs.rams], obs.images[n - 4], obs.images[n]


def play_round(bridge, s1: System1, opp: str, state: Optional[bytes], rng: random.Random, img_dir: Optional[str],
               game: int) -> Round:
    """One game (a round) until the ROM's round result is set: from ``state`` (a savestate), or with ``state`` None
    from wherever the game is now (arcade play from power-on). Every action is logged from its decision to the next
    decision, so the opponent's reaction (and a punish while System 1 cannot act) is part of it."""
    if state is not None:
        bridge.load_state(state)
        rows, prev, cur = _run(bridge, [[]] * (4 + rng.randrange(40)))    # a random start so games differ
    else:
        rows, prev, cur = _run(bridge, [[]] * 4)
    rnd = Round()
    pending = None          # (decision row, rows since, decision, actual, frame, images) of the last action
    while True:
        r = rows[-1]
        if r["result"] or rnd.frames >= MAX_FRAMES:
            rnd.result = RESULT.get(r["result"], "result_%d" % r["result"]) if r["result"] else "unfinished"
            break
        if not _can_act(r):
            rows, prev, cur = _run(bridge, [[]] * WAIT)
            rnd.frames += WAIT
            if pending:
                pending[1].extend(rows[1:])
            continue
        if pending:
            rnd.log.append(_close(game, s1.me, opp, pending))
        side = "left" if r["p1_x"] < r["p2_x"] else "right"
        text = prompt_text(note(s1.me, opp, view(r, 1), side), s1.short)
        d = s1.decide(prev, cur, text)
        images = None
        if img_dir:
            base = "g%02d_%05d" % (game, rnd.frames)
            save_png(prev, os.path.join(img_dir, base + "_prev.png"))
            save_png(cur, os.path.join(img_dir, base + "_now.png"))
            images = [base + "_prev.png", base + "_now.png"]
        # walks resolve F/B from x; other moves from the ROM's facing byte, which the stick is mirrored by
        facing_right = (r["p1_x"] < r["p2_x"]) if d["action"] in MOVEMENT else r["p1_facing"] == 0x40
        steps = [t for toks, n in actions(s1.me)[d["action"]] for t in [toks] * n]
        frames = [physical(t, facing_right, PAD) for t in steps]
        frames += [[]] * max(0, 4 - len(frames))
        seen, prev, cur = _run(bridge, frames)
        live = seen[1:]
        if d["action"] not in MOVEMENT:        # wait until it can act again, reading what the attack did
            for _ in range(MAX_RECOVER // WAIT):
                if _can_act(live[-1]) or live[-1]["result"]:
                    break
                more, prev, cur = _run(bridge, [[]] * WAIT)
                live += more[1:]
        actual = outcome([view(x, 1) for x in [r] + live], d["action"])["outcome"] if d["action"] not in MOVEMENT \
            else "none"
        pending = (r, list(live), dict(d, prompt=text), actual, rnd.frames, images)
        rnd.frames += len(live)
        rows = [r] + live
    if pending:
        rnd.log.append(_close(game, s1.me, opp, pending))
    rnd.summary = game_entry(game, rnd.result, rnd.frames, rows[-1], rnd.log)
    return rnd


def _close(game: int, me: str, opp: str, pending) -> Dict:
    before, rows, d, actual, frame, images = pending
    entry = action_entry(game, frame, me, opp, before, rows or [before], d, actual)
    entry["prompt"] = d["prompt"]                 # exactly what laya read: the note and the short memory
    if images:
        entry["images"] = images
    return entry
