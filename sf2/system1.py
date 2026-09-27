"""System 1: the stage-1 laya-vision checkpoint plays one character (player 1) against the arcade CPU.

Each decision, when the fighter can act (standing or crouching, on the ground): the model gets the two frames
(4 frames apart) and the RAM note built exactly as in training (sf2.vs_sweep.note), and answers the outcome
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
from .policy import make_state
from .short_memory import situation
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
    decisions: int = 0
    dealt: int = 0
    taken: int = 0
    log: List[Dict] = field(default_factory=list)


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
        self.memory = None                         # a ShortMemory, if System 1 plays with one
        self.attacks = [a for a in actions(me) if a not in MOVEMENT]
        self.questions = {a: outcome_question(a) for a in self.attacks}

    def decide(self, prev: np.ndarray, cur: np.ndarray, text: str, sit=None) -> Dict:
        """``sit``: the situation key (sf2.short_memory.situation); with a memory, it adjusts each attack's score."""
        if self.agent is None:
            a = self.rng.choice(self.attacks)
            return {"action": a, "best": a, "p_hit": None, "predicted": "hit", "probs": {}}
        ans = self.agent.predict(make_state(prev, cur, text), self.questions)["answers"]
        probs = {a: ans[a]["probabilities"] for a in self.attacks}
        p_hit = {a: probs[a]["hit"] for a in self.attacks}
        score = {a: self.memory.score(a, sit, p) for a, p in p_hit.items()} if self.memory else p_hit
        best = max(self.attacks, key=lambda a: score[a])
        action = best if score[best] >= self.threshold else "forward"
        out = {"action": action, "best": best, "p_hit": p_hit[best], "score": score[best],
               "predicted": max(probs[best], key=probs[best].get) if action != "forward" else "none",
               "probs": p_hit}
        if self.memory:
            out["memory"] = dict(zip(("hits", "tries"), self.memory.counts(best, sit)))
        return out


def _can_act(r: Dict[str, int]) -> bool:
    return r["p1_state"] in (0, 2) and r["p1_y"] == GROUND_Y


def _run(bridge, frames: List[List[str]]):
    """Run the frames; return (rows, frame n-4, frame n) so the next decision has its two images."""
    n = len(frames)
    obs = bridge.run(frames, caps=[n - 4, n])
    return [dict(zip(NAMES, r)) for r in obs.rams], obs.images[n - 4], obs.images[n]


def play_round(bridge, s1: System1, opp: str, state: bytes, rng: random.Random, img_dir: Optional[str],
               tag: str) -> Round:
    """One round from the savestate (round 1, first controllable frame) until the ROM's round result is set."""
    bridge.load_state(state)
    rows, prev, cur = _run(bridge, [[]] * (4 + rng.randrange(40)))    # a random start so rounds differ
    rnd = Round()
    while rnd.frames < MAX_FRAMES:
        r = rows[-1]
        if r["result"]:
            rnd.result = RESULT.get(r["result"], "result_%d" % r["result"])
            break
        if not _can_act(r):
            rows, prev, cur = _run(bridge, [[]] * WAIT)
            rnd.frames += WAIT
            continue
        me_row = view(r, 1)
        side = "left" if r["p1_x"] < r["p2_x"] else "right"
        text = note(s1.me, opp, me_row, side)
        sit = situation(abs(r["p2_x"] - r["p1_x"]), r["p2_state"])
        d = s1.decide(prev, cur, text, sit)
        # walks resolve F/B from x; other moves from the ROM's facing byte, which the stick is mirrored by
        facing_right = (r["p1_x"] < r["p2_x"]) if d["action"] in MOVEMENT else r["p1_facing"] == 0x40
        steps = [t for toks, n in actions(s1.me)[d["action"]] for t in [toks] * n]
        frames = [physical(t, facing_right, PAD) for t in steps]
        frames += [[]] * max(0, 4 - len(frames))
        seen, prev_img, cur_img = _run(bridge, frames)
        live = seen[1:]
        if d["action"] not in MOVEMENT:        # wait until it can act again, reading what the attack did
            for _ in range(MAX_RECOVER // WAIT):
                if _can_act(live[-1]) or live[-1]["result"]:
                    break
                more, prev_img, cur_img = _run(bridge, [[]] * WAIT)
                live += more[1:]
        vrows = [view(x, 1) for x in [r] + live]
        actual = outcome(vrows, d["action"])["outcome"] if d["action"] not in MOVEMENT else "none"
        dealt = sum(max(0, a["p2_life"] - b["p2_life"]) for a, b in zip([r] + live, live) if b["p2_life"] < 200)
        taken = sum(max(0, a["p1_life"] - b["p1_life"]) for a, b in zip([r] + live, live) if b["p1_life"] < 200)
        if s1.memory is not None and d["action"] not in MOVEMENT:
            s1.memory.record(d["action"], sit, actual)     # the result, remembered for the next decisions
        rnd.dealt += dealt
        rnd.taken += taken
        entry = dict(d, t=rnd.frames, note=text, side=side, gap=abs(r["p2_x"] - r["p1_x"]), actual=actual,
                     dealt=dealt, taken=taken, p2_state=r["p2_state"], frames=len(live))
        if img_dir:
            base = "%s_%05d" % (tag, rnd.frames)
            save_png(prev, os.path.join(img_dir, base + "_prev.png"))
            save_png(cur, os.path.join(img_dir, base + "_now.png"))
            entry["images"] = [base + "_prev.png", base + "_now.png"]
        rnd.log.append(entry)
        rnd.decisions += 1
        rnd.frames += len(live)
        rows, prev, cur = [r] + live, prev_img, cur_img
    return rnd
