"""System 1: the stage-1 laya-vision checkpoint plays one character (player 1) against the arcade CPU.

Each decision, when the fighter can act (standing or crouching, on the ground): the model gets the two frames
(4 frames apart) and its prompt: the RAM note built exactly as in training (sf2.data.vs_sweep.note) followed by the short
memory against this opponent (sf2.memory.prompt_text; nothing when it is empty), and answers the outcome
question for every attack in one predict call (the images are encoded once). It plays the attack with the highest
P(hit) if that is at least ``threshold``; otherwise it walks forward. After an attack it waits until the fighter can
act again and reads what really happened from RAM (hit / whiff / blocked), so every attack is also a check of the
model's prediction against the live game.

With an advisor (sf2.system1.advisor) the pick is text laya's instead: laya-vision reads the note only (as it was trained),
rates every move, and text laya chooses from the best-rated ones and the moves the short memory names, following
the memory. The threshold rule above is then unused.

The stage-1 model only knows what its own moves do to a still opponent: it has no notion of the CPU's attacks,
blocking or anti-air. This is the real-play test of that skill, not a finished player.
"""
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import PAD
from ..data.dataset import save_png
from .game_log import action_entry, game_entry
from .advice import FORWARD, opp_doing
from .advisor import choose
from .game_log import name as state_name
from .opp_moves import OppMoveTracker
from ..system2.memory import MAX_PROMPT_LESSONS, prompt_text
from ..data.vs_defense import BLOCKS
from ..data.vs_defense import outcome as block_outcome
from .policy import make_state
from ..emu.vs import GROUND_Y, NAMES, physical, view
from ..vocab import bar, range_of
from ..data.vs_sweep import MOVEMENT, actions, note, outcome, outcome_question

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


def choices(me: str) -> List[str]:
    """Every move System 1 can pick: the attacks and blocks laya-vision rates, and walking in (FORWARD). Other
    movement (back, crouch, jumps, idle) is never chosen, so advice about it cannot be followed."""
    return [a for a in actions(me) if a not in MOVEMENT and a not in BLOCKS] + list(BLOCKS) + [FORWARD]


class System1:
    def __init__(self, model: Optional[str], me: str, threshold: float = 0.5, device: Optional[str] = None,
                 seed: int = 0, advisor=None):
        """``model`` None: the explorer, a uniformly random move of all the character's actions every decision (for
        live data: every move gets real tries, labelled from RAM)."""
        self.agent, self.rng = None, random.Random(seed)
        if model:
            import laya

            self.agent = laya.load_vlm(model, device=device)
            size = self.agent.model.prep.image_size
            if size != 256:
                raise SystemExit("%s sees %d px images; the stage-1 data is 256x256" % (model, size))
        self.me, self.threshold = me, threshold
        self.advisor = advisor     # sf2.advisor.Advisor: text laya picks from laya-vision's ratings + the short memory
        self.advice_on = True      # False: text laya still picks, told "Advice: none" (the A/B control)
        self.short = None          # the short memory vs the current opponent (sf2.system2.memory), goes into laya's prompt
        self.attacks = [a for a in actions(me) if a not in MOVEMENT and a not in BLOCKS]
        self.blocks = list(BLOCKS)
        self.questions = {a: outcome_question(a) for a in self.attacks + self.blocks}

    def lessons(self) -> List[str]:
        if not self.advice_on:
            return []
        return [les["text"] for les in (self.short or {}).get("lessons", [])[:MAX_PROMPT_LESSONS]]

    def decide(self, prev: np.ndarray, cur: np.ndarray, text: str, situation: Optional[Tuple] = None) -> Dict:
        """``situation`` (range, what he is doing, my bar, his bar) is needed with an advisor."""
        if self.agent is None:        # the explorer: any of the character's moves, uniformly (data, not play)
            a = self.rng.choice(list(actions(self.me)))
            return {"action": a, "best": a, "p_hit": None, "predicted": "none", "probs": {}}
        ans = self.agent.predict(make_state(prev, cur, text), self.questions)["answers"]
        probs = {a: ans[a]["probabilities"] for a in self.attacks + self.blocks}
        # every action is scored by the outcome that makes it worth doing: an attack by P(hit), a block by P(blocked)
        score = {a: probs[a]["hit"] for a in self.attacks}
        score.update({b: probs[b].get("blocked", 0.0) for b in self.blocks})
        best = max(score, key=score.get)
        if self.advisor is not None:
            c = choose(self.advisor, situation, score, self.lessons(), self.attacks + self.blocks)
            action = c["action"]
            return dict(c, best=best, p_hit=score.get(action),
                        predicted=max(probs[action], key=probs[action].get) if action in probs else "none",
                        probs=score)
        action = best if score[best] >= self.threshold else "forward"
        return {"action": action, "best": best, "p_hit": score[best],
                "predicted": max(probs[best], key=probs[best].get) if action != "forward" else "none",
                "probs": score}


def _can_act(r: Dict[str, int]) -> bool:
    return r["p1_state"] in (0, 2) and r["p1_y"] == GROUND_Y


def _run(bridge, frames: List[List[str]]):
    """Run the frames; return (rows, frame n-4, frame n) so the next decision has its two images."""
    n = len(frames)
    obs = bridge.run(frames, caps=[n - 4, n])
    return [dict(zip(NAMES, r)) for r in obs.rams], obs.images[n - 4], obs.images[n]


RATING_SHORT = {"likely works": "works", "may work": "may", "likely fails": "fails", None: "walk"}


def decision_line(entry: Dict) -> str:
    """One decision as a short line for ``tail -f out/live/decisions.log`` in a terminal:
    time, the moment, laya-vision's rating of each move on the shortlist, text laya's pick and why."""
    text = entry.get("advice_text") or ""
    m = re.search(r"He is (up close|at mid range|far away) and (\w+)\. My bar is (\w+), his bar is (\w+)", text)
    where = "%-5s %-9s me:%-4s him:%-4s" % ({"up close": "close", "at mid range": "mid", "far away": "far"}[m.group(1)],
                                           m.group(2), m.group(3), m.group(4)) if m else text[:36]
    sl, probs = entry.get("shortlist") or {}, entry.get("advice_probs") or {}
    rated = " ".join("%s:%s" % (mv, RATING_SHORT.get(r, r)) for mv, r in sl.items() if mv != "forward")
    why = {"soft": "plan", "hard": "plan", "vision": "rating", "walk": "walk in"}.get(entry.get("rule"),
                                                                                    entry.get("rule"))
    return "%s  %s | %-44s | -> %s %.0f%% (%s)" % (time.strftime("%H:%M:%S"), where, rated[:44], entry.get("action"),
                                                    100 * probs.get(entry.get("action"), 0), why)


def write_live(path: str, entry: Dict) -> None:
    """The latest decision for a live viewer (scripts/brain_panel.py): written to a temp file, then renamed, so a
    reader never sees half a file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(entry, f)
    os.replace(tmp, path)
    if entry.get("shortlist"):                      # with the advisor: one line per decision to tail
        with open(os.path.join(os.path.dirname(path), "decisions.log"), "a") as f:
            f.write(decision_line(entry) + "\n")


def play_round(bridge, s1: System1, opp: str, state: Optional[bytes], rng: random.Random, img_dir: Optional[str],
               game: int, live_path: Optional[str] = None, echo: bool = False) -> Round:
    """One game (a round) until the ROM's round result is set: from ``state`` (a savestate), or with ``state`` None
    from wherever the game is now (arcade play from power-on). Every action is logged from its decision to the next
    decision, so the opponent's reaction (and a punish while System 1 cannot act) is part of it."""
    if state is not None:
        bridge.load_state(state)
        rows, prev, cur = _run(bridge, [[]] * (4 + rng.randrange(40)))    # a random start so games differ
    else:
        rows, prev, cur = _run(bridge, [[]] * 4)
    rnd = Round()
    opp_moves = OppMoveTracker()   # holds each entry until the attack episodes its window overlaps are classified
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
            rnd.log.extend(opp_moves.add(_close(game, s1.me, opp, pending), pending[0], pending[1]))
        d, text = _decide(s1, opp, r, prev, cur)
        if live_path or echo:
            _show(live_path, echo, game, rnd.frames, opp, r, d)
        images = _save_images(img_dir, game, rnd.frames, prev, cur) if img_dir else None
        live, prev, cur, actual = _act(bridge, s1.me, r, d)
        pending = (r, list(live), dict(d, prompt=text), actual, rnd.frames, images)
        rnd.frames += len(live)
        rows = [r] + live
    if pending:
        rnd.log.extend(opp_moves.add(_close(game, s1.me, opp, pending), pending[0], pending[1]))
    rnd.log.extend(opp_moves.flush())
    rnd.summary = game_entry(game, rnd.result, rnd.frames, rows[-1], rnd.log)
    return rnd


def _decide(s1: System1, opp: str, r: Dict[str, int], prev, cur) -> Tuple[Dict, str]:
    """(decision, the note laya-vision read). Without an advisor the short memory goes into laya-vision's note (the
    old path); with one, laya-vision reads the note only (as trained) and the memory goes to text laya."""
    side = "left" if r["p1_x"] < r["p2_x"] else "right"
    text = note(s1.me, opp, view(r, 1), side)
    if s1.advisor is None:
        text = prompt_text(text, s1.short)
        return s1.decide(prev, cur, text), text
    return s1.decide(prev, cur, text, situation(r)), text


def _show(live_path: Optional[str], echo: bool, game: int, frame: int, opp: str, r: Dict[str, int], d: Dict) -> None:
    """The decision for the brain panel (``live_path``) and, with ``echo``, one console line."""
    entry = {"t": time.time(), "game": game, "frame": frame, "opp": opp,
             "my_life": r["p1_life"], "opp_life": r["p2_life"], "timer": r["timer"],
             **{k: d.get(k) for k in ("action", "advice_text", "shortlist", "advice_probs", "rule", "follows_rule")}}
    if live_path:
        write_live(live_path, entry)
    if echo and entry.get("shortlist"):         # the console shows every decision (for watching/recording)
        print(decision_line(entry), flush=True)


def _save_images(img_dir: str, game: int, frame: int, prev, cur) -> List[str]:
    base = "g%02d_%05d" % (game, frame)
    save_png(prev, os.path.join(img_dir, base + "_prev.png"))
    save_png(cur, os.path.join(img_dir, base + "_now.png"))
    return [base + "_prev.png", base + "_now.png"]


def _act(bridge, me: str, r: Dict[str, int], d: Dict):
    """Press the decided move, wait until System 1 can act again (reading what an attack did), and judge it:
    (rows since the decision, prev frame, current frame, what happened)."""
    # walks resolve F/B from x; other moves from the ROM's facing byte, which the stick is mirrored by
    facing_right = (r["p1_x"] < r["p2_x"]) if d["action"] in MOVEMENT else r["p1_facing"] == 0x40
    steps = [t for toks, n in actions(me)[d["action"]] for t in [toks] * n]
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
    if d["action"] in BLOCKS or d["action"] in MOVEMENT:   # blocks and moves: what happened to MY health
        actual = block_outcome([view(x, 2) for x in [r] + live])["outcome"]
    else:
        actual = outcome([view(x, 1) for x in [r] + live], d["action"])["outcome"]
    return live, prev, cur, actual


ADVICE_KEYS = ("advice_text", "shortlist", "advice_probs", "rule_answers", "rule", "follows_rule")


def situation(r: Dict[str, int]) -> Tuple[str, str, str, str]:
    """The moment in text laya's words (sf2.advice.situation_text), from the RAM row at the decision."""
    gap = abs(r["p2_x"] - r["p1_x"])
    doing = opp_doing({"opp_air": r["p2_y"] != GROUND_Y, "opp_state": state_name(r["p2_state"])})
    return range_of(gap), doing, bar(r["p1_life"]), bar(r["p2_life"])


def _close(game: int, me: str, opp: str, pending) -> Dict:
    before, rows, d, actual, frame, images = pending
    entry = action_entry(game, frame, me, opp, before, rows or [before], d, actual)
    entry["prompt"] = d["prompt"]                 # exactly what laya-vision read
    for k in ADVICE_KEYS:                         # with an advisor: what text laya read, and what it picked
        if k in d:
            entry[k] = d[k]
    if images:
        entry["images"] = images
    return entry
