"""Screen-only play of System 1 (docs/laya_text_only_plan.md, build step 3; arm S0 = T0 with every fact from the screen).

T0 (sf2/system1/system1.py play_round with the lookup table + text laya, no advice) reads one RAM row per step. Here
the same loop reads one screen frame per step instead: frame -> sf2.screen.reader.RoundReader -> ScreenFacts ->
sf2.system1.screen_words (the same note / situation / can-I-act / facing T0 builds from RAM) -> System1.decide (the
table ranks the note, text laya picks) -> buttons. The emulator handle (sf2.system1.screen_emu.ScreenEmu) gives frames
and takes buttons only: a RAM read raises.

Round: from the start savestate, T0's random start delay (same rng use: same seeds, same delays), then until the
reader says the round is over (the time-over rule, or the NEXT round visibly starting: a KO is seen late, and inputs
after a KO do nothing). Unknown sprites: the reader's UnknownLog saves the crop + side + score to <round>/unknown/; the
words answer "stand" (screen_words.label).

Saved per round (<out>/):
    inputs.json      start state id, start delay, the buttons of every frame since the state was loaded ("-" = none)
    captures.json    sha256 of every frame captured during play, by frame index (scripts/replay_score.py checks them)
    decisions.jsonl  per decision: frame k (the "now" image; k_prev the one before), the ScreenFacts, the moment, the
                     note and situation, the pick, the frames pressed, ms (reading, deciding), unknown sides
    frames/<k>.png   the two frames of every decision
    reads.jsonl      every frame the reader read: frame k, can I act, both labels / air, unknown sides, ms
    round.json       end frame, why it ended, decisions, unknowns, ms
No hp, no result here: those come from the replay (RAM, offline).
"""
import dataclasses
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..config import PAD
from ..data.vs_sweep import MOVEMENT, actions
from ..emu.vs import physical
from ..screen.reader import RoundReader
from ..screen.unknown_log import UnknownLog
from .screen_emu import ScreenEmu
from .screen_words import CAN_ACT, Moment, label, moment, note, players, situation
from .system1 import MAX_FRAMES, MAX_RECOVER, WAIT, System1

DECISION_KEYS = ("action", "best", "values", "advice_text", "shortlist", "advice_probs", "rule", "follows_rule")


class WrongFight(RuntimeError):
    """The reader identified other characters than the run's (the start state or the catalog is wrong)."""


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError("not JSON: %r" % type(o))


def pad_text(buttons: List[str]) -> str:
    return "+".join(buttons) if buttons else "-"


class Eyes:
    """The reader with timing: every frame read goes through here."""

    def __init__(self, unknown_dir: str):
        self.log = UnknownLog(unknown_dir)
        self.reader = RoundReader(log=self.log)
        self.reads, self.read_ms = 0, 0.0
        self.rows: List[Dict] = []         # one short line per frame read (reads.jsonl): can I act, both labels

    def read(self, frame: np.ndarray, k: int):
        t = time.perf_counter()
        facts = self.reader.feed(frame)
        ms = 1000 * (time.perf_counter() - t)
        self.reads += 1
        self.read_ms += ms
        me, him = players(facts)
        self.rows.append({"k": k, "can_act": label(me) in CAN_ACT and not me.in_air, "me": label(me),
                          "me_air": bool(me.in_air), "him": label(him), "him_air": bool(him.in_air),
                          "unknown": [f.side for f in (me, him) if f.unknown], "round": facts.round_state,
                          "ms": round(ms, 2)})
        return facts, ms


def _step(emu: ScreenEmu, frames: List[List[str]]) -> Tuple[np.ndarray, np.ndarray, int]:
    """Run ``frames`` (at least 4); (frame n - 4, frame n, n as an absolute index)."""
    n = len(frames)
    imgs = emu.run(frames, caps=[n - 4, n])
    k = emu.frame
    return imgs[k - 4], imgs[k], k


def buttons(me: str, action: str, m: Moment) -> List[List[str]]:
    """The frames of ``action`` mirrored as T0 does (screen_words.Moment.facing_right), at least 4."""
    steps = [t for toks, n in actions(me)[action] for t in [toks] * n]
    frames = [physical(t, m.facing_right(action in MOVEMENT), PAD) for t in steps]
    return frames + [[]] * max(0, 4 - len(frames))


def _save(img: np.ndarray, path: str) -> None:
    if not os.path.exists(path):
        Image.fromarray(img).save(path)


def play_screen_round(emu: ScreenEmu, s1: System1, me: str, opp: str, state: bytes, state_id: Dict, delay: int,
               out: str) -> Dict:
    """One round in screen-only mode; writes the round's record under ``out`` and returns round.json's content."""
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    eyes = Eyes(os.path.join(out, "unknown"))
    first = emu.load(state)
    eyes.read(first, 0)            # the reader locks the two characters on the round-start frame (identify() needs
    chars = tuple(eyes.reader.lock.chars)  # the start places: probe 2026-10-02, Ryu lost from ~frame 30 of the delay)
    prev, cur, k = _step(emu, [[]] * delay)
    facts, ms = eyes.read(cur, k)
    if chars != (me, opp):
        raise WrongFight("the reader sees %s, the run is %s vs %s" % (chars, me, opp))
    end, last, n_dec, n_unknown, dec_ms = "max_frames", None, 0, 0, []
    with open(os.path.join(out, "decisions.jsonl"), "w") as log:
        while emu.frame < MAX_FRAMES:
            if facts.round_state == "over":
                end = "new_round" if getattr(facts, "new_round", False) else "time_over"
                break
            m = moment(facts, last)
            last = m
            if not m.can_act:
                prev, cur, k = _step(emu, [[]] * WAIT)
                facts, ms = eyes.read(cur, k)
                continue
            text, sit = note(me, m), situation(m)
            t = time.perf_counter()
            d = s1.decide(prev, cur, text, sit)
            d_ms = 1000 * (time.perf_counter() - t)
            press = buttons(me, d["action"], m)
            unknown = [f.side for f in (facts.left, facts.right) if f.unknown]
            n_unknown += len(unknown)
            _save(prev, os.path.join(out, "frames", "%05d.png" % (k - 4)))
            _save(cur, os.path.join(out, "frames", "%05d.png" % k))
            entry = {"i": n_dec, "k": k, "k_prev": k - 4, "facts": dataclasses.asdict(facts),
                     "moment": dict(dataclasses.asdict(m), side=m.side, dx=m.dx, can_act=m.can_act,
                                    his_attacking=m.his_attacking, doing=m.doing),
                     "note": text, "situation": list(sit), **{x: d[x] for x in DECISION_KEYS if x in d},
                     "pressed": [pad_text(f) for f in press], "read_ms": round(ms, 2), "decide_ms": round(d_ms, 2),
                     "unknown": unknown}
            log.write(json.dumps(entry, default=_jsonable) + "\n")
            log.flush()
            n_dec += 1
            dec_ms.append((ms, d_ms))
            prev, cur, k = _step(emu, press)
            facts, ms = eyes.read(cur, k)
            if d["action"] not in MOVEMENT:      # T0: wait until it can act again (or the round ends)
                for _ in range(MAX_RECOVER // WAIT):
                    if facts.round_state == "over" or moment(facts, last).can_act:
                        break
                    prev, cur, k = _step(emu, [[]] * WAIT)
                    facts, ms = eyes.read(cur, k)
    summary = {"end": end, "end_frame": emu.frame, "decisions": n_dec, "unknown_fighters_at_decisions": n_unknown,
               "unknown_logged": eyes.log.count, "reads": eyes.reads,
               "read_ms_mean": round(eyes.read_ms / max(eyes.reads, 1), 2),
               "decision_ms_mean": round(sum(a + b for a, b in dec_ms) / max(len(dec_ms), 1), 2),
               "decide_ms_mean": round(sum(b for _, b in dec_ms) / max(len(dec_ms), 1), 2),
               "chars": list(chars), "delay": delay}
    with open(os.path.join(out, "inputs.json"), "w") as f:
        json.dump({"state": state_id, "delay": delay, "frames": emu.frame,
                   "inputs": [pad_text(b) for b in emu.inputs]}, f)
    with open(os.path.join(out, "captures.json"), "w") as f:
        json.dump({str(i): h for i, h in sorted(emu.captures.items())}, f)
    with open(os.path.join(out, "reads.jsonl"), "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in eyes.rows))
    with open(os.path.join(out, "round.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary
