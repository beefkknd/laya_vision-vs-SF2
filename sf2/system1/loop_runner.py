"""The SCREEN-ONLY, Qwen-in-loop runner (README M1 / G3): the full self-learning loop's play half.

This is the clean replacement for the table-in-play arm (scripts/play_screen.py -> System1(oracle=table)): here text
laya chooses a move by FOLLOWING Qwen's short-memory advice over the UNRATED two-stage menu (sf2.system1.advice +
sf2.system1.action_menu) - no lookup table, no laya-vision, and NO RAM in play. It reads the screen exactly as the S0
reader harness does (sf2.system1.screen_play), but the decision is `two_stage_decide` instead of `System1.decide`, so
this module's whole import closure is free of the table and the RAM play paths (scripts/hard_gate.py --entry
sf2/system1/loop_runner.py is clean).

One decision (``two_stage_decide``): TWO trained checkpoints, one per round (docs/plan_two_finetune_textlaya.md):
  1. the screen words: the situation sentence (sf2.system1.screen_words.sentence) + "Advice: <lines>." (the short
     memory in force), built byte-for-byte as text laya's training data (scripts/build_advice_data.py);
  2. round 1 - the CATEGORY model picks one of the 7 categories (advice.category_question);
  3. round 2 - the MOVE model picks one move inside it, the stance having pruned the options (advice.move_question over
     advice.moves_in_stance); the model owns the default - when nothing applies the model has learned to block, no code
     forces it. block is only ever OFFERED (the block category on the round-1 menu, block_high always in round 2);
  4. the label rule (advice.two_stage) is logged next to the pick as ``follows_rule`` (did text laya follow the advice).

The two advisors are INJECTED (each an object with ``ask(text, question) -> {option: prob}``): two real text laya
servers (sf2.system1.advisor.Advisor on the cat_v3 / move_v2 checkpoints) in play, follower/mock stubs in tests.
Nothing here calls Qwen or the network.

Seeding, the screen evidence and the Qwen update that rotate the short memory live in the driver
(scripts/play_loop_screen.py) and sf2.system2.screen_evidence, OUTSIDE this clean play module; this module only plays a
game with the lines it is handed and records every decision, so the next game can be played with the updated lines.
"""
import dataclasses
import json
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from ..config import PAD
from ..moves_free import kind_of, steps_of
from ..screen.reader import RoundReader
from ..screen.unknown_log import UnknownLog
from .action_menu import CATEGORIES, CATEGORY_ORDER, DEFAULT_MOVE, category_of
from .advice import (advice_text, category_question, char_categories, char_menu_moves, move_question, moves_in_stance,
                     prompt, read as read_lesson, situation_text, stance_of, two_stage)
from .screen_emu import ScreenEmu
from .screen_words import CAN_ACT, Moment, label, moment, note, players, sentence, situation
from ..vocab import FULL_LIFE

WAIT = 4              # idle frames per step while the fighter cannot act (the 4-frame prev/now gap), as T0
MAX_RECOVER = 90      # frames to wait after a move for the fighter to be able to act again
MAX_FRAMES = 12000    # a round cannot last longer (99 s clock plus the KO)
MOVEMENT_MOVES = frozenset(CATEGORIES["move"])     # the "move" category: no recover wait after it, mirrored by x
# the two-stage menu move names (one flat list, category order), the vocabulary advice lines are parsed against
MENU_MOVES: Tuple[str, ...] = tuple(m for cat in CATEGORY_ORDER for m in CATEGORIES[cat])

DECISION_KEYS = ("category", "cat_probs", "move_options", "move_probs", "rule", "rule_cats", "rule_answers",
                 "follows_rule", "follows_cat", "advice_text", "lines", "prompt_lines",
                 "when", "values", "explored", "source",   # the value-table policy's fields (absent on the rules path)
                 "quorum")                                   # the bee-quorum record (sf2/quorum/decider.py)


def physical(tokens: Sequence[str], facing_right: bool, pad: Dict[str, str] = PAD) -> List[str]:
    """Relative tokens (U/D/F/B + buttons) -> SNES pad buttons, mirrored by facing (a local copy of sf2.emu.vs.physical
    so this clean module does not import the RAM emulator)."""
    fwd, back = ("right", "left") if facing_right else ("left", "right")
    table = {"U": "up", "D": "down", "F": fwd, "B": back}
    return [table.get(t) or pad[t] for t in tokens]


def _flatten(steps) -> List[Tuple[str, ...]]:
    """A move's step script -> one token tuple per frame. ``(tokens, n)`` holds ``tokens`` for ``n`` frames; an
    ``("until", cond, tokens, max)`` step (combos only, never reachable while grounded) presses ``tokens`` once."""
    out: List[Tuple[str, ...]] = []
    for step in steps:
        if step and step[0] == "until":
            out.append(tuple(step[2]))
        else:
            toks, n = step
            out += [tuple(toks)] * n
    return out


def press_frames(me: str, move: str, m: Moment) -> List[List[str]]:
    """The pad frames of ``move`` mirrored as T0 does (movement by x order, the rest by the drawn facing), >= 4
    frames (sf2.moves_free gives the RAM-free step script)."""
    facing_right = m.facing_right(move in MOVEMENT_MOVES)
    frames = [physical(t, facing_right) for t in _flatten(steps_of(me)[move])]
    return frames + [[]] * max(0, 4 - len(frames))


def _posture(m: Moment) -> str:
    """My posture for stance pruning (stand / crouch / air). A decision is only taken when I can act (stand or walk,
    grounded), so this is "stand" in play; the other values keep ``two_stage_decide`` correct if ever called off a
    non-neutral moment."""
    if m.my_air:
        return "air"
    if m.my_label not in CAN_ACT:        # a crouch label (not in CAN_ACT) reads as a crouch posture
        return "crouch"
    return "stand"


def two_stage_decide(cat_advisor, move_advisor, me: str, m: Moment, lines: Sequence[str]) -> Dict:
    """Text laya's move, by following the advice over the unrated two-stage menu. TWO trained checkpoints decide:
    ``cat_advisor`` scores round 1 (the CATEGORY model) and ``move_advisor`` scores round 2 (the MOVE model); each
    ``ask(text, question)`` returns {option: probability}. The model owns the default - no code forces block when
    nothing applies, the model's round-1 and round-2 picks stand. The one safety nuance: block is ALWAYS an offered
    option (the block category is always on the round-1 menu via CATEGORY_ORDER, and block_high is always added to the
    round-2 options below), so the model can always choose the safe move but is never forced to. Returns the pick and
    everything logged about how it was reached."""
    rng, doing, _, _ = situation(m)
    stance = stance_of(_posture(m), rng)
    categories = char_categories(me)                                # ME's own menu (chunli == action_menu.CATEGORIES)
    menu_moves = char_menu_moves(me)                                # ME's two-stage vocabulary (advice parses against it)
    lessons = [read_lesson(t, menu_moves) for t in lines]
    cats, move_answers, rule = two_stage(rng, doing, stance, lessons, m.fireball, categories)

    # Oracle routing (docs/blockages.md B1): show the CATEGORY/MOVE models ONLY the rules that apply to THIS moment.
    # A range-only applicable positive rule is suppressed to `block` when a non-applicable positive rule is also in the
    # prompt (confirmed offline on cat_v3: ['throw up close'] -> throw 0.97, but + 'c.mk at mid' -> block 1.00).
    # Filtering to applicable-only removes the distractor so the model follows; two_stage already ignores the
    # non-applicable rules, so the oracle answer (cats/move_answers/rule) is unchanged.
    prompt_lines = [t for t, lsn in zip(lines, lessons) if lsn.applies(rng, doing, m.fireball)]
    text = prompt(sentence(m), prompt_lines)
    cat_probs = cat_advisor.ask(text, category_question())          # round 1: the CATEGORY model (block always offered)
    category = max(cat_probs, key=cat_probs.get)                    # the model's category pick STANDS (no code floor)
    options = list(moves_in_stance(category, stance, categories))
    if DEFAULT_MOVE not in options:                                # safety: block_high is ALWAYS an offered move,
        options.append(DEFAULT_MOVE)                               # never a forced fallback - the model still picks
    move_probs = move_advisor.ask(text, move_question(options))     # round 2: the MOVE model
    pick = max(move_probs, key=move_probs.get)                     # the model's move pick STANDS
    return {"action": pick, "category": category, "cat_probs": cat_probs, "move_options": list(options),
            "move_probs": move_probs, "rule": rule, "rule_cats": cats, "rule_answers": move_answers,
            "follows_rule": pick in move_answers, "follows_cat": category in cats,
            "advice_text": advice_text(list(lines)), "lines": list(lines), "prompt_lines": prompt_lines}


def _jsonable(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError("not JSON: %r" % type(o))


def pad_text(buttons: Sequence[str]) -> str:
    return "+".join(buttons) if buttons else "-"


class Eyes:
    """The reader with timing; the reader is INJECTED so a test can feed scripted ScreenFacts without an emulator."""

    def __init__(self, unknown_dir: str, reader: Optional[RoundReader] = None):
        self.log = UnknownLog(unknown_dir)
        self.reader = reader if reader is not None else RoundReader(log=self.log)
        self.reads, self.read_ms = 0, 0.0
        self.rows: List[Dict] = []

    def read(self, frame: np.ndarray, k: int):
        t = time.perf_counter()
        facts = self.reader.feed(frame)
        ms = 1000 * (time.perf_counter() - t)
        self.reads += 1
        self.read_ms += ms
        me, him = players(facts)
        hud = facts.hud.health if facts.hud else (None, None)
        my_frac = me.health if me.health is not None else hud[0]
        his_frac = him.health if him.health is not None else hud[1]
        self.rows.append({"k": k, "can_act": label(me) in CAN_ACT and not me.in_air, "me": label(me),
                          "me_air": bool(me.in_air), "him": label(him), "him_air": bool(him.in_air),
                          # drawn health bars, so the LAST read gives the round-END life (the KO) for --no-score scoring
                          "my_life": int(round(my_frac * FULL_LIFE)) if my_frac is not None else None,
                          "his_life": int(round(his_frac * FULL_LIFE)) if his_frac is not None else None,
                          "unknown": [f.side for f in (me, him) if f.unknown], "round": facts.round_state,
                          "ms": round(ms, 2)})
        return facts, ms


def _step(emu: ScreenEmu, frames: List[List[str]]) -> Tuple[np.ndarray, np.ndarray, int]:
    n = len(frames)
    imgs = emu.run(frames, caps=[n - 4, n])
    k = emu.frame
    return imgs[k - 4], imgs[k], k


def _save(img: np.ndarray, path: str) -> None:
    if not os.path.exists(path):
        Image.fromarray(img).save(path)


class WrongFight(RuntimeError):
    """The reader locked other characters than the run's (the start state or the catalog is wrong)."""


def play_round(emu: ScreenEmu, cat_advisor, move_advisor, me: str, opp: str, state: bytes, state_id: Dict, delay: int,
               lines: Sequence[str], out: str, reader: Optional[RoundReader] = None,
               decide: Optional[Callable[["Moment"], Dict]] = None) -> Dict:
    """One round in screen-only mode. By default text laya follows ``lines`` (the rules policy, two_stage_decide).
    ``decide``, when given, REPLACES that per-decision pick with any policy's ``decide(moment) -> {"action", ...}``
    (the value-table policy, sf2.system1.value_table.decider) -- the extra keys it returns (``when``/``explored``)
    are logged via DECISION_KEYS. ``cat_advisor`` / ``move_advisor`` are the two trained checkpoints used by the
    default path. Writes the round record under ``out`` (the files scripts/replay_score.py reads) and returns
    round.json. No hp/result here (the offline replay sources them). ``reader`` is injectable for tests."""
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    eyes = Eyes(os.path.join(out, "unknown"), reader=reader)
    first = emu.load(state)
    eyes.read(first, 0)
    chars = tuple(eyes.reader.lock.chars)
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
            t = time.perf_counter()
            d = decide(m) if decide is not None else two_stage_decide(cat_advisor, move_advisor, me, m, lines)
            d_ms = 1000 * (time.perf_counter() - t)
            press = press_frames(me, d["action"], m)
            unknown = [f.side for f in (facts.left, facts.right) if f.unknown]
            n_unknown += len(unknown)
            _save(prev, os.path.join(out, "frames", "%05d.png" % (k - 4)))
            _save(cur, os.path.join(out, "frames", "%05d.png" % k))
            entry = {"i": n_dec, "k": k, "k_prev": k - 4, "facts": dataclasses.asdict(facts),
                     "moment": dict(dataclasses.asdict(m), side=m.side, dx=m.dx, can_act=m.can_act,
                                    his_attacking=m.his_attacking, doing=m.doing),
                     "note": note(me, m), "situation": list(situation(m)), "action": d["action"],
                     **{x: d[x] for x in DECISION_KEYS if x in d},
                     "pressed": [pad_text(f) for f in press], "read_ms": round(ms, 2), "decide_ms": round(d_ms, 2),
                     "unknown": unknown}
            log.write(json.dumps(entry, default=_jsonable) + "\n")
            log.flush()
            n_dec += 1
            dec_ms.append((ms, d_ms))
            prev, cur, k = _step(emu, press)
            facts, ms = eyes.read(cur, k)
            if d["action"] not in MOVEMENT_MOVES:
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
               "chars": list(chars), "delay": delay, "lines": list(lines)}
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
