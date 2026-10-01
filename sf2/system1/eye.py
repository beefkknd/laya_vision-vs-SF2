"""The U arm's decision path (docs/prereg_u_perception.md): laya-vision as the eye. A checkpoint tagged "perception"
(scripts/train.py on test_data_u: note v3, frames v3) answers every question of sf2.data.u_data.questions in ONE
predict per decision (the two images are encoded once). It sees only the two frames (HUD visible, u_data.hud_frame)
and "me=<char>" - nothing of the game's memory: the decision reads no field of it (tests/test_eye.py checks the code
and a poisoned row).

Question 8 -> per move a rank score, P(likely works) + 0.5 P(may work), and a rating word, the argmax answer (ties
to the first of perception.Q8_ANSWERS). Walking in is not asked: it is the baseline every answer is relative to.

- Without an advisor: the best rank score plays, or walking in when no move's word is better than "likely fails".
- With an advisor: text laya's shortlist is built as advisor.shortlist builds it (the top 3 by rank score, skipping
  moves an applying avoid lesson rules out; the moves an applying lesson names; walking in), but every word is
  question 8's own (no score scale). Text laya's situation is laya-vision's own answers in text laya's words:
  range  Q1 band: throw -> "close", poke / mid -> "mid", far -> "far" (``range_word``). Text laya's words (and the
         table's cells) cut at sf2.vocab CLOSE = 55 px; the throw band (<= 43) lies wholly below it, the poke band
         straddles it (Chun-Li 44-64; others up to 82): on Chun-Li's 1,473 poke-band training decisions of the U
         collection RAM's word was "mid" in 59% (867) and "close" in 41%, so poke -> "mid" (the eye's range word then
         agrees with RAM's on 95.2% of her decisions vs 93.5% with poke -> "close").
  doing  in opp_doing's order: Q3 airborne (jumping at me / away / landing) -> "jumping"; Q2 attacking or
         recovering after a miss -> "attacking" (both are his attack states, which opp_doing calls attacking);
         Q2 being hit or Q5 him stunned / knocked down / dizzy -> "stunned"; else (neutral, blocking) "standing"
         (``doing_of``). "crouching" is never said: no question asks it, so a lesson about his crouch never applies.
  bars   Q7 my bar / his bar: the same four words (sf2.vocab.BARS).
Every answer of every question is logged with the decision ("eye"), with the rank scores ("rank") and the situation
text laya was told ("eye_situation").
"""
import hashlib
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..data.perception import Q8_ANSWERS, QUESTIONS
from ..data.u_data import NOTE_VERSION, Q8_KEY, argmax, eye_note, hud_frame, q8_moves, questions
from .advice import FAILS, FORWARD, MAY, RATING, WORKS, answers, prompt, question, read, situation_text
from .advisor import SHORTLIST, applicable

CHECKPOINT_FILES = ("model.safetensors", "vlm_agent_config.json")
RANGE_WORD = {"throw": "close", "poke": "mid", "mid": "mid", "far": "far"}
NOT_FREE = ("stunned", "knocked down", "dizzy")
MAY_WEIGHT = 0.5


def checkpoint_sha256(path: str) -> str:
    """One sha256 for a checkpoint: over its weights' and its config's own sha256 (what run.json records for --eye)."""
    h = hashlib.sha256()
    for name in CHECKPOINT_FILES:
        f = os.path.join(path, name)
        if not os.path.isfile(f):
            raise SystemExit("no %s in the checkpoint %s" % (name, path))
        with open(f, "rb") as fh:
            h.update(("%s %s\n" % (name, hashlib.sha256(fh.read()).hexdigest())).encode())
    return h.hexdigest()


def range_word(band: str) -> str:
    return RANGE_WORD[band]


def doing_of(phase: str, air: str, him: str) -> str:
    if air != "grounded":
        return "jumping"
    if phase in ("attacking", "recovering after a miss"):
        return "attacking"
    if phase == "being hit" or him in NOT_FREE:
        return "stunned"
    return "standing"


def top(ans: Dict[str, Dict[str, float]], key: str) -> str:
    return argmax(ans[key], QUESTIONS[key])


def situation_of(ans: Dict[str, Dict[str, float]]) -> Tuple[str, str, str, str]:
    """(range, what he is doing, my bar, his bar) in text laya's words, from laya-vision's answers only."""
    return (range_word(top(ans, "range")), doing_of(top(ans, "phase"), top(ans, "air"), top(ans, "him_can_act")),
            top(ans, "my_bar"), top(ans, "his_bar"))


def rank_scores(ans: Dict[str, Dict[str, float]], moves: Sequence[str]) -> Dict[str, float]:
    return {m: float(ans[Q8_KEY + m][WORKS]) + MAY_WEIGHT * float(ans[Q8_KEY + m][MAY]) for m in moves}


def q8_words(ans: Dict[str, Dict[str, float]], moves: Sequence[str]) -> Dict[str, str]:
    return {m: argmax(ans[Q8_KEY + m], Q8_ANSWERS) for m in moves}


def pick(rank: Dict[str, float], words: Dict[str, str]) -> Tuple[str, str]:
    """(action, best): the best rank score (ties: the first move), or walking in when every word is "likely fails"."""
    best = max(rank, key=rank.get)
    return (best if any(RATING[w] > RATING[FAILS] for w in words.values()) else FORWARD), best


def shortlist(rank: Dict[str, float], words: Dict[str, str], lessons: Sequence[str], moves: Sequence[str],
              situation: Tuple[str, str]) -> Dict[str, Optional[str]]:
    """advisor.shortlist with question 8's words: the top SHORTLIST by rank score not ruled out by an applying avoid
    lesson, the moves an applying use / always lesson names, and walking in."""
    rng, doing = situation
    live = [les for les in (read(t, list(moves) + [FORWARD]) for t in lessons) if les.applies(rng, doing)]
    out_ruled = {les.move for les in live if les.polarity == "neg"}
    best = sorted((m for m in rank if m not in out_ruled), key=rank.get, reverse=True)[:SHORTLIST]
    named = [les.move for les in live if les.move in rank and les.move not in best and les.polarity in ("soft", "hard")]
    out: Dict[str, Optional[str]] = {m: words[m] for m in best + sorted(set(named))}
    out[FORWARD] = None
    return out


def choose(advisor, situation: Tuple[str, str, str, str], rank: Dict[str, float], words: Dict[str, str],
           lessons: Sequence[str], moves: Sequence[str]) -> Dict:
    """advisor.choose on the eye's shortlist: text laya's pick, what it read, and what the label rule says."""
    lessons = applicable(lessons, moves, situation[:2])
    options = shortlist(rank, words, lessons, moves, situation[:2])
    text = prompt(situation_text(*situation), lessons)
    probs = advisor.ask(text, question(options))
    choice = max(probs, key=probs.get)
    rule, why = answers(situation[0], situation[1], options, [read(t, list(moves) + [FORWARD]) for t in lessons])
    return {"action": choice, "advice_text": text, "shortlist": options, "advice_probs": probs,
            "rule_answers": rule, "rule": why, "follows_rule": choice in rule}


class Eye:
    """A perception checkpoint for one character. ``agent``: a loaded laya-vision agent (its cfg must say
    perception and note v3)."""

    def __init__(self, agent, me: str):
        cfg = getattr(agent, "cfg", None) or {}
        if not cfg.get("perception") or cfg.get("note_version") != NOTE_VERSION:
            raise ValueError("not a perception checkpoint (config perception=%r note_version=%r)" % (
                cfg.get("perception"), cfg.get("note_version")))
        self.agent, self.me = agent, me
        self.note = eye_note(me)
        self.moves: List[str] = q8_moves(me)
        self.questions = questions(me)

    def state(self, prev: np.ndarray, cur: np.ndarray) -> Dict:
        from PIL import Image

        return {"images": [Image.fromarray(hud_frame(prev)).convert("RGB"),
                           Image.fromarray(hud_frame(cur)).convert("RGB")], "context": self.note}

    def ask(self, prev: np.ndarray, cur: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Every question's answer probabilities, from one predict."""
        got = self.agent.predict(self.state(prev, cur), self.questions)["answers"]
        return {k: {a: float(p) for a, p in got[k]["probabilities"].items()} for k in self.questions}

    def decide(self, prev: np.ndarray, cur: np.ndarray, advisor=None, lessons: Sequence[str] = ()) -> Dict:
        t0 = time.time()
        ans = self.ask(prev, cur)
        ms = round(1000 * (time.time() - t0), 1)
        rank, words = rank_scores(ans, self.moves), q8_words(ans, self.moves)
        sit = situation_of(ans)
        action, best = pick(rank, words)
        log = {"best": best, "p_hit": None, "predicted": "none", "probs": {},
               "eye": {k: {a: round(p, 4) for a, p in v.items()} for k, v in ans.items()},
               "rank": {m: round(v, 4) for m, v in rank.items()}, "eye_situation": list(sit), "eye_ms": ms}
        if advisor is None:
            return dict(log, action=action)
        return dict(choose(advisor, sit, rank, words, lessons, self.moves), **log)
