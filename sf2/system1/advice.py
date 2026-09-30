"""Text laya's job: turn Qwen's short memory into one move. laya-vision watches the screen and rates the moves; text
laya reads the moment in words, the rated shortlist and the advice, and picks one.

The same functions build the text at training time and at play time (the words must be byte-identical):

    situation_text   "He is up close and jumping. My bar is half, his bar is high."
    option_text      "likely works" / "may work" / "likely fails" (laya-vision's score, in words), "walk in" (forward)
    advice_text      "Advice: use more lp up close; avoid sweep."
    question         the choice question over the shortlist

``answers`` is the label rule. What the advice is worth is not text laya's business (Qwen judges that from the games);
text laya must FOLLOW it faithfully:

  1. a lesson applies only when its conditions hold (distance: up close / at mid / at far; what he is doing: when he
     jumps / crouches / attacks / stands);
  2. an applying negative lesson ("avoid", "never", "stop", "less", "don't") rules its move out;
  3. an applying hard lesson ("always", "only") picks its move;
  4. an applying soft lesson ("use more", "prefer", "go for", "when he jumps, use") picks its move unless laya-vision
     says it likely fails right now;
  5. otherwise the move(s) laya-vision rates best, among those not ruled out; if every one likely fails, walk in.
Every answer this returns is correct; a training target spreads over all of them.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..vocab import BARS, OPP_STATES, RANGE_WORDS, RANGES

FORWARD = "forward"

WORKS, MAY, FAILS = "likely works", "may work", "likely fails"
RATING = {WORKS: 2, MAY: 1, FAILS: 0}
WORKS_AT, MAY_AT = 0.5, 0.3          # laya-vision's score for an attack, P(hit) -> words
# A block's score is P(blocked), on another scale (2026-09-29, 131k decisions vs Ken/Ryu/Honda): a block scored 0.1-0.2
# actually blocked 40% of the time and 0.2+ 59-65%, while an attack scored 0.5+ actually hit 25-44%. With one threshold
# a block was never rated above "likely fails" (independent review, docs/reviews/2026-09-29_dr_fable.md).
BLOCK_WORKS_AT, BLOCK_MAY_AT = 0.2, 0.1
BLOCK_MOVES = ("block_high", "block_low")
# The lookup table's scale (sf2.data.value_oracle, docs/prereg_2x2.md): a move's expected net (hp dealt - taken until
# the next decision) -> words. Chosen so the word mix of the table's top 3 matches runs/all8's shortlist mix on 78,024
# Chun-Li no-advice decisions: 16.1% likely works / 18.3% may work / 65.6% likely fails. Blocks use the same net scale
# (a net is a net: no separate block scale).
NET_WORKS_AT, NET_MAY_AT = 3.0, 1.8
SCALES = ("p_hit", "net")

INSTRUCTIONS = "Which move do I do now? Follow the advice when it fits this moment."


def rating(score: float, move: Optional[str] = None, scale: str = "p_hit") -> str:
    """A move's score in words. ``scale`` "p_hit": laya-vision's score, a block (``move``) on its own scale; without
    ``move`` the attack scale (what scripts/build_advice_data.py built text laya's training data with). "net": the
    lookup table's expected net in hp, every move (blocks too) on one scale."""
    if scale not in SCALES:
        raise ValueError("unknown rating scale %r (one of %s)" % (scale, ", ".join(SCALES)))
    if scale == "net":
        works, may = NET_WORKS_AT, NET_MAY_AT
    else:
        works, may = (BLOCK_WORKS_AT, BLOCK_MAY_AT) if move in BLOCK_MOVES else (WORKS_AT, MAY_AT)
    return WORKS if score >= works else MAY if score >= may else FAILS


def opp_doing(entry: Dict) -> str:
    """What the opponent is doing, from a game-log action entry (sf2.game_log.action_entry)."""
    if entry.get("opp_air"):
        return "jumping"
    return {"crouch": "crouching", "attack": "attacking", "special": "attacking", "hit_stun": "stunned"}.get(
        entry.get("opp_state"), "standing")


def situation_text(rng: str, doing: str, my_bar: str, opp_bar: str) -> str:
    if rng not in RANGES or doing not in OPP_STATES or my_bar not in BARS or opp_bar not in BARS:
        raise ValueError("bad situation %r %r %r %r" % (rng, doing, my_bar, opp_bar))
    return "He is %s and %s. My bar is %s, his bar is %s." % (RANGE_WORDS[rng], doing, my_bar, opp_bar)


def option_text(move: str, rated: Optional[str]) -> str:
    return "walk in" if move == FORWARD else rated


def advice_text(lessons: Sequence[str]) -> str:
    return ("Advice: " + "; ".join(lessons) + ".") if lessons else "Advice: none."


def question(options: Dict[str, str]) -> Dict:
    """``options``: move -> laya-vision's rating words (FORWARD's is ignored)."""
    return {"type": "choice", "instructions": INSTRUCTIONS,
            "criteria": {m: option_text(m, r) for m, r in options.items()}}


def prompt(situation: str, lessons: Sequence[str]) -> str:
    return situation + " " + advice_text(lessons)


# ---- reading a lesson's text (the label must follow what the TEXT says, since that is all text laya reads) ----

NEG = re.compile(r"\b(avoid|never|stop|drop|skip|don't|do not|less|fewer|no|quit|punish(es|ed)?)\b", re.I)
HARD = re.compile(r"\b(always|only)\b", re.I)
HE = r"(he|ryu|ken|blanka|guile|chun-?li|honda|e\.? honda|zangief|dhalsim)"      # the opponent, by pronoun or name
WHEN = {"jumping": re.compile(r"\bwhen %s (jumps|is jumping|is in the air)\b|\bjumping\b" % HE, re.I),
        "crouching": re.compile(r"\bwhen %s (crouches|is crouching|ducks)\b" % HE, re.I),
        "attacking": re.compile(r"\bwhen %s (attacks|is attacking|presses)\b" % HE, re.I),
        "standing": re.compile(r"\bwhen %s (stands|is standing)\b" % HE, re.I),
        "stunned": re.compile(r"\bwhen %s is (stunned|in hit_stun|reeling)\b|\bin hit_stun\b" % HE, re.I)}
WHERE = {"close": re.compile(r"\b(close|next to)\b", re.I),
         "mid": re.compile(r"\b(mid|middle)\b", re.I),
         "far": re.compile(r"\b(far|long range)\b", re.I)}
HE_DOES = re.compile(r"\b(he|him|ryu|ken|blanka|guile|chunli|chun-li|honda|zangief|dhalsim)\s+(is\s+)?\w+", re.I)


@dataclass(frozen=True)
class Lesson:
    text: str
    move: Optional[str]          # None: no move to do (e.g. an opponent habit)
    polarity: str                # "soft" | "hard" | "neg" | "none"
    where: Optional[str]         # a range, or None: anywhere
    when: Optional[str]          # an opponent state, or None: whatever he does

    def applies(self, rng: str, doing: str) -> bool:
        return (self.where is None or self.where == rng) and (self.when is None or self.when == doing)


def parse(text: str, moves: Sequence[str]) -> Lesson:
    """A lesson as its words say it. Qwen writes "<what to do>: <why>": the move and its polarity are read from the
    part before the first colon (the why may say "he does not punish it"), the conditions from the whole lesson
    ("use c.mk at mid: it lands when he stands"). The move is the character's move named there (longest name
    first, so "c.mk" is not read as "mk"; "throw" only when no other move is named, as in "throw c.mk"; what HE does,
    "he jumps", is not her move); none named -> no move."""
    what = text.split(":", 1)[0]
    named = [m for m in sorted(moves, key=len, reverse=True)
             if re.search(r"(?<![\w.])%s(ing|s)?(?![\w])" % re.escape(m), HE_DOES.sub(" ", what))]
    if len(named) > 1 and "throw" in named:
        named.remove("throw")
    move = named[0] if named else None
    if move is None:
        pol = "none"
    elif NEG.search(what):
        pol = "neg"
    elif HARD.search(what):
        pol = "hard"
    else:
        pol = "soft"
    where = [r for r, pat in WHERE.items() if pat.search(text)]      # conditions: anywhere in the lesson
    when = [s for s, pat in WHEN.items() if pat.search(text)]
    if len(where) > 1 or len(when) > 1:
        raise ValueError("lesson has conflicting conditions: %r" % text)
    return Lesson(text, move, pol, where[0] if where else None, when[0] if when else None)


def read(text: str, moves: Sequence[str]) -> Lesson:
    """``parse`` for play time: a lesson it cannot read (e.g. two ranges in one line) counts as naming no move, so
    System 1 and System 2's checks skip it instead of crashing the game."""
    try:
        return parse(text, moves)
    except ValueError:
        return Lesson(text, None, "none", None, None)


def answers(rng: str, doing: str, options: Dict[str, str], lessons: Sequence[Lesson]) -> Tuple[List[str], str]:
    """The correct move(s) and which rule decided. ``options``: move -> rating words (FORWARD always offered)."""
    if FORWARD not in options:
        raise ValueError("the shortlist must offer %r" % FORWARD)
    live = [les for les in lessons if les.move in options and les.applies(rng, doing)]
    out = {les.move for les in live if les.polarity == "neg"}
    hard = sorted({les.move for les in live if les.polarity == "hard"} - out)
    if hard:
        return hard, "hard"
    soft = sorted({les.move for les in live if les.polarity == "soft" and options[les.move] != FAILS} - out)
    if soft:
        return soft, "soft"
    moves = [m for m in options if m != FORWARD and m not in out]
    best = max((RATING[options[m]] for m in moves), default=0)
    if best > 0:
        return sorted(m for m in moves if RATING[options[m]] == best), "vision"
    if FORWARD not in out:
        return [FORWARD], "walk"
    return sorted(moves) or [FORWARD], "nothing_left"
