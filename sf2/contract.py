"""Step 0 of the two-system plan (docs/TWO_SYSTEM_PLAN.md): the formats System 1 and any System 2 share.

System 1 (laya, the hands) writes a *moment record* for decisions worth a second look. A System 2 (you, an LLM
coach such as Qwen) answers with *situation rules* over the RAM note's fields, or *moment advice* for one
recorded moment. Everything is validated on the way in: an unknown field, value or move is an error naming what is
wrong, so a bad instruction never reaches the hands.

    opp=ryu opp_state=jump dist=mid -> hp  weight=0.6 author=you  # anti-air with fierce
"""
import math
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .actions import ACTIONS

# --- the RAM note (sf2.ram.text_state) as named fields ----------------------------------------------------------------

CHARACTERS = ("ryu", "honda", "blanka", "guile", "ken", "chunli", "zangief", "dhalsim", "balrog", "vega", "sagat",
              "bison")
STATES = ("stand", "crouch", "block", "attack", "hit", "dizzy", "jump", "jumpattack", "other")
BINS = ("close", "mid", "far")
FIELDS: Dict[str, Tuple[str, ...]] = {  # field -> allowed values; () = a number from 0 to 100
    "me": CHARACTERS, "me_state": STATES, "me_hp": (),
    "opp": CHARACTERS, "opp_state": STATES, "opp_hp": (),
    "dist": BINS, "facing": ("left", "right"), "corner": ("me", "opp", "none"), "time": ("early", "mid", "late"),
    "last": tuple(ACTIONS), "fireball": BINS + ("none",),
}
NUMERIC = {f for f, values in FIELDS.items() if not values}
# the note's words, in order: key=value pairs, and the two bare state words after each fighter's name
_NOTE_ORDER = ["me", "me_state", "me_hp", "opp", "opp_state", "opp_hp", "dist", "facing", "corner", "time", "last",
               "fireball"]
_NOTE_KEYS = {"me": "me", "me_hp": "hp", "opp": "opp", "opp_hp": "hp", "dist": "dist", "facing": "facing",
              "corner": "corner", "time": "time", "last": "last", "fireball": "fireball"}

Situation = Dict[str, object]


def _check_value(name: str, value):
    if name in NUMERIC:
        if not isinstance(value, int) or not 0 <= value <= 100:
            raise ValueError("%s must be a number from 0 to 100, got %r" % (name, value))
    elif value not in FIELDS[name]:
        raise ValueError("%s=%s is not a known value (one of: %s)" % (name, value, ", ".join(FIELDS[name])))


def parse_note(note: str) -> Situation:
    """``me=chunli stand hp=80 opp=ryu jump hp=45 dist=mid ...`` -> {"me": "chunli", "me_state": "stand", ...}."""
    words = note.split()
    if len(words) != len(_NOTE_ORDER):
        raise ValueError("a note has %d words, got %d: %r" % (len(_NOTE_ORDER), len(words), note))
    out: Situation = {}
    for name, word in zip(_NOTE_ORDER, words):
        if name in _NOTE_KEYS:
            key, sep, value = word.partition("=")
            if not sep or key != _NOTE_KEYS[name]:
                raise ValueError("expected %s=..., got %r" % (_NOTE_KEYS[name], word))
        else:
            value = word
        if name in NUMERIC:
            if not value.isdigit():
                raise ValueError("%s must be a number, got %r" % (name, value))
            value = int(value)
        _check_value(name, value)
        out[name] = value
    return out


# --- situation rules ------------------------------------------------------------------------------------------------

DEFAULT_WEIGHT = 0.5
_OPS = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b, "<": lambda a, b: a < b, ">": lambda a, b: a > b,
        "=": lambda a, b: a == b}
_COND = re.compile(r"^([a-z_]+)(<=|>=|<|>|=)(\S+)$")


@dataclass(frozen=True)
class Rule:
    when: Tuple[Tuple[str, str, object], ...]
    move: str
    weight: float = DEFAULT_WEIGHT
    author: str = "you"
    reason: str = field(default="", compare=False)

    def matches(self, s: Situation) -> bool:
        return all(_OPS[op](s[name], value) for name, op, value in self.when)

    def __str__(self) -> str:
        conds = " ".join("%s%s%s" % c for c in self.when)
        text = "%s -> %s weight=%g author=%s" % (conds, self.move, self.weight, self.author)
        return text + (" # " + self.reason if self.reason else "")


def _condition(text: str) -> Tuple[str, str, object]:
    m = _COND.match(text)
    if not m:
        raise ValueError("bad condition %r (write field=value, or hp fields with < > <= >=)" % text)
    name, op, value = m.groups()
    if name not in FIELDS:
        raise ValueError("unknown field %r (fields: %s)" % (name, ", ".join(FIELDS)))
    if name in NUMERIC:
        if not value.isdigit():
            raise ValueError("%s needs a number from 0 to 100, got %r" % (name, value))
        value = int(value)
    elif op != "=":
        raise ValueError("operator %s works only on the hp fields, not %s" % (op, name))
    _check_value(name, value)
    return name, op, value


def parse_rule(line: str) -> Rule:
    text, _, reason = line.partition("#")
    if "->" not in text:
        raise ValueError("a rule is: conditions -> move, got %r" % line.strip())
    left, right = text.split("->", 1)
    conds = [_condition(c) for c in left.split()]
    if not conds:
        raise ValueError("a rule needs at least one condition (a rule for every situation is not a situation)")
    names = [c[0] for c in conds if c[1] == "="]
    if len(names) != len(set(names)):
        raise ValueError("a field is given twice with = in %r" % line.strip())
    parts = right.split()
    if not parts or parts[0] not in ACTIONS:
        raise ValueError("unknown move %r (moves: %s)" % (parts[0] if parts else "", ", ".join(ACTIONS)))
    if ("last", "=", parts[0]) in conds:
        raise ValueError("a rule on last=%s may not advise %s: it would repeat itself forever" % (parts[0], parts[0]))
    opts = {"weight": DEFAULT_WEIGHT, "author": "you"}
    for p in parts[1:]:
        key, sep, value = p.partition("=")
        if not sep or key not in opts:
            raise ValueError("unknown option %r (options: weight=0..1, author=name)" % p)
        opts[key] = value
    try:
        weight = float(opts["weight"])
    except ValueError:
        raise ValueError("weight must be a number from 0 to 1, got %r" % opts["weight"]) from None
    if not 0 <= weight <= 1:
        raise ValueError("weight must be from 0 to 1, got %g" % weight)
    return Rule(tuple(conds), parts[0], weight, str(opts["author"]), reason.strip())


def parse_rules(text: str) -> List[Rule]:
    """A playbook: one rule per line; blank lines and # comments are skipped. Errors name the line."""
    rules = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            rules.append(parse_rule(line))
        except ValueError as e:
            raise ValueError("line %d: %s" % (n, e)) from None
    return rules


# --- moment records and advice -------------------------------------------------------------------------------------

WHY = ("unsure", "surprised", "audit")


def moment_id(match: int, round: int, frame: int) -> str:
    return "m%dr%df%d" % (match, round, frame)


def moment_record(match: int, round: int, frame: int, why: str, notes_before: Sequence[str], note: str,
                  probs: Dict[str, float], played: str, dealt: int, taken: int, frames: Sequence[str],
                  **extra) -> Dict:
    """One flagged decision, validated, as a JSON-ready dict (``extra`` keys, e.g. margin, pass through)."""
    if why not in WHY:
        raise ValueError("why must be one of %s, got %r" % (WHY, why))
    if played not in ACTIONS:
        raise ValueError("played %r is not a move" % played)
    bad = [a for a in probs if a not in ACTIONS]
    if bad or any(p < 0 for p in probs.values()) or sum(probs.values()) > 1 + 1e-6:
        raise ValueError("probs must map moves to probabilities that sum to at most 1: %r" % probs)
    for name, value in (("dealt", dealt), ("taken", taken), ("match", match), ("round", round), ("frame", frame)):
        if not isinstance(value, int) or value < 0:
            raise ValueError("%s must be a whole number >= 0, got %r" % (name, value))
    try:
        for n in (*notes_before, note):
            parse_note(n)
    except ValueError as e:
        raise ValueError("note: %s" % e) from None
    if not frames:
        raise ValueError("frames: a moment needs its screenshots")
    return {"id": moment_id(match, round, frame), "match": match, "round": round, "frame": frame, "why": why,
            "notes_before": list(notes_before), "note": note, "probs": dict(probs), "played": played,
            "dealt": dealt, "taken": taken, "frames": list(frames), **extra}


@dataclass(frozen=True)
class Advice:
    moment: str
    move: str
    reason: str = ""
    author: str = "you"


def parse_advice(d: Dict) -> Advice:
    if not d.get("moment"):
        raise ValueError("advice needs the moment id it answers")
    if d.get("move") not in ACTIONS:
        raise ValueError("unknown move %r in advice" % d.get("move"))
    return Advice(str(d["moment"]), d["move"], str(d.get("reason", "")), str(d.get("author", "you")))


# --- why a decision gets flagged -----------------------------------------------------------------------------------

UNSURE_MARGIN = 0.10  # top-2 margin below this = unsure; tune so about 5% of decisions are flagged
AUDIT_RATE = 0.01


def confidence(probs: Dict[str, float]) -> Tuple[float, float]:
    """(top-2 margin, entropy in nats) of one decision's probabilities."""
    p = sorted(probs.values(), reverse=True) + [0.0, 0.0]
    entropy = -sum(x * math.log(x) for x in probs.values() if x > 0)
    return p[0] - p[1], entropy


def flag(margin: float, taken_next: int, round_lost: bool, rng: random.Random,
         unsure_margin: float = UNSURE_MARGIN, audit_rate: float = AUDIT_RATE) -> Optional[str]:
    """Why this decision is worth a System 2's look, or None. Surprise first, then doubt, then a random audit."""
    if taken_next > 0 or round_lost:
        return "surprised"
    if margin < unsure_margin:
        return "unsure"
    if rng.random() < audit_rate:
        return "audit"
    return None
