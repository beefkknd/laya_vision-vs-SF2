"""System 2's writer: what Qwen is told, and how its reply becomes validated rules (docs/TWO_SYSTEM_PLAN.md).

Qwen gets the game's *description* only: the moves, the note's fields and values, and the rule format. It gets no
tactics (the owner's rule: the playbook starts empty). Each batch it sees the moments (the decision before each hit,
and audits), the current memory, and how the last batch went, and answers with rules. Every rule is validated by
``sf2.contract`` before it reaches the memory; rejected lines are kept with the reason.
"""
import os
import re
from dataclasses import replace as with_fields
from typing import Dict, List, Optional, Sequence, Tuple

from . import contract as C
from .actions import ACTIONS, CRITERIA

MAX_NEW = 5


def _game_description(max_new: int) -> str:
    moves = "\n".join("- %s: %s" % (a, CRITERIA[a]) for a in ACTIONS)
    fields = "\n".join("- %s: %s" % (f, "a number from 0 to 100" if not v else ", ".join(v))
                       for f, v in C.FIELDS.items())
    return (
        "You are System 2, the coach of a Street Fighter II player. The player (System 1) picks one move every "
        "4 frames from what it sees. You cannot press buttons; you improve the player only by writing situation "
        "rules into its memory.\n\n"
        "The moves:\n%s\n\n"
        "Before every move the player reads a one-line note about the fight, for example:\n"
        "me=guile stand hp=60 opp=ryu jump hp=90 dist=mid facing=right corner=none time=early last=forward "
        "fireball=none\n"
        "As rule conditions its fields are named:\n%s\n"
        "(me_state and opp_state are the state words after each fighter's name; me_hp and opp_hp are the two hp "
        "values; dist close is under 80 px, mid under 120 px; corner says whose back is to a wall; fireball is how "
        "far a projectile is from you.)\n\n"
        "Rule format, one per line:\n"
        "  field=value field=value ... -> move  # short reason\n"
        "The hp fields also accept < > <= >=, for example me_hp<30. A rule applies when every condition matches; "
        "the most specific rule wins. A rule may not condition on last=X and advise X. The player plays your move "
        "when it does not strongly disagree, so rules for situations it already handles change nothing.\n\n"
        "Write at most %d rules. A rule with exactly the same conditions as an existing one replaces it. Reply "
        "with rules only." % (moves, fields, max_new))


def _moment_line(m: Dict) -> str:
    top = ", ".join("%s %.2f" % kv for kv in list(m["probs"].items())[:3])
    before = ("\n  1 s before: %s" % m["notes_before"][0]) if m.get("notes_before") else ""
    return ("#%s [%s] took %d, dealt %d in the next 0.5 s; played %s; its choices: %s\n  now: %s%s"
            % (m["id"], m["why"], m["taken"], m["dealt"], m["played"], top, m["note"], before))


def build_messages(moments: Sequence[Dict], rules: Sequence[C.Rule], feedback: Optional[str],
                   max_new: int = MAX_NEW) -> List[Dict]:
    memory = "\n".join(str(r).replace(" author=%s" % r.author, "") for r in rules) or "(empty)"
    parts = ["The memory now:\n" + memory]
    if feedback:
        parts.append("How the last batch went:\n" + feedback)
    parts.append("Moments from the last batch (surprised = the last decision before the player was hit; "
                 "audit = a random decision):\n" + "\n".join(_moment_line(m) for m in moments))
    parts.append("Write your rules.")
    return [{"role": "system", "content": _game_description(max_new)},
            {"role": "user", "content": "\n\n".join(parts)}]


_THINK = re.compile(r"<think>.*?(</think>|$)", re.S)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def parse_reply(text: str) -> Tuple[List[C.Rule], List[Tuple[str, str]]]:
    """Qwen's reply -> (valid rules, [(line, why it was rejected)]). Lines without '->' are prose, not rules."""
    rules, rejected, seen = [], [], set()
    for raw in _THINK.sub("", text).splitlines():
        if "->" not in raw:
            continue
        line = _BULLET.sub("", raw.replace("`", "")).strip()
        try:
            r = C.parse_rule(line)
        except ValueError as e:
            rejected.append((raw.strip(), str(e)))
            continue
        if (r.when, r.move) not in seen:
            seen.add((r.when, r.move))
            rules.append(r)
    return rules, rejected


def merge(folder: str, new: Sequence[C.Rule], replace: bool = False, author: str = "qwen") -> int:
    """Add new rules to ``<folder>/<author>.txt``; with ``replace`` a rule for the same conditions replaces the old
    one. Returns how many rules were added or replaced."""
    path = os.path.join(folder, author + ".txt")
    old: List[C.Rule] = []
    if os.path.exists(path):
        with open(path) as f:
            old = C.parse_rules(f.read())
    changed = 0
    for r in new:
        if replace and any(o.when == r.when for o in old):
            old = [o for o in old if o.when != r.when]
        elif any((o.when, o.move) == (r.when, r.move) for o in old):
            continue
        old.append(r)
        changed += 1
    os.makedirs(folder, exist_ok=True)
    with open(path, "w") as f:
        f.write("".join(str(with_fields(r, author=author)) + "\n" for r in old))
    return changed
