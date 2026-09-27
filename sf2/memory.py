"""The memory System 2 writes and System 1 consults before every move (docs/TWO_SYSTEM_PLAN.md §2.2).

A memory is a folder of rule files, one per author, in rank order: ``owner.txt``, ``claude.txt``, ``qwen.txt``.
The file decides the author, whatever a line says. For each decision:

- the most specific matching rule is picked (most conditions; ties go to the higher-ranked author, then weight);
- its move is played only if ``p[top] - p[move] <= tau``, where tau is the rule's own weight (how firmly it pushes:
  0 never, 1 always), unless the caller fixes one tau for every rule;
- the use is reported as *fired* (a rule matched) and *changed* (the move played is not laya's top move).
"""
import os
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from . import contract as C

AUTHORS = ("owner", "claude", "qwen")  # rank order


class Memory:
    def __init__(self, rules: List[C.Rule]):
        self.rules = rules

    def pick(self, situation: C.Situation) -> Optional[C.Rule]:
        matches = [r for r in self.rules if r.matches(situation)]
        if not matches:
            return None
        return min(matches, key=lambda r: (-len(r.when), AUTHORS.index(r.author), -r.weight))

    def apply(self, note: str, probs: Dict[str, float], top: str, tau: Optional[float] = None) -> Tuple[str, Dict]:
        if not self.rules:
            return top, {"rule": None, "author": None, "fired": False, "changed": False}
        rule = self.pick(C.parse_note(note))
        if rule is None:
            return top, {"rule": None, "author": None, "fired": False, "changed": False}
        limit = rule.weight if tau is None else tau
        change = rule.move != top and probs.get(top, 0.0) - probs.get(rule.move, 0.0) <= limit
        return (rule.move if change else top), {"rule": str(rule), "author": rule.author, "fired": True,
                                                 "changed": change}


def load(folder: str) -> Memory:
    """Read ``<folder>/{owner,claude,qwen}.txt`` (each optional); other files are ignored."""
    rules: List[C.Rule] = []
    for author in AUTHORS:
        path = os.path.join(folder, author + ".txt")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            try:
                parsed = C.parse_rules(f.read())
            except ValueError as e:
                raise ValueError("%s: %s" % (path, e)) from None
        rules += [replace(r, author=author) for r in parsed]
    return Memory(rules)
