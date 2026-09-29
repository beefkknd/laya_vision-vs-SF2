"""How much System 2 changes a memory from one version to the next. A playbook meant to be long-term knowledge
should move slowly; a rewrite that replaces most of it, or turns "use X" into "avoid X", is flagged.

A lesson's identity is what it tells System 1 to do, not its wording: (kind, move, polarity, where, when) as
sf2.advice reads the text. "avoid sweep up close: it whiffs" and "no sweep up close" are the same lesson; "use sweep
up close" vs "avoid sweep up close" is a FLIP (same move and conditions, opposite advice).
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .advice import read

RADICAL_SHARE = 0.5          # more than half the lessons added or dropped in one rewrite


@dataclass(frozen=True)
class Change:
    kept: int
    added: List[str]
    dropped: List[str]
    flipped: List[Tuple[str, str]]       # (old text, new text)
    size: Tuple[int, int]                # lessons before, after

    @property
    def churn(self) -> float:
        """Share of the memory that changed: (added + dropped) / the larger version (a flip counts in both)."""
        return (len(self.added) + len(self.dropped)) / max(1, max(self.size))

    @property
    def radical(self) -> bool:
        return bool(self.flipped) or self.churn > RADICAL_SHARE

    def line(self) -> str:
        flag = "  RADICAL" if self.radical else ""
        return "kept %d, added %d, dropped %d, flipped %d (churn %.0f%%)%s" % (
            self.kept, len(self.added), len(self.dropped), len(self.flipped), 100 * self.churn, flag)


def diff(old: Dict, new: Dict, moves: Sequence[str]) -> Change:
    """``old`` / ``new``: memory files (sf2.memory format); a missing one counts as empty."""
    def keyed(mem):
        out = {}
        for les in (mem or {}).get("lessons", []):
            p = read(les["text"], moves)
            kind = "negative" if p.polarity == "neg" else "positive" if p.polarity in ("soft", "hard") else les["kind"]
            out.setdefault((kind, p.move, p.where, p.when), les["text"])
        return out
    a, b = keyed(old), keyed(new)
    kept = set(a) & set(b)
    flips = []
    for k in set(a) - kept:
        opposite = ({"negative": "positive", "positive": "negative"}.get(k[0]),) + k[1:]
        if opposite in b and k[1] is not None:
            flips.append((a[k], b[opposite]))
    return Change(kept=len(kept), added=[b[k] for k in b if k not in kept], dropped=[a[k] for k in a if k not in kept],
                  flipped=sorted(flips), size=(len(a), len(b)))
