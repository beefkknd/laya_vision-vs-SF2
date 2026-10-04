"""The forced-exploration pool for the simple live policy (sf2/system2/short_memory.py).

When she is losing and the Coach (Qwen) proposes nothing usable, the survival loop falls back on one of
these grounded, situational offensive rules so her behaviour MUST change (escape the losing repetition).
Every rule is valid for any of the 8 fighters (standing normals / movement / throw). ``pick`` returns a
CLAIM (so it is admitted as a `trying` line, measured like any other -- never an unmeasured `verified`),
and skips a pool rule already COVERED by a line in play (range-agnostic 'use more s.mk when he stands'
covers 'use more s.mk at mid range ...'), not just exact-string duplicates.
"""
from typing import Dict, Optional, Sequence

EXPLORE_POOL = (
    "use more s.mk at mid range when he stands",
    "use more walk_forward at mid range when he stands",
    "use more cl.hk up close when he stands",        # cl.* (close normal): FOLLOWABLE up close -- 's.hk up close' was not
    "use more throw up close when he stands",
    "use more s.mp at mid range when he attacks",
    "use more walk_back at mid range when he attacks",
)


def render_line(claim: Optional[Dict]) -> str:
    """A claim's advice line (or "" for None) -- for logging/tests."""
    from sf2.system2 import lessons as L
    return L.render(claim) if claim else ""


def _covered(in_claims: Sequence[Dict], c: Dict) -> bool:
    """True if some in-play claim applies wherever ``c`` does, in the same direction (so ``c`` adds nothing)."""
    from sf2.system2 import lessons as L
    return any(L.RIGHT[ic["kind"]] == L.RIGHT[c["kind"]] and L._covers(ic, c) for ic in in_claims)


def pick(in_play_lines: Sequence[str], moves: set, rotate: int = 0, followable=None) -> Optional[Dict]:
    """The next pool rule NOT already covered in play AND followable, as a claim dict (rotating by ``rotate``).
    None when the pool is exhausted. ``followable(move, range)`` (optional) skips a rule text-laya can't play at
    its range. Invalid pool lines for this move set are skipped (defensive; the pool is validated)."""
    from sf2.system2.rule_entry import claim_of
    in_claims = []
    for line in in_play_lines:
        try:
            in_claims.append(claim_of(line, moves))
        except ValueError:
            continue                      # an in-play line we cannot parse just does not cover anything
    n = len(EXPLORE_POOL)
    for k in range(n):
        line = EXPLORE_POOL[(rotate + k) % n]
        try:
            c = claim_of(line, moves)
        except ValueError:
            continue
        if followable is not None and not followable(c["move"], c.get("range")):
            continue
        if not _covered(in_claims, c):
            return c
    return None
