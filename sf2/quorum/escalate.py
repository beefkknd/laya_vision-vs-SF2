"""The optional System-2 hook: Qwen picks one of the candidates when the swarm has no quorum.

``make_qwen_pick(ask_qwen)`` adapts the driver's Qwen caller (``ask_qwen(messages, task) -> str``, as
scripts/play_loop_screen._real_qwen) into ``pick(text, candidates, proposals) -> Optional[str]``. Qwen may only
choose among the candidates; anything else (an unparseable reply, a move outside the list, an exception) returns
None and the decider falls back to text-laya. The bridge runs in lockstep, so a slow reply costs wall-clock time,
never game time.
"""
import re
from typing import Callable, Dict, List, Optional, Sequence

from .tally import Proposal

TASK = "quorum_pick"
SYSTEM = ("You break ties for a Street Fighter II player. The fast voters disagree about the next move. "
          "Reply with exactly one move name from the list, nothing else.")


def messages(text: str, candidates: Sequence[str], proposals: Sequence[Proposal]) -> List[Dict]:
    votes = "; ".join("%s says %s (%.2f)" % (p.voter, p.action, p.confidence) for p in proposals)
    user = "%s\nVotes: %s.\nChoose one of: %s." % (text, votes, ", ".join(candidates))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse(reply: str, candidates: Sequence[str]) -> Optional[str]:
    """The candidate named in ``reply``: an exact match first, else the first candidate the reply mentions."""
    r = (reply or "").strip().strip(".").strip()
    if r in candidates:
        return r
    # a move name may contain dots and plus signs (s.hp, throw_F+hp): a dot only continues a name when a word
    # character follows it, so "I pick walk_forward." still names walk_forward
    hits = [(m.start(), c) for c in candidates
            for m in [re.search(r"(?<![\w.+])%s(?![\w+]|\.\w)" % re.escape(c), reply or "")] if m]
    return min(hits)[1] if hits else None


def make_qwen_pick(ask_qwen: Callable[[List[Dict], str], str]):
    def pick(text: str, candidates: Sequence[str], proposals: Sequence[Proposal]) -> Optional[str]:
        try:
            return parse(ask_qwen(messages(text, candidates, proposals), TASK), candidates)
        except Exception:                                    # System 2 is optional: never let it break a round
            return None
    return pick
