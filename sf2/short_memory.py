"""Short memory for System 1: what each attack REALLY did in this session, in the situation it was tried.

After every attack System 1 records the real outcome (read from RAM) under a situation key: the attack, the gap in
16 px buckets and what the opponent was doing (still / attacking / air / guard / stun). At the next decision each
attack's score is the model's P(hit) with that record folded in, the model counting as ``prior`` results:

    score = (prior * p_model + hits) / (prior + tries)

So an attack the model is sure of but that keeps whiffing here drops after a couple of tries, and one that keeps
landing rises. Only the last ``keep`` results count (a short memory); it starts blank.
"""
import collections
import json
from typing import Dict, Tuple

GAP_BUCKET = 16
OPP = {0x00: "still", 0x02: "still", 0x04: "air", 0x06: "still", 0x08: "guard", 0x0A: "attack", 0x0C: "attack",
       0x0E: "stun", 0x14: "stun"}


def situation(gap: int, opp_state: int) -> Tuple[int, str]:
    return gap // GAP_BUCKET, OPP.get(opp_state, "other")


class ShortMemory:
    def __init__(self, prior: float = 2.0, keep: int = 200):
        self.prior = prior
        self.events = collections.deque(maxlen=keep)     # (action, situation, hit)

    def record(self, action: str, sit: Tuple[int, str], outcome: str) -> None:
        self.events.append((action, sit, outcome == "hit"))

    def counts(self, action: str, sit: Tuple[int, str]) -> Tuple[int, int]:
        tries = [hit for a, s, hit in self.events if a == action and s == sit]
        return sum(tries), len(tries)

    def score(self, action: str, sit: Tuple[int, str], p_model: float) -> float:
        hits, tries = self.counts(action, sit)
        return (self.prior * p_model + hits) / (self.prior + tries)

    def to_json(self) -> Dict:
        table = collections.defaultdict(lambda: [0, 0])
        for a, s, hit in self.events:
            key = "%s @ gap %d-%d px, opp %s" % (a, s[0] * GAP_BUCKET, s[0] * GAP_BUCKET + GAP_BUCKET - 1, s[1])
            table[key][0] += hit
            table[key][1] += 1
        return {"prior": self.prior, "keep": self.events.maxlen, "events": len(self.events),
                "hits_tries": {k: v for k, v in sorted(table.items())}}

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=1)
