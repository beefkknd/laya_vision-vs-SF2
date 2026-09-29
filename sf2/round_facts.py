"""The facts of one round, counted by code (System 2 interprets them, never counts): my attack, my defence, what the
opponent showed, what contradicted the notebook, and how last round's plan went. Hit points = damage dealt minus
damage taken (the score for now).

The opponent's moves are known only by kind (jump-in / ground attack / special) and range: RAM does not name them.
"""
import collections
from typing import Dict, List, Sequence

from .advice import FORWARD, read

MIN_TRIES = 3          # a move needs this many tries in a round before it can contradict the notebook


def his_kind(a: Dict) -> str:
    """What the opponent was doing when he hit me, from the action log entry."""
    if a.get("opp_air"):
        return "jump-in"
    return "special" if "special" in a.get("opp_reaction", []) else "ground attack"


def facts(acts: Sequence[Dict], summary: Dict) -> Dict:
    """Counts from one round's action log (sf2.game_log.action_entry) and its summary."""
    mine = collections.defaultdict(lambda: {"tries": 0, "hit": 0, "whiff": 0, "blocked": 0, "punished": 0, "hp": 0})
    for a in acts:
        if a["kind"] == "attack":
            m = mine[(a["action"], a["range"])]
            m["tries"] += 1
            m[a["actual"] if a["actual"] in ("hit", "whiff", "blocked") else "whiff"] += 1
            m["punished"] += bool(a["i_was_hit"])
            m["hp"] += a["dealt"] - a["taken"]
    hit_by = collections.Counter((his_kind(a), a["range"], "attacking" if a["kind"] == "attack" else
                                  "blocking" if a["kind"] == "defense" else "walking") for a in acts if a["i_was_hit"])
    n = max(1, len(acts))
    him = {rng: {"decisions": sum(a["range"] == rng for a in acts),
                 "attacked": sum(a["range"] == rng and a["opp_attacked"] for a in acts),
                 "jumped": sum(a["range"] == rng and a["opp_air"] for a in acts),
                 "blocked me": sum(a["range"] == rng and a["opp_blocked"] for a in acts)}
           for rng in ("close", "mid", "far")}
    return {"result": summary["result"], "dealt": summary["dealt"], "taken": summary["taken"],
            "hp": summary["dealt"] - summary["taken"], "decisions": len(acts),
            "walked": sum(a["action"] == FORWARD for a in acts) / n,
            "blocks": sum(a["kind"] == "defense" for a in acts),
            "mine": {"%s@%s" % k: v for k, v in sorted(mine.items(), key=lambda kv: -kv[1]["tries"])},
            "hit_by": {"%s at %s while I was %s" % k: v for k, v in hit_by.most_common()},
            "him": him}


def unexpected(f: Dict, notebook_lines: Sequence[str], moves: Sequence[str]) -> List[str]:
    """Notebook lines this round contradicted: a move it calls good lost hit points, or one it calls bad gained."""
    out = []
    for line in notebook_lines:
        les = read(line, list(moves) + [FORWARD])
        if not les.move or les.polarity == "none":
            continue
        for key, m in f["mine"].items():
            move, rng = key.split("@")
            if move != les.move or (les.where and les.where != rng) or m["tries"] < MIN_TRIES:
                continue
            if les.polarity in ("soft", "hard") and m["hp"] < 0:
                out.append("'%s' but %s at %s lost %d hit points in %d tries" % (line, move, rng, -m["hp"], m["tries"]))
            elif les.polarity == "neg" and m["hp"] > 0:
                out.append("'%s' but %s at %s gained %d hit points in %d tries" % (line, move, rng, m["hp"], m["tries"]))
    return out


def plan_check(f: Dict, plan: Sequence[str], moves: Sequence[str]) -> List[str]:
    """How each line of last round's plan went: was its move used, and what it scored."""
    out = []
    for line in plan:
        les = read(line, list(moves) + [FORWARD])
        if not les.move:
            continue
        rows = [m for k, m in f["mine"].items() if k.split("@")[0] == les.move and
                (les.where is None or k.split("@")[1] == les.where)]
        tries, hp = sum(m["tries"] for m in rows), sum(m["hp"] for m in rows)
        out.append("'%s': %s" % (line, "not used" if not tries else "used %d times, %+d hit points" % (tries, hp)))
    return out


def text(f: Dict, surprises: Sequence[str], plan_lines: Sequence[str], opp: str) -> str:
    """The facts as the plain table Qwen reads."""
    lines = ["Round result: %s. Hit points: dealt %d, taken %d, net %+d. %d decisions, walked in %.0f%%, blocks %d." % (
        f["result"], f["dealt"], f["taken"], f["hp"], f["decisions"], 100 * f["walked"], f["blocks"])]
    lines.append("MY ATTACKS (move@range: tries, hit/whiff/blocked, punished, net hit points):")
    for k, m in f["mine"].items():
        lines.append("  %s: %d tries, %d/%d/%d, punished %d, net %+d" % (
            k, m["tries"], m["hit"], m["whiff"], m["blocked"], m["punished"], m["hp"]))
    lines.append("HOW %s HIT ME (his attack kind at range, what I was doing: times):" % opp.upper())
    lines += ["  %s: %d" % kv for kv in f["hit_by"].items()] or ["  never"]
    lines.append("WHAT %s DID (per range: my decisions there, he attacked, he jumped, he blocked me):" % opp.upper())
    for rng, h in f["him"].items():
        lines.append("  %s: %d, %d, %d, %d" % (rng, h["decisions"], h["attacked"], h["jumped"], h["blocked me"]))
    lines.append("UNEXPECTED (the notebook said otherwise):")
    lines += ["  " + s for s in surprises] or ["  nothing"]
    lines.append("LAST ROUND'S PLAN, HOW IT WENT:")
    lines += ["  " + s for s in plan_lines] or ["  (no plan yet)"]
    return "\n".join(lines)
