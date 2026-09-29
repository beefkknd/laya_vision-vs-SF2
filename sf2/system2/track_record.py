"""The track record: per opponent and lesson line, how the rounds it was in play went against the same round without
advice (the paired no-advice arm), over earlier lesson-loop runs.

The verifier (sf2.system2.lessons) judges a lesson inside one situation: its move against her other moves there. That
cannot see where a lesson takes her - round 3 (2026-09-29): "use more forward at mid range when he jumps" was better
than her other choices there in every Honda run, while its rounds lost 16 hp to no advice. The round outcome can, but
only across runs, so the record is built from finished runs into a frozen file (scripts/track_record.py) and a loop
reads it (qwen_lessons.py --track): Qwen sees it, and a lesson whose record clearly hurts is refused.

The run is the unit (rounds of one run share its other lessons and its luck): the mean is the mean of the per-run
means, the 95% interval a two-level bootstrap (runs, then rounds within each). Lessons in play together share the
credit - a record is a warning with an interval, not proof of cause.
"""
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

from ..eval.stats import BOOT_SEED, REPS, paired

MIN_RUNS = 3           # a record needs this many runs with the lesson in play
MIN_ROUNDS = 15        # and this many rounds in all
SHOWN = 10             # lines shown to Qwen
Record = Dict[str, object]


def _read(path: str) -> List[Dict]:
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def _runs(roots: Sequence[str]) -> List[Tuple[str, str]]:
    """(run dir, opponent) of every finished run (verdict.json written), in name order."""
    out = []
    for root in roots:
        for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
            d = os.path.join(root, name)
            if os.path.exists(os.path.join(d, "verdict.json")):
                out.append((d, name.split("_")[1]))
    return out


def _interval(groups: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    means = [sum(g) / len(g) for g in groups]
    rng, boot = random.Random(BOOT_SEED), []
    for _ in range(REPS):
        pick = [groups[rng.randrange(len(groups))] for _ in groups]
        boot.append(sum(sum(rng.choice(g) for _ in g) / len(g) for g in pick) / len(pick))
    boot.sort()
    return sum(means) / len(means), boot[int(0.025 * REPS)], boot[int(0.975 * REPS) - 1]


def record(groups: Sequence[Sequence[float]]) -> Record:
    """One line's record from its per-run lists of paired hp differences."""
    groups = [list(g) for g in groups if g]
    n = sum(len(g) for g in groups)
    if not groups:
        return {"runs": 0, "rounds": 0, "mean": 0.0, "lo": 0.0, "hi": 0.0, "verdict": "few"}
    mean, lo, hi = _interval(groups)
    few = len(groups) < MIN_RUNS or n < MIN_ROUNDS
    return {"runs": len(groups), "rounds": n, "mean": round(mean, 2), "lo": round(lo, 2), "hi": round(hi, 2),
            "verdict": "few" if few else "hurts" if hi < 0 else "helps" if lo > 0 else "unclear"}


def build(roots: Sequence[str]) -> Dict:
    """{"opponents": {opp: {line: record}}, "sources": [run dirs], "skipped": [run dirs whose arms do not pair]}."""
    by: Dict[str, Dict[str, List[List[float]]]] = {}
    sources, skipped = [], []
    for d, opp in _runs(roots):
        try:
            loop, none = _read(os.path.join(d, "loop", "rounds.jsonl")), _read(os.path.join(d, "none", "rounds.jsonl"))
            diffs = paired(loop, none)
        except (OSError, ValueError, KeyError):
            skipped.append(d)
            continue
        sources.append(d)
        per_line: Dict[str, List[float]] = {}
        for r, diff in zip(loop, diffs):
            for line in r.get("lines", []):
                per_line.setdefault(line, []).append(diff)
        for line, ds in per_line.items():
            by.setdefault(opp, {}).setdefault(line, []).append(ds)
    return {"opponents": {opp: {line: record(gs) for line, gs in sorted(lines.items())}
                          for opp, lines in sorted(by.items())},
            "sources": sources, "skipped": skipped}


def hurts(track: Dict[str, Record], line: str) -> bool:
    return bool(track) and track.get(line, {}).get("verdict") == "hurts"


def describe(line: str, r: Record) -> str:
    return "%s: %d rounds in %d runs, %+.1f hp per round vs no advice [%+.1f, %+.1f]: %s" % (
        line, r["rounds"], r["runs"], r["mean"], r["lo"], r["hi"], r["verdict"])


def prompt_lines(track: Dict[str, Record]) -> List[str]:
    """The record for Qwen: clear verdicts first (hurts, then helps), worst first; at most SHOWN lines."""
    order = {"hurts": 0, "helps": 1, "unclear": 2, "few": 3}
    ranked = sorted(track.items(), key=lambda kv: (order[kv[1]["verdict"]], kv[1]["mean"]))
    return ["- " + describe(line, r) for line, r in ranked if r["verdict"] != "few"][:SHOWN]


PROMPT_HEAD = ("TRACK RECORD against %s in earlier sessions - the rounds each lesson was in play, against the same "
               "round played without advice (hurts / helps = clearly worse / better; lessons in play together share "
               "the credit; a lesson that hurts will be refused):")


def prompt_part(opp: str, track: Dict[str, Record]) -> str:
    return PROMPT_HEAD % opp + "\n" + ("\n".join(prompt_lines(track)) or "(nothing clear yet)")


def violations(reg: Sequence[Dict], track: Dict[str, Record]) -> List[str]:
    """Registered lessons whose record hurts (must be empty: the verifier refuses them)."""
    return [r["line"] for r in reg if r["state"] == "registered" and hurts(track, r["line"])]
