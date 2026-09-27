"""Build the two memories (sf2/memory.py) for every character from a game log: a stand-in for System 2, with fixed
rules so the result is repeatable and every lesson points at its evidence in the log.

    python scripts/build_memory.py --log rollouts/games_v1

Playbook (long term, the character's own moves by range, from every opponent in the log):
    use_more  a move at a range that hits >= USE_RATE of >= MIN_TRIES tries
    avoid     a move at a range that hits <= AVOID_RATE, or gets punished (I am hit before my next turn) >= PUNISH_RATE
Short memory (this opponent only, at most sf2.memory.MAX_PROMPT_LESSONS, the ones that go into laya's prompt):
    avoid           what this opponent punishes
    use_more        what works best against him
    counter         what hit him while he was attacking or jumping (or that nothing did)
    opponent_habit  what he does at a range on >= HABIT_SHARE of my turns (attack, jump, guard)
"""
import argparse
import collections
import json
import os
import sys
from typing import Dict, List, Tuple

import _path  # noqa: F401
from sf2.memory import MAX_PROMPT_LESSONS, check, playbook_path, short_path
from sf2.vs_sweep import actions

MIN_TRIES = 6
USE_RATE, AVOID_RATE, PUNISH_RATE, HABIT_SHARE = 0.55, 0.25, 0.35, 0.25
MAX_PLAYBOOK = 8
HABITS = {"attack": "attacks", "jump": "jumps in", "guard": "guards"}


def ref(a: Dict) -> str:
    return "g%df%d" % (a["game"], a["frame"])


def lesson(text: str, kind: str, action, rng, group: List[Dict], counts) -> Dict:
    """``counts``: which of the ``group`` actions the lesson's claim is about (landed, punished, a habit)."""
    n = sum(1 for a in group if counts(a))
    return {"text": text, "kind": kind, "action": action, "range": rng,
            "evidence": {"tries": len(group), "count": n, "rate": round(n / len(group), 2),
                         "refs": [ref(a) for a in group if counts(a)][:5]}}


def by_move(acts: List[Dict]) -> Dict[Tuple[str, str], List[Dict]]:
    out = collections.defaultdict(list)
    for a in acts:
        if a["kind"] == "attack":
            out[(a["action"], a["range"])].append(a)
    return out


def playbook(me: str, acts: List[Dict]) -> List[Dict]:
    out = []
    for (act, rng), g in sorted(by_move(acts).items(), key=lambda kv: -len(kv[1])):
        if len(g) < MIN_TRIES:
            continue
        rate = sum(a["actual"] == "hit" for a in g) / len(g)
        punished = sum(a["i_was_hit"] for a in g) / len(g)
        if rate >= USE_RATE:
            out.append(lesson("%s at %s range lands (%d%%)" % (act, rng, 100 * rate), "use_more", act, rng, g,
                              lambda a: a["actual"] == "hit"))
        elif rate <= AVOID_RATE or punished >= PUNISH_RATE:
            if punished >= PUNISH_RATE:     # the evidence counts what the lesson says: punishes, or landings
                out.append(lesson("avoid %s at %s range: gets punished %d%%" % (act, rng, 100 * punished), "avoid",
                                  act, rng, g, lambda a: a["i_was_hit"]))
            else:
                out.append(lesson("avoid %s at %s range: lands %d%%" % (act, rng, 100 * rate), "avoid", act, rng, g,
                                  lambda a: a["actual"] == "hit"))
    return out[:MAX_PLAYBOOK]


def short_memory(me: str, opp: str, acts: List[Dict]) -> List[Dict]:
    moves = {k: g for k, g in by_move(acts).items() if len(g) >= MIN_TRIES}
    rate = {k: sum(a["actual"] == "hit" for a in g) / len(g) for k, g in moves.items()}
    punish = {k: sum(a["i_was_hit"] for a in g) / len(g) for k, g in moves.items()}
    out = []
    worst = sorted((k for k in moves if punish[k] >= PUNISH_RATE), key=lambda k: -punish[k])[:1]
    for act, rng in worst:
        out.append(lesson("%s punishes my %s at %s range (%d%%)" % (opp, act, rng, 100 * punish[(act, rng)]),
                          "avoid", act, rng, moves[(act, rng)], lambda a: a["i_was_hit"]))
    # a move he punishes is not also recommended (both would be true, but as instructions they contradict)
    best = sorted((k for k in moves if rate[k] >= USE_RATE and k not in worst),
                  key=lambda k: -rate[k] * len(moves[k]))[:2]
    for act, rng in best:
        out.append(lesson("use more %s at %s range (lands %d%%)" % (act, rng, 100 * rate[(act, rng)]), "use_more",
                          act, rng, moves[(act, rng)], lambda a: a["actual"] == "hit"))
    for state, verb in (("attack", "attacks"), ("jump", "jumps in")):
        g = [a for a in acts if a["kind"] == "attack" and a["opp_state"] == state]
        if len(g) < MIN_TRIES:
            continue
        per = collections.defaultdict(list)
        for a in g:
            per[a["action"]].append(a)
        good = [(m, x) for m, x in per.items() if len(x) >= 3 and sum(a["actual"] == "hit" for a in x) / len(x) >= 0.5]
        if good:
            m, x = max(good, key=lambda mx: sum(a["actual"] == "hit" for a in mx[1]))
            out.append(lesson("when %s %s, %s hits him" % (opp, verb, m), "counter", m, None, x,
                              lambda a: a["actual"] == "hit"))
        elif sum(a["actual"] == "hit" for a in g) / len(g) <= AVOID_RATE:
            out.append(lesson("when %s %s, my attacks rarely land (%d of %d)" % (
                opp, verb, sum(a["actual"] == "hit" for a in g), len(g)), "counter", None, None, g,
                lambda a: a["actual"] == "hit"))
    per_range = collections.defaultdict(list)
    for a in acts:
        per_range[a["range"]].append(a)
    habits = []
    for rng, g in per_range.items():
        for state, verb in HABITS.items():
            share = sum(a["opp_state"] == state for a in g) / len(g)
            if len(g) >= MIN_TRIES and share >= HABIT_SHARE:
                habits.append((share, lesson("%s %s a lot at %s range (%d%% of my turns)" % (opp, verb, rng, 100 * share),
                                             "opponent_habit", None, rng, g, lambda a, s=state: a["opp_state"] == s)))
    out += [h for _, h in sorted(habits, key=lambda sh: -sh[0])]
    return out[:MAX_PROMPT_LESSONS]


def write(path: str, mem: Dict, me: str) -> None:
    problems = check(mem, list(actions(me)))
    if problems:
        raise SystemExit("%s would be malformed:\n  %s" % (path, "\n  ".join(problems)))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(mem, f, indent=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="a play_system1.py output dir, e.g. rollouts/games_v1")
    args = ap.parse_args()
    for me in sorted(d for d in os.listdir(args.log) if os.path.isdir(os.path.join(args.log, d))):
        acts = [json.loads(x) for x in open(os.path.join(args.log, me, "actions.jsonl"))]
        opps = sorted({a["opp"] for a in acts})
        src = os.path.join(args.log, me, "actions.jsonl")
        pb = {"me": me, "source": [src], "lessons": playbook(me, acts)}
        write(playbook_path(me), pb, me)
        print("== %s playbook (%d lessons)" % (me, len(pb["lessons"])))
        for les in pb["lessons"]:
            print("   %-9s %s  [%d/%d]" % (les["kind"], les["text"], les["evidence"]["count"], les["evidence"]["tries"]))
        for opp in opps:
            sm = {"me": me, "opp": opp, "source": [src], "lessons": short_memory(me, opp, [a for a in acts
                                                                                          if a["opp"] == opp])}
            write(short_path(me, opp), sm, me)
            print("   short memory vs %s (%d lessons)" % (opp, len(sm["lessons"])))
            for les in sm["lessons"]:
                print("   %-15s %s  [%d/%d]" % (les["kind"], les["text"], les["evidence"]["count"],
                                               les["evidence"]["tries"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
