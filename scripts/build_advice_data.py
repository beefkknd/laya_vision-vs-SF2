"""Two-stage training data for text laya (sf2.system1.advice + sf2.system1.action_menu): reading-comprehension over the
situation WORDS and Qwen's advice LINES, never game outcomes and never laya-vision ratings. Each decision is TWO rows:

    round "cat"   which of the 7 categories (move/punch/kick/block/throw/special/combo) — options are the bare names;
    round "move"  which move inside the chosen category — options are that category's moves PRUNED to the stance.

The label follows the advice (advice.two_stage): an applying hard line, else an applying soft line, else the hardcoded
default (block, action_menu.DEFAULT_MOVE); a negative line rules its move out. A line applies only when its conditions
hold (range / what he does / a fireball coming) AND its move is one this stance can do. The situation prunes the stance
(advice.stance_of: grounded+far -> standing, grounded+close -> close, crouch -> c., air -> j./jf.).

    ~/work/laya_mlx/.venv/bin/python scripts/build_advice_data.py                 # -> test_data/advice/{train,val,test}.jsonl
    ~/work/laya_mlx/.venv/bin/python scripts/build_advice_data.py --per-char 2000

Chun-Li only (the two-stage menu is her move set, sf2.data.vs_moves.chunli()); opponent-agnostic (advice names her own
moves, the opponent appears only through the generic states). Some polarity wordings are held out of training entirely
(TEST_WORDS) so the test split measures following the MEANING of a phrasing it never trained on. Log: logs/advice/build.log.
"""
import argparse
import collections
import json
import os
import random
from typing import Dict, List, Tuple

import _path  # noqa: F401
from sf2.system1.action_menu import CATEGORIES, CATEGORY_ORDER, DEFAULT_MOVE, category_of
from sf2.system1.advice import (available_moves, category_question, move_question, moves_in_stance, parse, prompt,
                                 situation_text, stance_of, two_stage)
from sf2.vocab import BARS, OPP_STATES, RANGES

OUT = "test_data/advice"
ME = "chunli"
MOVES = [m for cat in CATEGORY_ORDER for m in CATEGORIES[cat]]
POSTURES = ("stand", "crouch", "air")

WHERE_WORDS = {"close": ["up close", "at close range", "next to him"],
               "mid": ["at mid", "at mid range", "in the middle"], "far": ["far away", "at far", "at long range"]}
WHEN_WORDS = {"jumping": "when he jumps", "crouching": "when he crouches", "attacking": "when he attacks",
              "standing": "when he stands", "stunned": "when he is stunned"}
FIRE_WORDS = ["when a fireball comes", "when a fireball is coming", "against a fireball", "when a fireball approaches"]
TRAIN_WORDS = {
    "soft": ["use more {m}", "prefer {m}", "go for {m}", "use {m}", "hit him with {m}"],
    "hard": ["always use {m}", "only use {m}"],
    "neg": ["avoid {m}", "never use {m}", "stop using {m}", "use less {m}", "don't use {m}", "no {m}"],
}
TEST_WORDS = {           # never in training: does it generalise the polarity of a new phrasing?
    "soft": ["lean on {m}", "try {m} more often"],
    "hard": ["always go for {m}"],
    "neg": ["fewer {m}", "quit using {m}", "do not use {m}"],
}
REASONS = {"soft": ["it lands", "he does not punish it", "it works"], "hard": ["it is the only thing that works"],
           "neg": ["it whiffs", "he punishes it", "it misses"]}
HABITS = ["he jumps a lot {w}, be ready", "he crouches a lot {w}", "he attacks a lot {w}"]
CASES = {"hard": 0.12, "soft": 0.24, "neg": 0.16, "fireball": 0.18, "condition_off": 0.15, "plain": 0.15}


def rand_sit(rng: random.Random) -> Tuple[str, str, str, str]:
    return rng.choice(RANGES), rng.choice(OPP_STATES), rng.choice(BARS), rng.choice(BARS)


def line(rng: random.Random, move: str, pol: str, words: Dict,
         where: str = None, when: str = None, fire: bool = False) -> str:
    """One advice line: the polarity phrasing of ``move`` plus any conditions, messily ordered, sometimes a reason."""
    text = rng.choice(words[pol]).format(m=move)
    conds = []
    if when:
        conds.append(WHEN_WORDS[when])
    if where:
        conds.append(rng.choice(WHERE_WORDS[where]))
    if fire:
        conds.append(rng.choice(FIRE_WORDS))
    rng.shuffle(conds)
    lead = next((c for c in conds if c.startswith("when")), None) if rng.random() < 0.5 else None
    if lead:
        rest = [c for c in conds if c is not lead]
        text = lead + ", " + text + ((" " + " ".join(rest)) if rest else "")
    elif conds:
        text = text + " " + " ".join(conds)
    if rng.random() < 0.4:
        text += ": " + rng.choice(REASONS[pol])
    return text


def distractors(rng: random.Random, target: str) -> List[str]:
    """0-2 lines that never change the answer: opponent habits, and negatives on OTHER moves (with no ratings a move
    is only ever chosen by a positive line naming it, and the default is never gated by a negative)."""
    out = []
    for _ in range(rng.choice([0, 0, 1, 1, 2])):
        if rng.random() < 0.5:
            out.append(rng.choice(HABITS).format(w=rng.choice(sum(WHERE_WORDS.values(), []))))
        else:
            other = rng.choice([m for m in MOVES if m != target])
            out.append(rng.choice(TRAIN_WORDS["neg"]).format(m=other))
    return out


def hold(rng: random.Random, p: float, train: Dict, held: Dict) -> Tuple[Dict, str]:
    return (held, "held_out") if rng.random() < p else (train, "trained")


def make(rng: random.Random, case: str, hold_p: float) -> Dict:
    """One decision realised as the intended ``case``: the situation, stance, whether a fireball is on screen, and the
    advice lines. ``hold_p`` is the chance a polarity phrasing is drawn from the held-out set."""
    rng_, doing, my_bar, opp_bar = rand_sit(rng)
    posture = rng.choice(("stand", "crouch") if case in ("neg", "fireball", "condition_off") else POSTURES)
    stance = stance_of(posture, rng_)
    avail = sorted(available_moves(stance))
    fire_sit = False
    lessons: List[str] = []
    tags: List[str] = []

    def words_of():
        w, tag = hold(rng, hold_p, TRAIN_WORDS, TEST_WORDS)
        tags.append(tag)
        return w

    if case == "plain":
        lessons = distractors(rng, DEFAULT_MOVE) or [rng.choice(HABITS).format(w=rng.choice(WHERE_WORDS[rng_]))]
        fire_sit = rng.random() < 0.15
    elif case in ("hard", "soft"):
        target = rng.choice(avail)
        where = rng_ if rng.random() < 0.5 else None
        when = doing if rng.random() < 0.5 else None
        lessons = [line(rng, target, case, words_of(), where=where, when=when)] + distractors(rng, target)
        fire_sit = rng.random() < 0.15
    elif case == "neg":                                   # a soft line names X, a negative line rules X out -> default
        target = rng.choice(avail)
        lessons = [line(rng, target, "soft", words_of()), line(rng, target, "neg", words_of())]
        lessons += distractors(rng, target)
        fire_sit = rng.random() < 0.15
    elif case == "fireball":                              # a line conditioned on a fireball; present -> follow, else default
        target = rng.choice(avail)
        pol = rng.choice(["soft", "hard"])
        fire_sit = rng.random() < 0.5
        where = rng_ if rng.random() < 0.4 else None
        lessons = [line(rng, target, pol, words_of(), where=where, fire=True)] + distractors(rng, target)
    else:                                                 # condition_off: a positive line that does not apply here
        target = rng.choice(MOVES)
        pol = rng.choice(["soft", "hard"])
        if target in avail and rng.random() < 0.6:        # off by a range/state condition that does not hold
            where = rng.choice([x for x in RANGES if x != rng_]) if rng.random() < 0.5 else None
            when = rng.choice([x for x in OPP_STATES if x != doing]) if not where else None
            lessons = [line(rng, target, pol, words_of(), where=where, when=when)]
        else:                                             # off because the move is not one this stance can do
            off = rng.choice([m for m in MOVES if m not in avail]) if len(avail) < len(MOVES) else target
            lessons = [line(rng, off, pol, words_of(), where=rng_)]
        lessons += distractors(rng, target)
        fire_sit = rng.random() < 0.15

    rng.shuffle(lessons)
    parsed = [parse(t, MOVES) for t in lessons]
    cats, moves, rule = two_stage(rng_, doing, stance, parsed, fire_sit)
    return {"rng": rng_, "doing": doing, "my_bar": my_bar, "opp_bar": opp_bar, "stance": stance, "fire": fire_sit,
            "lessons": lessons, "cats": cats, "moves": moves, "rule": rule, "case": case,
            "words": "held_out" if "held_out" in tags else "trained"}


EXPECT = {"hard": {"hard"}, "soft": {"soft"}, "neg": {"default"}, "condition_off": {"default"}, "plain": {"default"},
          "fireball": {"hard", "soft", "default"}}


def rows_of(rng: random.Random, d: Dict, words_tag: str) -> List[Dict]:
    """The round-1 and round-2 rows of one decision (same prompt, different question)."""
    sit = situation_text(d["rng"], d["doing"], d["my_bar"], d["opp_bar"], fireball=d["fire"])
    text = prompt(sit, d["lessons"])
    chosen_cat = category_of(d["moves"][0]) if d["rule"] != "default" else "block"
    move_answers = [m for m in d["moves"] if category_of(m) == chosen_cat]
    options = moves_in_stance(chosen_cat, d["stance"])
    base = {"me": ME, "text": text, "lessons": d["lessons"], "rule": d["rule"], "case": d["case"],
            "stance": d["stance"], "fireball": d["fire"], "words": words_tag}
    return [dict(base, round="cat", question=category_question(), answers=d["cats"]),
            dict(base, round="move", question=move_question(options), answers=move_answers)]


def build(per_char: int, seed: int) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    splits: Dict[str, List[Dict]] = collections.defaultdict(list)
    for split, n, hold_p in (("train", per_char, 0.0), ("val", max(8, per_char // 16), 0.0),
                             ("test", max(16, per_char // 8), 0.5)):
        quota = {c: round(n * s) for c, s in CASES.items()}
        while any(v > 0 for v in quota.values()):
            case = rng.choices(list(quota), weights=[max(0, v) for v in quota.values()])[0]
            for _ in range(20):
                d = make(rng, case, hold_p)
                if d["rule"] in EXPECT[case]:
                    break
            quota[case] -= 1
            splits[split].extend(rows_of(rng, d, d["words"]))
    for split, rows in splits.items():
        rng.shuffle(rows)
        for i, r in enumerate(rows):
            r["id"] = "%s_%06d" % (split, i)
    return splits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-char", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    splits = build(args.per_char, args.seed)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs("logs/advice", exist_ok=True)
    with open("logs/advice/build.log", "w") as log:
        for split, rows in splits.items():
            with open(os.path.join(OUT, split + ".jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            stats = "%-5s %6d rows  round %s\n      cases %s\n      rule %s  held_out %d" % (
                split, len(rows), dict(collections.Counter(r["round"] for r in rows)),
                dict(collections.Counter(r["case"] for r in rows)),
                dict(collections.Counter(r["rule"] for r in rows)),
                sum(r["words"] == "held_out" for r in rows))
            print(stats)
            log.write(stats + "\n")


if __name__ == "__main__":
    main()
