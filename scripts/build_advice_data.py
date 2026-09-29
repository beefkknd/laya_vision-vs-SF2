"""Training data for text laya (sf2.advice): real moments from every character's game logs, laya-vision's real
top-3 ratings, advice lessons (Qwen's real ones plus templated ones over every polarity word and condition), and the
answer from the label rule. Nothing opponent-specific: the lessons name the character's own moves only.

    python scripts/build_advice_data.py                     # -> test_data/advice/{train,val,test}.jsonl + stats
    python scripts/build_advice_data.py --per-char 2000

Splits: moments by log file (a file's moments are in one split only); templated wordings: some are held out of
training entirely (TEST_WORDS) so test measures following the MEANING of a word it never trained on; real Qwen lessons
by text (each real lesson in one split only). Log: logs/advice/build.log.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import random
from dataclasses import replace
from typing import Dict, List, Tuple

import _path  # noqa: F401
from sf2.advice import (FAILS, FORWARD, MAY, WORKS, answers, opp_doing, parse, prompt,
                        question, rating, situation_text)
from sf2.vocab import OPP_STATES, RANGES, bar
from sf2.data.vs_sweep import MOVEMENT, SPECIALS, actions

OUT = "test_data/advice"
CHARS = sorted(SPECIALS)

WHERE_WORDS = {"close": ["up close", "at close range"], "mid": ["at mid", "at mid range"], "far": ["far away", "at far"]}
WHEN_WORDS = {"jumping": "when he jumps", "crouching": "when he crouches", "attacking": "when he attacks",
              "standing": "when he stands", "stunned": "when he is stunned"}
TRAIN_WORDS = {
    "soft": ["use more {m}", "prefer {m}", "go for {m}", "use {m}", "hit him with {m}"],
    "hard": ["always use {m}", "only use {m}"],
    "neg": ["avoid {m}", "never use {m}", "stop using {m}", "use less {m}", "don't use {m}", "no {m}"],
}
TEST_WORDS = {           # never in training: does it generalise the polarity of a new word?
    "soft": ["lean on {m}", "try {m} more often"],
    "hard": ["always go for {m}"],
    "neg": ["fewer {m}", "quit using {m}", "do not throw {m}"],
}
REASONS = {"soft": ["it lands", "he does not punish it", "it works"], "hard": ["it is the only thing that works"],
           "neg": ["it whiffs", "he punishes it", "it misses"]}
CASES = {"hard": 0.1, "soft": 0.2, "neg": 0.2, "condition_off": 0.2, "plain": 0.3}   # share of each split
HABITS = ["he jumps a lot {w}, be ready", "he crouches a lot {w}", "he attacks a lot {w}"]


def moments() -> Dict[str, List[Tuple[str, Dict]]]:
    """character -> [(log file, action entry)] for every logged decision with laya-vision's top-3."""
    out = collections.defaultdict(list)
    for f in sorted(glob.glob("rollouts/**/actions.jsonl", recursive=True)):
        for line in open(f):
            r = json.loads(line)
            if r.get("top3") and r.get("me") in SPECIALS:
                out[r["me"]].append((f, r))
    return out


def real_lessons() -> Dict[str, List[str]]:
    seen = collections.defaultdict(set)
    for f in glob.glob("logs/system2/memory/*.json") + glob.glob("memory/*/*.json"):
        d = json.load(open(f))
        for les in d.get("lessons", []):
            seen[d["me"]].add(les["text"])
    return {me: sorted(t) for me, t in seen.items()}


def condition(rng: random.Random, sit_range: str, sit_doing: str) -> str:
    """A condition phrase: half the time one that holds now, else one that does not (or none)."""
    kind = rng.choice(["none", "where", "when"])
    if kind == "none":
        return ""
    hold = rng.random() < 0.5
    if kind == "where":
        r = sit_range if hold else rng.choice([x for x in RANGES if x != sit_range])
        return rng.choice(WHERE_WORDS[r])
    s = sit_doing if hold else rng.choice([x for x in OPP_STATES if x != sit_doing])
    return WHEN_WORDS[s]


def templated(rng: random.Random, moves: List[str], sit: Tuple[str, str], words: Dict) -> str:
    if rng.random() < 0.1:
        return rng.choice(HABITS).format(w=rng.choice(sum(WHERE_WORDS.values(), [])))
    pol = rng.choices(["soft", "hard", "neg"], weights=[4, 1, 4])[0]
    m = rng.choice(moves)
    text = rng.choice(words[pol]).format(m=m)
    cond = condition(rng, *sit)
    if cond.startswith("when") and rng.random() < 0.5:
        text = cond + ", " + text
    elif cond:
        text = text + " " + cond
    if rng.random() < 0.4:
        text += ": " + rng.choice(REASONS[pol])
    return text


def example(rng: random.Random, me: str, entry: Dict, lesson_pool: List[str], words: Dict) -> Dict:
    moves = [a for a in actions(me) if a not in MOVEMENT] + ([FORWARD] if FORWARD in MOVEMENT else [])
    sit = (entry["range"], opp_doing(entry))
    my_bar, opp_bar = bar(entry["my_life"]), bar(entry["opp_life"])
    options = {m: rating(p) for m, p in entry["top3"] if m in moves and m != FORWARD}
    lessons = []
    for _ in range(rng.choices([0, 1, 2, 3], weights=[1, 4, 3, 2])[0]):
        if lesson_pool and rng.random() < 0.3:
            lessons.append(rng.choice(lesson_pool))
        else:
            lessons.append(templated(rng, [m for m in moves if m != FORWARD], sit, words))
    parsed = [parse(t, moves) for t in lessons]
    for les in parsed:            # the move a lesson names is always on the shortlist (System 1 adds it)
        if les.move and les.move != FORWARD and les.move not in options:
            options[les.move] = rng.choices([WORKS, MAY, FAILS], weights=[1, 2, 3])[0]
    options[FORWARD] = None
    order = list(options)
    rng.shuffle(order)
    options = {m: options[m] for m in order}
    good, rule = answers(sit[0], sit[1], options, parsed)
    return {"me": me, "text": prompt(situation_text(sit[0], sit[1], my_bar, opp_bar), lessons),
            "question": question(options), "answers": good, "rule": rule, "lessons": lessons,
            "case": case(sit, options, parsed, good, rule)}


def case(sit: Tuple[str, str], options: Dict, parsed: List, good: List[str], rule: str) -> str:
    """What the example tests: following a hard/soft lesson, a negative lesson that changed the answer, a condition
    that did NOT hold and so changed the answer (the lesson must be ignored), or plain (vision decides)."""
    if rule in ("hard", "soft"):
        return rule
    everywhere = [replace(les, where=None, when=None) for les in parsed]
    if answers(sit[0], sit[1], options, everywhere)[0] != good:
        return "condition_off"
    if answers(sit[0], sit[1], options, [])[0] != good:
        return "neg"
    return "plain"


def text_split(text: str) -> str:
    """A real lesson's split, by its text alone: the same sentence (written for two characters) is in one split."""
    h = int(hashlib.sha1(text.lower().encode()).hexdigest(), 16) % 10
    return "test" if h < 2 else "val" if h == 2 else "train"


def build(per_char: int, seed: int) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    by_char, real = moments(), real_lessons()
    splits = collections.defaultdict(list)
    for me in CHARS:
        files = sorted({f for f, _ in by_char[me]})
        rng.shuffle(files)
        n_test = max(1, len(files) // 6)
        file_split = {f: ("test" if i < n_test else "val" if i == n_test and len(files) > 2 else "train")
                      for i, f in enumerate(files)}
        pool = real.get(me, [])
        rng.shuffle(pool)
        real_split = {t: text_split(t) for t in pool}
        for split, n in (("train", per_char), ("val", per_char // 16), ("test", per_char // 8)):
            rows = [e for f, e in by_char[me] if file_split[f] == split] or [e for _, e in by_char[me]]
            lp = [t for t in pool if real_split[t] == split]
            quota = {c: round(n * share) for c, share in CASES.items()}
            while any(quota.values()):
                words = TRAIN_WORDS if split != "test" or rng.random() < 0.5 else TEST_WORDS
                ex = example(rng, me, rng.choice(rows), lp, words)
                if quota[ex["case"]]:
                    quota[ex["case"]] -= 1
                    splits[split].append(dict(ex, words="held_out" if words is TEST_WORDS else "trained"))
    for split, rows in splits.items():
        rng.shuffle(rows)
        for i, r in enumerate(rows):
            r["id"] = "%s_%05d" % (split, i)
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
            stats = "%-5s %5d rows  chars %s\n      cases %s\n      answer is forward %d" % (
                split, len(rows), dict(collections.Counter(r["me"] for r in rows)),
                dict(collections.Counter(r["case"] for r in rows)),
                sum(r["answers"] == [FORWARD] for r in rows))
            print(stats)
            log.write(stats + "\n")


if __name__ == "__main__":
    main()
