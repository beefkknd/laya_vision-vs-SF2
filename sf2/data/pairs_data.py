"""The movement-pairs datasets (docs/prereg_movement_pairs.md; scripts/build_pairs_data.py): one collection
(scripts/collect_pairs.py, <root>/<A>_vs_<B>/) -> one dataset per question, ready for separate fine-tunes:

<out>/frames/<A>_vs_<B>            symlink to the collection's <A>_vs_<B>/images (no copies)
<out>/<question>/frames            symlink to ../frames; a row's images are frames/<A>_vs_<B>/g<game>_k<k>.png
<out>/<question>/{train,test}.jsonl
<out>/build.json                   the counts per cell and per question, the shortfalls, the sources

Questions: movement (movement + direction in one answer, MOVEMENT_ANSWERS), facing, air, distance
(sf2.data.pairs_labels). A row is about ONE fighter: slot 1 (player 1, controller "directed") or slot 2 (player 2,
"cpu"); it records game, pair [A, B], slot, controller, char, opp and all five labels.

Selection (the one set every question file is cut from): pairs of committed games only; split by game (game % 3 == 2
test, else train); per cell (char, controller, movement, direction, facing) at most CAPS[split] rows, taken round-robin
over the (pair, game) groups in a seeded order so they spread over games and opponents. Shortfalls are reported, never
padded. A question file leaves out the rows whose answer to it is unknown (distance with an impossible x).
"""
import collections
import json
import os
import random
from typing import Dict, Iterable, List, Sequence, Tuple

from . import pairs_collect_io as IO
from . import pairs_labels as L
from .movement_collect import LAG

CAP = 60
CAPS = {"train": 40, "test": 20}
SPLIT_MOD, TEST_REST = 3, 2
MOVEMENT_ANSWERS = ("stand", "walk toward", "walk away", "crouch", "jump toward", "jump away", "jump up", "attack",
                    "special", "block", "hit", "down")
QUESTION_ANSWERS: Dict[str, Tuple[str, ...]] = {"movement": MOVEMENT_ANSWERS, "facing": L.FACINGS, "air": L.AIRS,
                                                "distance": L.DISTANCES}
FILES = ("train", "test")
LABEL_KEYS = ("movement", "direction", "facing", "air", "distance")


def split_of_game(game: int) -> str:
    return "test" if game % SPLIT_MOD == TEST_REST else "train"


def movement_answer(mv: str, direction: str) -> str:
    if mv == "walk":
        return "walk " + direction
    if mv == "jump":
        return "jump up" if direction == "none" else "jump " + direction
    return mv


def answer(question: str, p: Dict) -> str:
    if question == "movement":
        return movement_answer(p["movement"], p["direction"])
    return p[question]


def cell(p: Dict) -> Tuple[str, str, str, str, str]:
    return (p["char"], p["controller"], p["movement"], p["direction"], p["facing"])


CONTROLLERS = ("directed", "cpu")


def universe(chars: Iterable[str], caps: Dict[str, int] = CAPS,
             controllers: Sequence[str] = CONTROLLERS) -> List[Tuple]:
    """Every (split, char, controller, movement, direction, facing) cell a full collection could fill."""
    combos = [(m, "none") for m in L.MOVEMENTS if m not in ("walk", "jump")]
    combos += [("walk", d) for d in ("toward", "away")] + [("jump", d) for d in L.DIRECTIONS]
    return [(s, c, ctl, m, d, f) for s in caps for c in sorted(chars) for ctl in controllers
            for m, d in combos for f in L.FACINGS]


def select(pairs: Sequence[Dict], caps: Dict[str, int] = CAPS, seed: int = 0) -> List[Dict]:
    """At most caps[split] per (split, cell), round-robin over (pair, game) groups in a seeded order."""
    groups: Dict[Tuple, Dict[Tuple, List[Dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in pairs:
        groups[(split_of_game(p["game"]),) + cell(p)][(p["pair_name"], p["game"])].append(p)
    out = []
    for key in sorted(groups):
        rng = random.Random("%d:%s" % (seed, "|".join(key)))
        g = groups[key]
        order = sorted(g)
        rng.shuffle(order)
        queues = []
        for gk in order:
            q = sorted(g[gk], key=lambda p: (p["slot"], p["t"]))
            rng.shuffle(q)
            queues.append(q)
        cap, taken = caps[key[0]], []
        while len(taken) < cap and any(queues):
            for q in queues:
                if q and len(taken) < cap:
                    taken.append(q.pop(0))
        out += taken
    return out


def row_of(question: str, p: Dict) -> Dict:
    a = answer(question, p)
    rid = "pairs_%s_g%04d_s%d_t%05d" % (p["pair_name"], p["game"], p["slot"], p["t"])
    return dict({"id": "%s-%s" % (rid, question), "decision": rid, "question_key": question,
                 "answer": a, "label": QUESTION_ANSWERS[question].index(a), "split": split_of_game(p["game"]),
                 "pair": [p["pair_name"].split("_vs_")[0], p["pair_name"].split("_vs_")[1]],
                 "pair_name": p["pair_name"], "images": ["frames/%s/%s" % (p["pair_name"], n) for n in p["images"]]},
                **{k: p[k] for k in ("game", "slot", "controller", "char", "opp", "t", "k_prev", "k_now", "stage",
                                     "stage_bin", "pos", "length", "episode", "long", "cut_end") + LABEL_KEYS})


def pair_problems(p: Dict, img_dir: str) -> List[str]:
    where = "%s g%s s%s t%s" % (p.get("pair_name"), p.get("game"), p.get("slot"), p.get("t"))
    out = []
    for k, answers in L.QUESTIONS.items():
        if p.get(k) not in answers and not (k == "distance" and p.get(k) == L.UNKNOWN):
            out.append("%s: %s %r is not an answer" % (where, k, p.get(k)))
    want = ["g%04d_k%05d.png" % (p["game"], k) for k in (p["t"] - 4 + LAG, p["t"] + LAG)]
    if p["images"] != want:
        out.append("%s: images %s are not frames (t - 4, t) at lag %d %s" % (where, p["images"], LAG, want))
    out += ["%s: missing image %s" % (where, n) for n in p["images"] if not os.path.exists(os.path.join(img_dir, n))]
    return out


def collection_pairs(root: str) -> Tuple[List[Dict], Dict[str, int], List[str]]:
    """(committed pairs of every pair dir with pair_name set, dropped-uncommitted per dir, the pair dirs)."""
    names = sorted(n for n in os.listdir(root) if "_vs_" in n and os.path.isdir(os.path.join(root, n)))
    pairs, dropped = [], {}
    for n in names:
        kept, drop = IO.committed_pairs(os.path.join(root, n))
        dropped[n] = drop
        pairs += [dict(p, pair_name=n) for p in kept]
    return pairs, dropped, names


def build(root: str, out: str, caps: Dict[str, int] = CAPS, seed: int = 0,
          controllers: Sequence[str] = CONTROLLERS) -> Dict:
    """controllers: whose rows to keep (owner 2026-10-01: label only the player we control -> ("directed",))."""
    bad = set(controllers) - set(CONTROLLERS)
    if bad or not controllers:
        raise ValueError("controllers must be a non-empty subset of %s, got %s" % (CONTROLLERS, controllers))
    pairs, dropped, names = collection_pairs(root)
    pairs = [p for p in pairs if p["controller"] in controllers]
    problems = [x for p in pairs for x in pair_problems(p, os.path.join(root, p["pair_name"], "images"))]
    if problems:
        raise ValueError("%d bad pairs, e.g. %s" % (len(problems), problems[:5]))
    chosen = select(pairs, caps, seed)
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    for n in names:
        link = os.path.join(out, "frames", n)
        if not os.path.islink(link):
            os.symlink(os.path.abspath(os.path.join(root, n, "images")), link)
    per_q = {}
    for q in QUESTION_ANSWERS:
        qd = os.path.join(out, q)
        os.makedirs(qd, exist_ok=True)
        if not os.path.islink(os.path.join(qd, "frames")):
            os.symlink(os.path.join("..", "frames"), os.path.join(qd, "frames"))
        rows = [row_of(q, p) for p in chosen if answer(q, p) in QUESTION_ANSWERS[q]]
        per_q[q] = {}
        for f in FILES:
            mine = [r for r in rows if r["split"] == f]
            _write_jsonl(os.path.join(qd, f + ".jsonl"), mine)
            per_q[q][f] = dict(collections.Counter(r["answer"] for r in mine))
    counts = collections.Counter((split_of_game(p["game"]),) + cell(p) for p in chosen)
    chars = {c for n in names for c in n.split("_vs_")}
    short = {"|".join(k): caps[k[0]] - counts[k] for k in universe(chars, caps, controllers) if counts[k] < caps[k[0]]}
    meta = {"root": os.path.abspath(root), "pairs": names, "caps": caps, "seed": seed, "lag": LAG,
            "controllers": list(controllers),
            "split": "game %% %d == %d test" % (SPLIT_MOD, TEST_REST), "collected": len(pairs),
            "selected": len(chosen), "dropped_uncommitted": dropped,
            "cells": {"|".join(k): n for k, n in sorted(counts.items())}, "short": short, "questions": per_q}
    with open(os.path.join(out, "build.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return meta


def _write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.writelines(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    os.replace(tmp, path)


def fill_report(pairs: Sequence[Dict], chars: Iterable[str], cap: int = CAP) -> Dict:
    """How full the buckets (char, controller, movement, direction, facing) are, over every collected pair (both
    splits): per bucket min(n, cap) / cap, summed per character, per (character, controller) and overall, with the
    empty buckets named."""
    n = collections.Counter(cell(p) for p in pairs)
    buckets = sorted({k[1:] for k in universe(chars, {"all": cap})})
    filled = {b: min(n[b], cap) for b in buckets}

    def pct(keys):
        keys = list(keys)
        return round(100.0 * sum(filled[b] for b in keys) / (cap * len(keys)), 1) if keys else None
    out = {"cap": cap, "buckets": len(buckets), "overall_pct": pct(buckets), "per_char": {}, "zero": {}}
    for c in sorted({b[0] for b in buckets}):
        mine = [b for b in buckets if b[0] == c]
        out["per_char"][c] = {"all": pct(mine), **{ctl: pct(b for b in mine if b[1] == ctl)
                                                   for ctl in ("directed", "cpu")}}
        out["zero"][c] = ["%s %s %s %s" % b[1:] for b in mine if not n[b]]
    out["per_controller"] = {ctl: pct(b for b in buckets if b[1] == ctl) for ctl in ("directed", "cpu")}
    out["counts"] = {"|".join(b): n[b] for b in buckets}
    return out
