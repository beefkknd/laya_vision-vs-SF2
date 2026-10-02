"""The movement-pairs datasets (docs/prereg_movement_pairs.md; scripts/build_pairs_data.py): one collection
(scripts/collect_pairs.py, <root>/<A>_vs_<B>/) -> one dataset per question, ready for separate fine-tunes:

<out>/frames/<A>_vs_<B>            symlink to the collection's <A>_vs_<B>/images (no copies)
<out>/<question>/frames            symlink to ../frames; a row's images are frames/<A>_vs_<B>/g<game>_k<k>.png
<out>/<question>/{train,test}.jsonl
<out>/build.json                   the counts per cell and per question, the shortfalls, the sources

Questions: movement (the owner's 10 grid movements, sf2.data.pairs_labels.movement10: walk split toward / away, jump
not), facing, air, distance (sf2.data.pairs_labels). A row is about ONE fighter: slot 1 (player 1) or slot 2 (player
2); controller "directed" / "cpu" (P1 vs CPU) or "p1" / "p2" (Plan B, both ours); it records game, pair [A, B], slot,
controller, char, opp and all five labels.

Selection (the one set every question file is cut from): pairs of committed games only; split by whole game (Plan B:
crc32 of "<A>_vs_<B>:<game>" % 3 == 2 test, else train - by match, so round 1's games (all game 0) already give both
splits); per grid cell (char, movement10, facing) at most CAPS[split] rows, taken round-robin over the (pair, game)
groups in a seeded order so they spread over games and opponents. Shortfalls are reported, never
padded. A question file leaves out the rows whose answer to it is unknown (distance with an impossible x).
"""
import collections
import json
import os
import random
import zlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import movement_collect_io as MIO
from . import pairs_collect_io as IO
from . import pairs_labels as L
from . import pairs_moves as PM
from .pairs_collect import PER_GAME
from .movement_collect import LAG

CAP = 60
CAPS = {"train": 40, "test": 20}
SPLIT_MOD, TEST_REST = 3, 2
MOVEMENT_ANSWERS = L.MOVEMENTS10
QUESTION_ANSWERS: Dict[str, Tuple[str, ...]] = {"movement": MOVEMENT_ANSWERS, "facing": L.FACINGS, "air": L.AIRS,
                                                "distance": L.DISTANCES}
FILES = ("train", "test")
LABEL_KEYS = ("movement", "direction", "facing", "air", "distance")


def split_of_game(pair_name: str, game: int) -> str:
    """By whole game (match): the same game of the same ordered pair is always on one side."""
    h = zlib.crc32(("%s:%d" % (pair_name, game)).encode())
    return "test" if h % SPLIT_MOD == TEST_REST else "train"


def movement_answer(mv: str, direction: str) -> str:
    return L.movement10(mv, direction)


def answer(question: str, p: Dict) -> str:
    if question == "movement":
        return movement_answer(p["movement"], p["direction"])
    return p[question]


def cell(p: Dict) -> Tuple[str, str, str]:
    """The owner's grid cell: (character, movement10, facing); both players we control pool into it."""
    return (p["char"], L.movement10(p["movement"], p["direction"]), p["facing"])


def split_of(p: Dict) -> str:
    return split_of_game(p["pair_name"], p["game"])


CONTROLLERS = ("directed", "cpu", "p1", "p2")


def universe(chars: Iterable[str], caps: Dict[str, int] = CAPS) -> List[Tuple]:
    """Every (split, char, movement10, facing) cell a full collection could fill: 8 x 20 per split."""
    return [(s, c, m, f) for s in caps for c in sorted(chars) for m in L.MOVEMENTS10 for f in L.FACINGS]


MovementCaps = Optional[Dict[str, Dict[str, int]]]


def cap_of(caps: Dict[str, int], split: str, movement10: str, movement_caps: MovementCaps = None) -> int:
    """The cap of a (split, char, movement10, facing) cell: the movement's own cap when given (round 4: attack and
    special 100 train / 30 test), else the split's."""
    own = (movement_caps or {}).get(movement10)
    return own[split] if own else caps[split]


def cell_caps(chars: Iterable[str], caps: Dict[str, int] = CAPS, movement_caps: MovementCaps = None) -> Dict[Tuple, int]:
    return {k: cap_of(caps, k[0], k[2], movement_caps) for k in universe(chars, caps)}


def parse_movement_caps(items: Iterable[str]) -> Dict[str, Dict[str, int]]:
    """["attack:100:30", ...] -> {"attack": {"train": 100, "test": 30}}; a grid movement, two counts >= 0, once."""
    out: Dict[str, Dict[str, int]] = {}
    for item in items:
        parts = item.split(":")
        if len(parts) != 3 or parts[0] not in L.MOVEMENTS10 or parts[0] in out:
            raise ValueError("bad movement cap %r (movement:train:test, a grid movement of %s, once)"
                             % (item, L.MOVEMENTS10))
        tr, te = int(parts[1]), int(parts[2])
        if tr < 0 or te < 0:
            raise ValueError("bad movement cap %r (counts >= 0)" % item)
        out[parts[0]] = {"train": tr, "test": te}
    return out


def pair_key(p: Dict) -> Tuple[str, int, int, int]:
    return (p["pair_name"], p["game"], p["slot"], p["t"])


def keep_keys(build_dir: str) -> set:
    """The pairs (pair, game, slot, t) of an earlier build's movement files (both splits)."""
    return {pair_key(r) for f in FILES for r in MIO.read_jsonl(os.path.join(build_dir, "movement", f + ".jsonl"))}


def select(pairs: Sequence[Dict], caps: Dict[str, int] = CAPS, seed: int = 0, movement_caps: MovementCaps = None,
           keep: Optional[set] = None) -> List[Dict]:
    """At most the cell's cap (cap_of) per (split, cell), round-robin over (pair, game) groups in a seeded order. The
    ``keep`` pairs (pair_key; round 4: an earlier build's rows) are taken first in each cell; every one must be in
    ``pairs`` and fit under its cell's cap (else ValueError)."""
    keep = set(keep or ())
    missing = keep - {pair_key(p) for p in pairs}
    if missing:
        raise ValueError("%d kept pairs are not in the pool, e.g. %s" % (len(missing), sorted(missing)[:3]))
    groups: Dict[Tuple, Dict[Tuple, List[Dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in pairs:
        groups[(split_of(p),) + cell(p)][(p["pair_name"], p["game"])].append(p)
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
        cap = cap_of(caps, key[0], key[2], movement_caps)
        taken = sorted((p for q in queues for p in q if pair_key(p) in keep), key=pair_key)
        if len(taken) > cap:
            raise ValueError("cell %s: %d kept pairs > its cap %d" % ("|".join(key), len(taken), cap))
        queues = [[p for p in q if pair_key(p) not in keep] for q in queues]
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
                 "answer": a, "label": QUESTION_ANSWERS[question].index(a), "split": split_of(p),
                 "pair": [p["pair_name"].split("_vs_")[0], p["pair_name"].split("_vs_")[1]],
                 "pair_name": p["pair_name"], "images": ["frames/%s/%s" % (p["pair_name"], n) for n in p["images"]]},
                **{k: p[k] for k in ("game", "slot", "controller", "char", "opp", "t", "k_prev", "k_now", "stage",
                                     "stage_bin", "pos", "length", "episode", "long", "cut_end", "pressed",
                                     "pressed_class", "mv_source") + LABEL_KEYS})


def pair_problems(p: Dict, img_dir: str) -> List[str]:
    where = "%s g%s s%s t%s" % (p.get("pair_name"), p.get("game"), p.get("slot"), p.get("t"))
    out = []
    if L.movement10(p.get("movement", ""), p.get("direction", "")) not in L.MOVEMENTS10:
        out.append("%s: movement %r / direction %r is no grid cell" % (where, p.get("movement"), p.get("direction")))
    for k, answers in L.QUESTIONS.items():
        if p.get(k) not in answers and not (k == "distance" and p.get(k) == L.UNKNOWN) and not (
                k == "direction" and p.get(k) == L.UNKNOWN and p.get("movement") == "jump"):     # a jump over him
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


def pressed_classes(char: str, moves: Sequence, n: int, slot: int) -> List[Optional[str]]:
    """The pressed class (attack / special / None) of ``slot`` for each of ``n`` RAM rows, from the move log."""
    return [PM.pressed_class(char, w) for w in PM.pressed_words(moves, n)[slot]]


def relabel(root: str, pairs: Sequence[Dict], bands: Dict[str, int]) -> List[Dict]:
    """Every pair's labels re-derived from its game's stored RAM and move log (owner after round 1: attack vs special
    from the move we pressed, sf2.data.pairs_labels.movement_pressed), with the pressed word, its class and the
    movement's source recorded. Same frames; collections written before the fix (round 1) get the fixed labels."""
    by_game: Dict[Tuple[str, int], List[Dict]] = collections.defaultdict(list)
    for p in pairs:
        by_game[(p["pair_name"], p["game"])].append(p)
    logs = {n: {g["game"]: g.get("moves", []) for g in IO.committed(os.path.join(root, n))}
            for n in {n for n, _ in by_game}}
    out = []
    for (n, game), mine in sorted(by_game.items()):
        rows = MIO.read_ram(os.path.join(root, n, "ram", "g%04d.json.gz" % game))
        words = PM.pressed_words(logs[n][game], len(rows))
        classes = {}
        for p in mine:
            s = p["slot"]
            if s not in classes:
                classes[s] = [PM.pressed_class(p["char"], w) for w in words[s]]
            w = words[s][p["t"]]
            cls = classes[s][p["t"]]
            lab = L.labels(rows, p["t"], s, bands, cls)
            out.append(dict(p, **lab, pressed=w, pressed_class=cls,
                            mv_source=L.movement_pressed(rows, p["t"], s, cls)[1],
                            in_episode=L.same_episode(rows, p["t"], s, classes[s])))
    return out


def in_episode_only(pairs: Sequence[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    """The relabelled pairs whose two frames lie inside one movement episode, and the dropped ones per grid
    movement (owner fix after the label quality check)."""
    kept = [p for p in pairs if p["in_episode"]]
    dropped = collections.Counter(L.movement10(p["movement"], p["direction"]) for p in pairs if not p["in_episode"])
    return kept, dict(sorted(dropped.items()))


def cap_per_game(pairs: Sequence[Dict], per_game: int = PER_GAME, seed: int = 0) -> List[Dict]:
    """At most ``per_game`` pairs per (pair, game, slot, grid cell) after a relabel (a cell may have gained pairs from
    another), a seeded choice."""
    groups: Dict[Tuple, List[Dict]] = collections.defaultdict(list)
    for p in pairs:
        groups[(p["pair_name"], p["game"], p["slot"]) + cell(p)].append(p)
    out = []
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda p: p["t"])
        if len(g) > per_game:
            g = sorted(random.Random("%d:%s" % (seed, "|".join(map(str, key)))).sample(g, per_game),
                       key=lambda p: p["t"])
        out += g
    return out


def mv_sources(pairs: Sequence[Dict]) -> Dict[str, Dict[str, int]]:
    """Per character: how each pair's movement was decided ("<movement> <source>": pressed / fallback / ram)."""
    out: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for p in pairs:
        out[p["char"]]["%s %s" % (p["movement"], p.get("mv_source", "ram"))] += 1
        out[p["char"]][p.get("mv_source", "ram")] += 1
    return {c: dict(v) for c, v in sorted(out.items())}


def eligible(root: str, seed: int = 0, controllers: Sequence[str] = CONTROLLERS,
             bands: Optional[Dict[str, int]] = None) -> Tuple[List[Dict], Dict[str, int], List[str], Dict[str, int]]:
    """The pool a build selects from: committed pairs of the controllers' rows, checked as collected, relabelled,
    both frames inside one episode, at most PER_GAME per (pair, game, slot, cell). (pool, dropped uncommitted per
    dir, pair dirs, dropped outside an episode per movement)."""
    bad = set(controllers) - set(CONTROLLERS)
    if bad or not controllers:
        raise ValueError("controllers must be a non-empty subset of %s, got %s" % (CONTROLLERS, controllers))
    pairs, dropped, names = collection_pairs(root)
    dropped_ep: Dict[str, int] = {}
    pairs = [p for p in pairs if p["controller"] in controllers]
    problems = [x for p in pairs for x in pair_problems(p, os.path.join(root, p["pair_name"], "images"))]
    if not problems:            # as collected, then as relabelled
        pairs, dropped_ep = in_episode_only(relabel(root, pairs, L.poke_bands() if bands is None else bands))
        pairs = cap_per_game(pairs, PER_GAME, seed)
        problems = [x for p in pairs for x in pair_problems(p, os.path.join(root, p["pair_name"], "images"))]
    if problems:
        raise ValueError("%d bad pairs, e.g. %s" % (len(problems), problems[:5]))
    return pairs, dropped, names, dropped_ep


def build(root: str, out: str, caps: Dict[str, int] = CAPS, seed: int = 0,
          controllers: Sequence[str] = CONTROLLERS, bands: Optional[Dict[str, int]] = None,
          movement_caps: MovementCaps = None, keep_from: Optional[str] = None) -> Dict:
    """controllers: whose rows to keep (owner 2026-10-01: label only the player we control -> ("directed",); Plan B
    controls both: ("p1", "p2"), the default keeps every row). movement_caps: per grid movement {split: cap} over
    ``caps`` (round 4: attack / special). keep_from: an earlier build whose every movement row is selected again
    (round 4: round 3's rows, so its test rows stay in the test)."""
    pairs, dropped, names, dropped_ep = eligible(root, seed, controllers, bands)
    keep = keep_keys(keep_from) if keep_from else set()
    chosen = select(pairs, caps, seed, movement_caps, keep)
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
    counts = collections.Counter((split_of(p),) + cell(p) for p in chosen)
    chars = {c for n in names for c in n.split("_vs_")}
    want = cell_caps(chars, caps, movement_caps)
    short = {"|".join(k): n - counts[k] for k, n in want.items() if counts[k] < n}
    meta = {"root": os.path.abspath(root), "pairs": names, "caps": caps, "seed": seed, "lag": LAG,
            "movement_caps": dict(movement_caps or {}), "keep_from": os.path.abspath(keep_from) if keep_from else None,
            "kept": len(keep),
            "controllers": list(controllers),
            "split": "crc32(pair:game) %% %d == %d test" % (SPLIT_MOD, TEST_REST), "collected": len(pairs),
            "selected": len(chosen), "dropped_uncommitted": dropped, "mv_sources": mv_sources(pairs),
            "dropped_episode": dropped_ep,
            "mv_sources_selected": mv_sources(chosen),
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
    """How full the grid is over every collected pair (both splits, both controllers): per cell (char, movement10,
    facing) min(n, cap) / cap, per character and overall, with the empty cells named and the raw counts."""
    n = collections.Counter(cell(p) for p in pairs)
    cells = sorted({k[1:] for k in universe(chars, {"all": cap})})
    filled = {c: min(n[c], cap) for c in cells}

    def pct(keys):
        keys = list(keys)
        return round(100.0 * sum(filled[c] for c in keys) / (cap * len(keys)), 1) if keys else None
    out = {"cap": cap, "cells": len(cells), "overall_pct": pct(cells), "per_char": {}, "zero": {}, "pct": {}}
    for ch in sorted({c[0] for c in cells}):
        mine = [c for c in cells if c[0] == ch]
        out["per_char"][ch] = pct(mine)
        out["zero"][ch] = ["%s %s" % c[1:] for c in mine if not n[c]]
        out["pct"][ch] = {"%s|%s" % c[1:]: pct([c]) for c in mine}
    out["counts"] = {"|".join(c): n[c] for c in cells}
    return out
