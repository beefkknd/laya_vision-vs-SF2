"""Round 3 of docs/prereg_movement_finetunes.md, the "fireball" dataset (scripts/build_mv3_data.py fireball):

  "Is there a fireball on the screen?" -> none / left (thrown by the fighter on the left) / right

Fireball rows: the projectile-trigger samples of the collection (<root>/<pair>/shots.jsonl of committed games,
sf2.data.pairs_shots), re-checked from the stored RAM at the displayed row t (lag 1): the thrower's slot on and drawn
(blink bit clear), the other slot off (else dropped: "two_shots"), the thrower pressing its projectile word at the
spawn row (else dropped: "not_projectile", e.g. Dhalsim's yoga flame), the thrower's x different from the other's
(else dropped: "equal_x"). The answer = the thrower's side by x at t. At most CAPS[split] per (split, thrower, side,
flight stage), split = by whole match (test: crc32("<pair>:<game>") % 3 == 2), round-robin over matches in a seeded
order; val = 1 in 6 training matches (sf2.data.pairs_train.split3), cut from the capped train rows.
"none" rows: rows of the existing gated pairs build (one per image pair) with both slots off over rows t - 8 .. t (no
projectile and no impact spark in either frame), sampled (seeded) per split to the count of the larger fireball
answer of that split. One dir per answer (none/, left/, right/): scripts/train.py --balance draws them equally.
"""
import collections
import json
import os
import random
import zlib
from typing import Dict, List, Optional, Sequence, Tuple

from . import movement_collect_io as MIO
from . import pairs_data as D
from . import pairs_shots as S
from . import pairs_train as T
from .movement_collect import STAGES

FIRE_ANSWERS = ("none", "left", "right")
FIRE_CRITERIA = {"none": "no fireball", "left": "a fireball thrown by the fighter on the left",
                 "right": "a fireball thrown by the fighter on the right"}
FIRE_TEXT = "Is there a fireball on the screen?"
CAPS = {"train": 40, "test": 20}
NONE_WINDOW = 8
OUT = "test_data_mv3_fireball"


def question_fireball() -> Dict:
    return {"type": "choice", "instructions": FIRE_TEXT, "criteria": dict(FIRE_CRITERIA)}


def _ram_path(root: str, pair: str, game: int) -> str:
    return os.path.join(root, pair, "ram", "g%04d.json.gz" % game)


class _Rams:
    def __init__(self, root: str):
        self.root, self.cache = root, {}

    def __call__(self, pair: str, game: int) -> List[Dict[str, int]]:
        if (pair, game) not in self.cache:
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[(pair, game)] = MIO.read_ram(_ram_path(self.root, pair, game))
        return self.cache[(pair, game)]


def shot_samples(root: str) -> List[Dict]:
    out = []
    for n in sorted(x for x in os.listdir(root) if "_vs_" in x and os.path.isdir(os.path.join(root, x))):
        kept, _ = S.committed_shots(os.path.join(root, n))
        out += [dict(r, pair_name=n) for r in kept]
    return out


def checked(root: str, samples: Sequence[Dict]) -> Tuple[List[Dict], Dict[str, int]]:
    """The samples that may carry a fireball answer (re-checked from RAM), and the dropped ones per reason."""
    rams, kept, dropped = _Rams(root), [], collections.Counter()
    for r in samples:
        ram = rams(r["pair_name"], r["game"])
        s, t = r["slot"], r["t"]
        if not S.is_projectile(r["thrower"], r["spawn_word"]):
            dropped["not_projectile"] += 1
        elif not S.visible(ram[t], s):
            dropped["not_drawn"] += 1
        elif ram[t]["shot%d" % (3 - s)]:
            dropped["two_shots"] += 1
        elif S.side_of(ram[t], s) is None:
            dropped["equal_x"] += 1
        else:
            kept.append(dict(r, side=S.side_of(ram[t], s)))
    return kept, {k: dropped.get(k, 0) for k in ("not_projectile", "not_drawn", "two_shots", "equal_x")}


def _cell(r: Dict) -> Tuple[str, str, str, str]:
    return (D.split_of_game(r["pair_name"], r["game"]), r["thrower"], r["side"], r["flight_stage"])


def select(samples: Sequence[Dict], caps: Dict[str, int] = CAPS, seed: int = 0) -> List[Dict]:
    groups: Dict[Tuple, Dict[Tuple, List[Dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in samples:
        groups[_cell(r)][(r["pair_name"], r["game"])].append(r)
    out = []
    for key in sorted(groups):
        rng = random.Random("%d:%s" % (seed, "|".join(key)))
        order = sorted(groups[key])
        rng.shuffle(order)
        queues = [sorted(groups[key][m], key=lambda r: (r["slot"], r["t"])) for m in order]
        taken = []
        while len(taken) < caps[key[0]] and any(queues):
            for q in queues:
                if q and len(taken) < caps[key[0]]:
                    taken.append(q.pop(0))
        out += taken
    return out


def universe(caps: Dict[str, int] = CAPS) -> List[Tuple[str, str, str, str]]:
    return [(s, c, side, st) for s in caps for c in S.THROWERS for side in T.SIDES for st in STAGES]


def fill_report(root: str, caps: Dict[str, int] = CAPS) -> Dict:
    kept, dropped = checked(root, shot_samples(root))
    n = collections.Counter(_cell(r) for r in kept)
    cells = {"|".join(k): n[k] for k in universe(caps)}
    filled = sum(min(n[k], caps[k[0]]) for k in universe(caps))
    full = sum(caps[k[0]] for k in universe(caps))
    return {"cells": cells, "filled_pct": round(100.0 * filled / full, 1), "dropped": dropped,
            "not_full": {k: "%d/%d" % (v, caps[k.split("|")[0]]) for k, v in cells.items()
                         if v < caps[k.split("|")[0]]}}


def _record(r: Dict, answer: str, source: str, **extra) -> Dict:
    base = "%s_%s_g%04d_s%d_t%05d" % (source, r["pair_name"], r["game"], r["slot"], r["t"])
    return dict({"id": base + "-fireball", "decision": base, "images": list(r["images"]),
                 "question": question_fireball(), "label": FIRE_ANSWERS.index(answer), "answer": answer,
                 "question_key": "fireball", "split": T.split3(r["pair_name"], r["game"]), "source": source,
                 "pair_name": r["pair_name"], "pair": r["pair_name"].split("_vs_"), "game": r["game"], "t": r["t"],
                 "k_prev": r["k_prev"], "k_now": r["k_now"], "facing": "-"}, **extra)


def fire_record(r: Dict) -> Dict:
    return _record(dict(r, images=["frames/%s/%s" % (r["pair_name"], n) for n in r["images"]]), r["side"], "shot",
                   char=r["thrower"], side=r["side"], thrower_slot=r["slot"], slot=r["slot"],
                   flight_stage=r["flight_stage"], flight=r["flight"], spawn_word=r["spawn_word"],
                   shot_x=r["shot_x"])


def none_rows(src: str, root: str) -> List[Dict]:
    """Pairs-build rows (one per image pair) with both slots off over t - NONE_WINDOW .. t."""
    rams, seen, out = _Rams(root), set(), []
    for r in sorted(T.source_rows(src, "movement"), key=lambda x: x["id"]):
        if tuple(r["images"]) in seen:
            continue
        seen.add(tuple(r["images"]))
        ram = rams(r["pair_name"], r["game"])
        if all(ram[u]["shot1"] == 0 == ram[u]["shot2"] for u in range(r["t"] - NONE_WINDOW, r["t"] + 1)):
            out.append(_record(r, "none", "pairs", char="none", side="none", thrower_slot=0, slot=r["slot"],
                               flight_stage="none", source_id=r["id"]))
    return out


def build_fireball(src: str, root: str, out: str, caps: Dict[str, int] = CAPS, seed: int = 0) -> Dict:
    if os.path.exists(out):
        raise FileExistsError("%s exists: pick a new --out (never overwritten)" % out)
    src_root = json.load(open(os.path.join(src, "build.json")))["root"]
    if os.path.realpath(src_root) != os.path.realpath(root):
        raise ValueError("the shots collection %s is not the pairs build's root %s (frames would not resolve)"
                         % (root, src_root))
    samples = shot_samples(root)
    kept, dropped = checked(root, samples)
    fire = [fire_record(r) for r in select(kept, caps, seed)]
    pool = none_rows(src, root)
    want = {f: max(sum(r["split"] == f and r["answer"] == a for r in fire) for a in ("left", "right"))
            for f in T.FILES}
    none = []
    for f in T.FILES:
        mine = [r for r in pool if r["split"] == f]
        if len(mine) < want[f]:
            raise ValueError("%s: %d none rows for %d wanted" % (f, len(mine), want[f]))
        none += random.Random("%d:none:%s" % (seed, f)).sample(mine, want[f])
    rows = fire + none
    frames = os.path.abspath(os.path.join(src, "frames"))
    os.makedirs(out)
    counts = {}
    for a in FIRE_ANSWERS:
        base = os.path.join(out, a)
        os.makedirs(base)
        os.symlink(frames, os.path.join(base, "frames"))
        counts[a] = {}
        for f in T.FILES:
            mine = sorted((r for r in rows if r["answer"] == a and r["split"] == f), key=lambda x: x["id"])
            D._write_jsonl(os.path.join(base, f + ".jsonl"), mine)
            counts[a][f] = len(mine)
    n = collections.Counter(_cell(r) for r in kept)
    taken = collections.Counter((D.split_of_game(r["pair_name"], r["game"]), r["char"], r["side"], r["flight_stage"])
                                for r in fire)
    meta = {"source": os.path.abspath(src), "root": os.path.abspath(root), "dataset": "fireball",
            "question": "fireball", "answers": list(FIRE_ANSWERS), "answer_dirs": True, "dirs": list(FIRE_ANSWERS),
            "caps": caps, "seed": seed, "none_window": NONE_WINDOW, "counts": counts,
            "samples": len(samples), "eligible": len(kept), "dropped": dropped, "none_pool": len(pool),
            "cells": {"|".join(k): taken[k] for k in universe(caps)},
            "short": {"|".join(k): caps[k[0]] - taken[k] for k in universe(caps) if taken[k] < caps[k[0]]},
            "eligible_cells": {"|".join(k): n[k] for k in universe(caps)},
            "matches": {f: len({(r["pair_name"], r["game"]) for r in rows if r["split"] == f}) for f in T.FILES}}
    bad = problems_fireball(out, src, root, caps)
    meta["problems"] = bad
    with open(os.path.join(out, "build.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    if bad:
        raise ValueError("%d problems in %s, e.g. %s" % (len(bad), out, bad[:5]))
    return meta


# ---- the independent check (from the prereg's rules; not the builder's functions) ---------------------------------

_THROWN = {"ryu": "hadoken", "ken": "hadoken", "guile": "sonic_boom", "dhalsim": "yoga_fire"}


def _pressed(moves: List[list], t: int, p: int) -> Optional[str]:
    for m in moves:
        if (m[4] if len(m) >= 5 else 1) == p and m[1] < t <= m[2]:
            return m[0]
    return None


def _fire_problems(r: Dict, ram: List[Dict[str, int]], moves: List[list]) -> List[str]:
    p, t, bad = r["thrower_slot"], r["t"], []
    on = lambda u, q: ram[u]["shot%d" % q] != 0
    if p not in (1, 2) or not on(t, p) or ram[t]["shot%d_hide" % p] % 2 == 1 or on(t, 3 - p):
        return ["%s: no single drawn projectile of slot %s at t" % (r["id"], p)]
    me, him = ram[t]["p%d_x" % p], ram[t]["p%d_x" % (3 - p)]
    side = "left" if me < him else "right" if me > him else None
    char = r["pair_name"].split("_vs_")[p - 1]
    if r["answer"] != side or r.get("side") != side or r.get("char") != char:
        bad.append("%s: answer %s / side %s / char %s, RAM says %s thrown by %s" % (
            r["id"], r["answer"], r.get("side"), r.get("char"), side, char))
    a = t
    while a > 0 and on(a - 1, p):
        a -= 1
    e = t
    while e + 1 < len(ram) and on(e + 1, p):
        e += 1
    if _pressed(moves, a, p) != _THROWN.get(char):
        bad.append("%s: %s pressed %s at the spawn row %d" % (r["id"], char, _pressed(moves, a, p), a))
    if r.get("flight_stage") != ("start", "middle", "end")[min(2, 3 * (t - a) // (e - a + 1))]:
        bad.append("%s: stage %s of flight [%d, %d]" % (r["id"], r.get("flight_stage"), a, e))
    return bad


def problems_fireball(out: str, src: str, root: str, caps: Dict[str, int] = CAPS) -> List[str]:
    """Every row re-derived from the stored RAM and move log: the dir = the answer, label = its index, the exact
    question; images = captures t - 3 and t + 1 of its game; the match's split (no match in two splits); fireball
    rows: one drawn projectile at t, of the slot named, the answer = that fighter's side by x, its character's
    projectile word pressed at the spawn row, the flight stage; caps per (split, thrower, side, stage) and per (match,
    slot, stage) SHOT_PER_GAME; none rows: an existing pairs-build image pair with both slots off over t - 8 .. t,
    as many per split as the larger fireball answer; ids unique; only committed games."""
    data = T.read_dataset(out)
    src_imgs = {tuple(r["images"]) for r in T.source_rows(src, "movement")}
    rams, logs = _Rams(root), {}
    bad: List[str] = []
    ids = collections.Counter()
    split_of_match = collections.defaultdict(set)
    per_cell, per_game, per_split = collections.Counter(), collections.Counter(), collections.Counter()
    for d, files in data.items():
        if d not in FIRE_ANSWERS:
            bad.append("dir %s is no answer" % d)
        for f, rows in files.items():
            for r in rows:
                ids[r["id"]] += 1
                m = (r["pair_name"], r["game"])
                if m[0] not in logs:
                    logs[m[0]] = {g["game"]: g.get("moves", []) for g in
                                  MIO.read_jsonl(os.path.join(root, m[0], "games.jsonl"))}
                if m[1] not in logs[m[0]]:
                    bad.append("%s: game %d is not committed" % (r["id"], m[1]))
                    continue
                split_of_match[m].add(f)
                test = zlib.crc32(("%s:%d" % m).encode()) % 3 == 2
                want = "test" if test else "val" if zlib.crc32(("val:%s:%d" % m).encode()) % 6 == 0 else "train"
                if r["split"] != f or f != want:
                    bad.append("%s: in %s, split %s, match belongs to %s" % (r["id"], f, r["split"], want))
                if r["answer"] != d or r["label"] != list(FIRE_ANSWERS).index(d) or \
                        tuple(r["question"]["criteria"]) != FIRE_ANSWERS or \
                        r["question"]["instructions"] != "Is there a fireball on the screen?":
                    bad.append("%s: answer %s / label %s / question in dir %s" % (r["id"], r["answer"], r["label"], d))
                imgs = ["frames/%s/g%04d_k%05d.png" % (m[0], m[1], k) for k in (r["t"] - 3, r["t"] + 1)]
                if r["images"] != imgs or not all(os.path.exists(os.path.join(out, d, x)) for x in imgs):
                    bad.append("%s: images %s, want %s" % (r["id"], r["images"], imgs))
                ram = rams(*m)
                per_split[(f, d)] += 1
                if d == "none":
                    if tuple(r["images"]) not in src_imgs:
                        bad.append("%s: none row not from the pairs build" % r["id"])
                    if any(ram[u]["shot1"] or ram[u]["shot2"] for u in range(r["t"] - 8, r["t"] + 1)):
                        bad.append("%s: a shot slot on within t - 8 .. t" % r["id"])
                    continue
                bad += _fire_problems(r, ram, logs[m[0]][m[1]])
                per_cell[("test" if test else "train", r.get("char"), r["answer"], r.get("flight_stage"))] += 1
                per_game[m + (r["thrower_slot"], r.get("flight_stage"))] += 1
    bad += ["id %s on %d rows" % (i, n) for i, n in ids.items() if n > 1]
    bad += ["match %s:%d in %s" % (m[0], m[1], sorted(s)) for m, s in split_of_match.items() if len(s) > 1]
    bad += ["cell %s: %d > cap %d" % ("|".join(map(str, k)), n, caps[k[0]]) for k, n in per_cell.items()
            if n > caps[k[0]]]
    bad += ["match %s: %d > %d per game" % (k, n, S.SHOT_PER_GAME) for k, n in per_game.items() if n > S.SHOT_PER_GAME]
    for f in T.FILES:
        want_none = max(per_split[(f, "left")], per_split[(f, "right")])
        if per_split[(f, "none")] != want_none:
            bad.append("%s: %d none rows, the larger fireball answer has %d" % (f, per_split[(f, "none")], want_none))
    return bad
