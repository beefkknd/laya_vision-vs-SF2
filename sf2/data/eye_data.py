"""The eye's aligned datasets (docs/eye_questions_v1.md, "Datasets v1"; scripts/build_eye_data.py), cut from ONE frame
pool (sf2.data.eye_pool) with ONE whole-match split table (sf2.data.pairs_train.split3: test = crc32 of the match
% 3 == 2, so round 3's test matches stay test; val = 1 in 6 training matches; train = the rest).

  q1 fireball  "Is there a fireball on the screen?"                           yes / no
  q3 act       "What is the fighter on the <side> doing?"                      moving / attack / special
  q4 air       "Is the fighter on the <side> on the ground or in the air?"     ground / air
  q5 dist      "Are the two fighters close or far?"                            close / far
(q2 was dropped for this round by the owner.)

Alignment (the doc's rules 1-5): every candidate row has a STRATUM - the metadata a shortcut could use - and within
each stratum every answer gets the same number of rows (min over the answers, at most ``cap`` per answer), so the
answer is not predictable from metadata:
  q1  (split, match pair, game range, the pose of player 1, the pose of player 2)   pose: "projectile" (the
      throwing pose, with or without a projectile; sf2.data.eye_pool.pose_of), else the fighter's grid movement
  q3  (split, match pair, game, side asked, the other fighter's grid movement)
  q4  (split, match pair, game, side asked, the other fighter's grid movement)
  q5  (split, match pair, game, the left fighter's grid movement, the right fighter's grid movement)
(The first build matched on the coarse act class - moving / attack / special - of the other fighter / the poses: the
shortcut check failed q1, q3 and q5 on the finer grid movement, so the strata are the grid movements now.)
Inside a stratum the rows of an answer are taken in tiers: round 3's test rows first (``prefer``), then the q1 hard
negatives (blink / before spawn / after impact / throwing pose), then the rest, each tier in a seeded order. q4's
take-off and landing pairs come in at their natural share (preferring them made a third of the air rows take-offs). q3 keeps only pairs whose two frames lie in one movement episode of the fighter asked
(pairs_labels.same_episode); q3 / q4 / q5 drop pairs with the two x equal at t (no side). One data dir per answer
(train.py --balance sampling draws them equally, no copies). Each row keeps its shortcut features (``shortcut``) for
sf2.data.eye_shortcut.
"""
import collections
import json
import os
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import pairs_data as D
from . import pairs_train as T
from .eye_pool import ACT_ANSWERS

FILES = T.FILES
GAME_RANGES = (0, 10, 16, 24, 32)          # q1's game ranges: 0-9, 10-15, 16-23, 24-31, 32+
HARD_FIRST = ("blink", "before_spawn", "after_impact", "pose")
Q: Dict[str, Dict] = {
    "q1": {"name": "fireball", "answers": ("yes", "no"), "side": False, "cap": None,
           "text": "Is there a fireball on the screen?",
           "criteria": {"yes": "a fireball is on the screen", "no": "no fireball on the screen"}},
    "q3": {"name": "act", "answers": ACT_ANSWERS, "side": True, "cap": 2,
           "text": "What is the fighter on the %s doing?",
           "criteria": {"moving": "moving (standing, walking, crouching, jumping, blocking, being hit or knocked down)",
                        "attack": "attacking (a normal, a jump attack or a throw)",
                        "special": "doing a special move"}},
    "q4": {"name": "air", "answers": ("ground", "air"), "side": True, "cap": 1,
           "text": "Is the fighter on the %s on the ground or in the air?",
           "criteria": {"ground": "on the ground", "air": "in the air"}},
    "q5": {"name": "dist", "answers": ("close", "far"), "side": False, "cap": 2,
           "text": "Are the two fighters close or far?",
           "criteria": {"close": "close (throw or poke range)", "far": "far"}},
}
OUT = "test_data_eye_%s_%s"


def out_of(q: str) -> str:
    return OUT % (q, Q[q]["name"])


def question(q: str, side: Optional[str] = None) -> Dict:
    spec = Q[q]
    if spec["side"] != (side is not None) or (side is not None and side not in T.SIDES):
        raise ValueError("%s: side %r" % (q, side))
    text = spec["text"] % side if spec["side"] else spec["text"]
    return {"type": "choice", "instructions": text, "criteria": dict(spec["criteria"])}


def game_range(game: int) -> str:
    for lo, hi in zip(GAME_RANGES, GAME_RANGES[1:]):
        if lo <= game < hi:
            return "g%d-%d" % (lo, hi - 1)
    return "g%d+" % GAME_RANGES[-1]


def _base(f: Dict, q: str, rid: str, answer: str, stratum: Tuple, shortcut: Dict, tier: int, **extra) -> Dict:
    return dict({"id": rid, "pair_name": f["pair_name"], "pair": list(f["pair"]), "game": f["game"], "t": f["t"],
                 "k_prev": f["k_prev"], "k_now": f["k_now"], "images": list(f["images"]), "answer": answer,
                 "question_key": Q[q]["name"], "split": T.split3(f["pair_name"], f["game"]),
                 "stratum": "|".join(map(str, stratum)), "shortcut": shortcut, "tier": tier}, **extra)


def _rid(q: str, f: Dict, slot: Optional[int] = None) -> str:
    base = "eye_%s_%s_g%04d_t%05d" % (Q[q]["name"], f["pair_name"], f["game"], f["t"])
    return base if slot is None else "%s_s%d" % (base, slot)


def _tier(prefer: set, key: Tuple, hard: bool) -> int:
    return 0 if key in prefer else 1 if hard else 2


def cands_q1(pool: Iterable[Dict], prefer: set) -> Tuple[List[Dict], Dict[str, int]]:
    out, drop = [], collections.Counter()
    for f in pool:
        if f["fire"] is None:
            drop[f["fire_why"]] += 1
            continue
        poses = [f["fighters"][s]["pose"] for s in (1, 2)]
        fine = [p if p == "projectile" else f["fighters"][s]["mv10"] for s, p in zip((1, 2), poses)]
        split = T.split3(f["pair_name"], f["game"])
        stratum = (split, f["pair_name"], game_range(f["game"])) + tuple(fine)
        hard = bool(set(f["hard"]) & set(HARD_FIRST))
        out.append(_base(f, "q1", _rid("q1", f), f["fire"], stratum,
                         {"pair": f["pair_name"], "game": f["game"], "pose1": fine[0], "pose2": fine[1]},
                         _tier(prefer, (f["pair_name"], f["game"], f["t"]), hard),
                         hard=list(f["hard"]), poses=poses,
                         shots={str(s): f["shots"][s] for s in (1, 2)}))
    return out, dict(drop)


def _side_rows(f: Dict, s: int) -> Tuple[Dict, Dict]:
    return f["fighters"][s], f["fighters"][3 - s]


def cands_side(pool: Iterable[Dict], q: str, prefer: set) -> Tuple[List[Dict], Dict[str, int]]:
    """q3 / q4: one candidate per (pair, fighter) with a side."""
    out, drop = [], collections.Counter()
    for f in pool:
        for s in (1, 2):
            me, him = _side_rows(f, s)
            if me["side"] is None:
                drop["equal_x"] += 1
                continue
            if q == "q3" and (me["act"] is None or not me["in_episode"]):
                drop["unknown_movement" if me["act"] is None else "outside_episode"] += 1
                continue
            answer = me["act"] if q == "q3" else me["air"]
            split = T.split3(f["pair_name"], f["game"])
            stratum = (split, f["pair_name"], f["game"], me["side"], him["mv10"])
            out.append(_base(f, q, _rid(q, f, s), answer, stratum,
                             {"pair": f["pair_name"], "side": me["side"], "game": f["game"],
                              "other_mv10": him["mv10"]},
                             _tier(prefer, (f["pair_name"], f["game"], f["t"], s), False),
                             slot=s, side=me["side"], char=me["char"], other=him["char"], mv10=me["mv10"],
                             other_mv10=him["mv10"], air=me["air"], air_prev=me["air_prev"]))
    return out, dict(drop)


def cands_q5(pool: Iterable[Dict], prefer: set) -> Tuple[List[Dict], Dict[str, int]]:
    out, drop = [], collections.Counter()
    for f in pool:
        a, b = f["fighters"][1], f["fighters"][2]
        if a["side"] is None or f["dist"] is None:
            drop["equal_x" if a["side"] is None else "bad_x"] += 1
            continue
        left, right = (a, b) if a["side"] == "left" else (b, a)
        split = T.split3(f["pair_name"], f["game"])
        stratum = (split, f["pair_name"], f["game"], left["mv10"], right["mv10"])
        out.append(_base(f, "q5", _rid("q5", f), f["dist"], stratum,
                         {"pair": f["pair_name"], "game": f["game"], "left_mv10": left["mv10"],
                          "right_mv10": right["mv10"]},
                         _tier(prefer, (f["pair_name"], f["game"], f["t"]), False), gap=f["gap"]))
    return out, dict(drop)


def candidates(pool: Sequence[Dict], q: str, prefer: Optional[set] = None) -> Tuple[List[Dict], Dict[str, int]]:
    prefer = prefer or set()
    if q == "q1":
        return cands_q1(pool, prefer)
    if q == "q5":
        return cands_q5(pool, prefer)
    if q in ("q3", "q4"):
        return cands_side(pool, q, prefer)
    raise ValueError("unknown question %r (of %s)" % (q, sorted(Q)))


def select(cands: Sequence[Dict], answers: Sequence[str], cap: Optional[int], seed: int = 0) -> List[Dict]:
    """Per stratum: n = min over ``answers`` of the candidates (at most ``cap``); n of each answer, by tier, then a
    seeded order. Strata missing an answer give nothing."""
    groups: Dict[str, Dict[str, List[Dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for c in cands:
        groups[c["stratum"]][c["answer"]].append(c)
    out = []
    for key in sorted(groups):
        g = groups[key]
        n = min(len(g[a]) for a in answers)
        if cap is not None:
            n = min(n, cap)
        for a in answers:
            mine = sorted(g[a], key=lambda c: c["id"])
            random.Random("%d:%s:%s" % (seed, key, a)).shuffle(mine)
            out += sorted(mine, key=lambda c: c["tier"])[:n]
    return out


def record(q: str, c: Dict) -> Dict:
    spec = Q[q]
    side = c.get("side") if spec["side"] else None
    return dict(c, question=question(q, side), label=list(spec["answers"]).index(c["answer"]))


# ---- prefer: round 3's test rows ----------------------------------------------------------------------------------

def prefer_keys(q: str, dirs: Sequence[str]) -> set:
    """Round 3's test rows (q3: test_data_mv3_act, q1: test_data_mv3_fireball) as pool keys: q1 (pair, game, t),
    q3 (pair, game, t, slot). Missing dirs give nothing."""
    keys = set()
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for sub in sorted(os.listdir(d)):
            path = os.path.join(d, sub, "test.jsonl")
            if os.path.exists(path):
                for line in open(path):
                    r = json.loads(line)
                    k = (r["pair_name"], r["game"], r["t"])
                    keys.add(k + (r["slot"],) if q in ("q3", "q4") else k)
    return keys


# ---- the build -----------------------------------------------------------------------------------------------------

def _link_frames(out: str, root: str, pairs: Iterable[str]) -> None:
    os.makedirs(os.path.join(out, "frames"))
    for n in sorted(pairs):
        os.symlink(os.path.abspath(os.path.join(root, n, "images")), os.path.join(out, "frames", n))


def _write(out: str, q: str, rows: List[Dict]) -> Dict:
    counts = {}
    for a in Q[q]["answers"]:
        base = os.path.join(out, a)
        os.makedirs(base)
        os.symlink(os.path.join("..", "frames"), os.path.join(base, "frames"))
        counts[a] = {}
        for f in FILES:
            mine = sorted((r for r in rows if r["answer"] == a and r["split"] == f), key=lambda x: x["id"])
            D._write_jsonl(os.path.join(base, f + ".jsonl"), mine)
            counts[a][f] = len(mine)
    return counts


def extras(q: str, rows: List[Dict]) -> Dict:
    """The hard cases asked for by rule 3, counted per split and answer."""
    by = lambda pred: {f: dict(sorted(collections.Counter(r["answer"] for r in rows if r["split"] == f and pred(r))
                                      .items())) for f in FILES}
    if q == "q1":
        no = [r for r in rows if r["answer"] == "no"]
        tags = collections.Counter(t for r in no for t in r["hard"])
        return {"hard_negatives": {f: sum(1 for r in no if r["split"] == f and set(r["hard"]) & set(HARD_FIRST))
                                   for f in FILES},
                "hard_tags": dict(sorted(tags.items())),
                "yes_in_throwing_pose": sum(1 for r in rows if r["answer"] == "yes" and "projectile" in r["poses"])}
    if q == "q3":
        return {"jump_attacks": by(lambda r: r["answer"] == "attack" and r["air"] == "air"),
                "plain_jumps": by(lambda r: r["mv10"] == "jump"),
                "movements": dict(sorted(collections.Counter(r["mv10"] for r in rows).items()))}
    if q == "q4":
        return {"take_off": by(lambda r: r["air_prev"] == "ground" and r["air"] == "air"),
                "landing": by(lambda r: r["air_prev"] == "air" and r["air"] == "ground")}
    return {"gap": {a: [min(r["gap"] for r in rows if r["answer"] == a), max(r["gap"] for r in rows
                                                                           if r["answer"] == a)]
                    for a in Q[q]["answers"] if any(r["answer"] == a for r in rows)}}


def build(pool: Sequence[Dict], q: str, out: str, root: str, seed: int = 0, prefer: Optional[set] = None,
          cap: Optional[int] = -1) -> Dict:
    """Write <out>/<answer>/{train,val,test}.jsonl (+ frames links) and <out>/build.json; refuses an existing out."""
    if os.path.exists(out):
        raise FileExistsError("%s exists: pick a new --out (never overwritten)" % out)
    cap = Q[q]["cap"] if cap == -1 else cap
    cands, dropped = candidates(pool, q, prefer)
    rows = [record(q, c) for c in select(cands, Q[q]["answers"], cap, seed)]
    if not rows:
        raise ValueError("%s: no rows selected" % q)
    os.makedirs(out)
    _link_frames(out, root, {f["pair_name"] for f in pool})
    counts = _write(out, q, rows)
    meta = {"question": q, "name": Q[q]["name"], "answers": list(Q[q]["answers"]), "dirs": list(Q[q]["answers"]),
            "root": os.path.abspath(root), "seed": seed, "cap_per_stratum": cap, "pool": len(pool),
            "candidates": dict(sorted(collections.Counter(c["answer"] for c in cands).items())),
            "dropped": dropped, "counts": counts, "strata": len({r["stratum"] for r in rows}),
            "prefer": {"given": len(prefer or ()), "selected": sum(1 for r in rows if r["tier"] == 0),
                       "selected_test": sum(1 for r in rows if r["tier"] == 0 and r["split"] == "test")},
            "matches": {f: len({(r["pair_name"], r["game"]) for r in rows if r["split"] == f}) for f in FILES},
            "split": "pairs_train.split3 (test crc32(pair:game) %% 3 == 2; val crc32('val:pair:game') %% 6 == 0)",
            "extras": extras(q, rows)}
    bad = problems(out, q)
    meta["problems"] = bad
    with open(os.path.join(out, "build.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    if bad:
        raise ValueError("%d problems in %s, e.g. %s" % (len(bad), out, bad[:5]))
    return meta


# ---- the builder's own check (not the selection code) --------------------------------------------------------------

def read_dataset(out: str) -> Dict[str, Dict[str, List[Dict]]]:
    """{dir: {split: rows}} of every answer dir (the shared frames/ dir is no answer)."""
    res: Dict[str, Dict[str, List[Dict]]] = {}
    for d in sorted(os.listdir(out)):
        if d != "frames" and os.path.isdir(os.path.join(out, d)):
            res[d] = {}
            for f in FILES:
                with open(os.path.join(out, d, f + ".jsonl")) as fh:
                    res[d][f] = [json.loads(x) for x in fh if x.strip()]
    return res


def problems(out: str, q: str) -> List[str]:
    """Dirs = the answers; every row: its answer = its dir, label = its index, the exact question (side asked for
    q3 / q4, none for q1 / q5, no character name), its match's split, images resolve; ids unique; no match in two
    splits; every stratum holds every answer equally (the alignment)."""
    spec, bad = Q[q], []
    data = read_dataset(out)
    if sorted(data) != sorted(spec["answers"]):
        bad.append("dirs %s are not the answers %s" % (sorted(data), sorted(spec["answers"])))
    ids, split_of_match = collections.Counter(), collections.defaultdict(set)
    strata: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for d, files in data.items():
        for f, rows in files.items():
            for r in rows:
                ids[r["id"]] += 1
                m = (r["pair_name"], r["game"])
                split_of_match[m].add(f)
                strata[r["stratum"]][r["answer"]] += 1
                text = spec["text"] % r.get("side") if spec["side"] else spec["text"]
                if r["answer"] != d or list(r["question"]["criteria"]) != list(spec["answers"]) or \
                        list(spec["answers"]).index(d) != r["label"] or r["question"]["instructions"] != text:
                    bad.append("%s: answer %s / label %s / question %r in dir %s" % (
                        r["id"], r["answer"], r["label"], r["question"]["instructions"], d))
                if spec["side"] and r.get("side") not in T.SIDES:
                    bad.append("%s: no side" % r["id"])
                if any(n in r["question"]["instructions"] for n in T.NAMES.values()):
                    bad.append("%s: question names a character" % r["id"])
                if r["split"] != f or f != T.split3(*m):
                    bad.append("%s: in %s, split %s, match belongs to %s" % (r["id"], f, r["split"], T.split3(*m)))
                if not all(os.path.exists(os.path.join(out, d, p)) for p in r["images"]):
                    bad.append("%s: image missing" % r["id"])
    bad += ["id %s on %d rows" % (i, n) for i, n in ids.items() if n > 1]
    bad += ["match %s:%d in %s" % (m[0], m[1], sorted(s)) for m, s in split_of_match.items() if len(s) > 1]
    bad += ["stratum %s unequal: %s" % (k, dict(c)) for k, c in strata.items()
            if len({c[a] for a in spec["answers"]}) != 1]
    return bad


def summary(meta: Dict) -> str:
    return "; ".join("%s %s" % (a, " ".join("%s %d" % (f, n) for f, n in fs.items()))
                     for a, fs in meta["counts"].items())
