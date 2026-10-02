"""Round 3 of docs/prereg_movement_finetunes.md, the "act" dataset (scripts/build_mv3_data.py act):

  "What is the fighter on the <side> doing?" -> moving / attack / special

moving = stand, walk (toward / away), crouch, jump, block, hit, down; attack = the grid's "attack" (normal, jump attack,
throw); special = the grid's "special" (the pressed-move rule, RAM-confirmed). Built from the SAME gated pairs build as
round 2 (sf2.data.pairs_train.records(src, "move", ask="side")): the same frames, sides (smaller x at the displayed
row t, equal x dropped and counted) and splits (test by match, val = 1 in 6 training matches). One data dir per answer
(moving/, attack/, special/), so scripts/train.py --balance draws the three answers equally without copies.
"""
import collections
import json
import os
import zlib
from typing import Dict, List, Sequence

from . import pairs_data as D
from . import pairs_train as T

ACT_ANSWERS = ("moving", "attack", "special")
ACT_CRITERIA = {"moving": "moving (standing, walking, crouching, jumping, blocking, being hit or knocked down)",
                "attack": "attacking (a normal, a jump attack or a throw)",
                "special": "doing a special move"}
ACT_TEXT = "What is the fighter on the %s doing?"
_ACT_OF = {"attack": "attack", "special": "special"}
OUT = "test_data_mv3_act"


def act_of(movement10: str) -> str:
    if movement10 not in D.MOVEMENT_ANSWERS:
        raise ValueError("%r is not one of the grid movements %s" % (movement10, D.MOVEMENT_ANSWERS))
    return _ACT_OF.get(movement10, "moving")


def collapse(movements: Sequence[str]) -> List[str]:
    return [act_of(m) for m in movements]


def question_act(side: str) -> Dict:
    if side not in T.SIDES:
        raise ValueError("side %r is not one of %s" % (side, T.SIDES))
    return {"type": "choice", "instructions": ACT_TEXT % side, "criteria": dict(ACT_CRITERIA)}


def record_act(r: Dict) -> Dict:
    """A round-2 movement record (asked by side) -> the act record: same images, side, split; answer, label and
    question replaced; the 10-movement answer kept as movement10 and its id as source_id."""
    act = act_of(r["answer"])
    out = dict(r, id=r["id"].rsplit("-", 1)[0] + "-act", source_id=r["id"], movement10=r["answer"], answer=act,
               label=ACT_ANSWERS.index(act), question_key="act", question=question_act(r["side"]))
    return out


def build_act(src: str, out: str) -> Dict:
    """Write <out>/{moving,attack,special}/{train,val,test}.jsonl (+ frames link) and <out>/build.json."""
    if os.path.exists(out):
        raise FileExistsError("%s exists: pick a new --out (never overwritten)" % out)
    recs, dropped = T.records(src, "move", ask="side")
    acts = [record_act(r) for r in recs]
    by_dir: Dict[str, Dict[str, List[Dict]]] = {a: {f: [] for f in T.FILES} for a in ACT_ANSWERS}
    for r in acts:
        by_dir[r["answer"]][r["split"]].append(r)
    frames = os.path.abspath(os.path.join(src, "frames"))
    os.makedirs(out)
    counts = {}
    for d, files in by_dir.items():
        base = os.path.join(out, d)
        os.makedirs(base)
        os.symlink(frames, os.path.join(base, "frames"))
        for f, rows in files.items():
            D._write_jsonl(os.path.join(base, f + ".jsonl"), sorted(rows, key=lambda x: x["id"]))
        counts[d] = {f: len(rows) for f, rows in files.items()}
    meta = {"source": os.path.abspath(src), "dataset": "act", "question": "act", "answers": list(ACT_ANSWERS),
            "answer_dirs": True, "dirs": list(ACT_ANSWERS), "counts": counts, "ask": "side",
            "mapping": {m: act_of(m) for m in D.MOVEMENT_ANSWERS},
            "movement10": {f: dict(sorted(collections.Counter(r["movement10"] for r in acts if r["split"] == f).items()))
                           for f in T.FILES},
            "sides": {f: dict(sorted(collections.Counter(r["side"] for r in acts if r["split"] == f).items()))
                      for f in T.FILES},
            "matches": {f: len({(r["pair_name"], r["game"]) for r in acts if r["split"] == f}) for f in T.FILES},
            "dropped_equal_x": {"total": len(dropped), "ids": sorted(dropped)}}
    bad = problems_act(out, src)
    meta["problems"] = bad
    with open(os.path.join(out, "build.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    if bad:
        raise ValueError("%d problems in %s, e.g. %s" % (len(bad), out, bad[:5]))
    return meta


def problems_act(out: str, src: str) -> List[str]:
    """Independent of how the files were written: every movement row of the pairs build exactly once except the
    equal-x rows (absent); the act answer re-derived from the source row's movement / direction (attack -> attack,
    special -> special, anything else moving), in its own dir, label = its index; the side from RAM at t
    (pairs_train._side_expected) and the exact question text, no character name; the match's split (val only from
    training matches, no match in two splits); images resolve."""
    src_rows = {r["id"]: r for r in T.source_rows(src, "movement")}
    sides = T._side_expected(src, "movement")
    want_ids = {i for i in src_rows if sides[i] is not None}
    seen = collections.Counter()
    split_of_match = collections.defaultdict(set)
    bad: List[str] = []
    for d, files in T.read_dataset(out).items():
        if d not in ACT_ANSWERS:
            bad.append("dir %s is no act answer" % d)
        for f, rows in files.items():
            for r in rows:
                sid = r.get("source_id")
                seen[sid] += 1
                s = src_rows.get(sid)
                if s is None:
                    bad.append("%s: source %s not in the pairs build" % (r.get("id"), sid))
                    continue
                mv = s["movement"]
                act = "attack" if mv == "attack" else "special" if mv == "special" else "moving"
                if r["answer"] != act or d != act:
                    bad.append("%s: answer %s in dir %s, source movement %s -> %s" % (sid, r["answer"], d, mv, act))
                crit = r["question"]["criteria"]
                if tuple(crit) != ACT_ANSWERS or r["label"] != ACT_ANSWERS.index(act):
                    bad.append("%s: label %s is not %s" % (sid, r["label"], act))
                side = sides.get(sid)
                text = r["question"]["instructions"]
                if r.get("side") != side or text != "What is the fighter on the %s doing?" % side:
                    bad.append("%s: side %s / %r, RAM says %s" % (sid, r.get("side"), text, side))
                if any(n in text for n in T.NAMES.values()):
                    bad.append("%s: question names a character" % sid)
                m = (s["pair_name"], s["game"])
                split_of_match[m].add(f)
                want = ("test" if D.split_of_game(*m) == "test" else
                        "val" if zlib.crc32(("val:%s:%d" % m).encode()) % T.VAL_MOD == T.VAL_REST else "train")
                if r["split"] != f or f != want:
                    bad.append("%s: in %s, split %s, match belongs to %s" % (sid, f, r["split"], want))
                if r["images"] != s["images"] or not all(os.path.exists(os.path.join(out, d, p)) for p in r["images"]):
                    bad.append("%s: images %s" % (sid, r["images"]))
    bad += ["%s on %d rows" % (i, n) for i, n in seen.items() if n > 1]
    bad += ["source row %s missing" % i for i in sorted(want_ids - set(seen))[:20]]
    bad += ["row %s should be dropped (equal x)" % i for i in sorted(set(seen) - want_ids)[:20]]
    bad += ["match %s:%d in %s" % (m[0], m[1], sorted(s)) for m, s in split_of_match.items() if len(s) > 1]
    return bad
