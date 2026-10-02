"""The four fine-tune datasets of docs/prereg_movement_finetunes.md (scripts/build_mv_data.py), cut from ONE gated
movement-pairs build (sf2.data.pairs_data; scripts/build_pairs_data.py + gate_pairs_data.py), in laya-vision's record
layout (laya.vlm_train.jsonl_example): one question per dataset, never mixed.

  move   "What is <Name> doing?"                                the 10 grid movements   one dir: move/
  face   "Which way is <Name> facing?"                          left / right            one dir: face/
  air    "Is <Name> on the ground or in the air?"               ground / air            one dir per answer
  dist   "Is <Name> close to or far from the other fighter?"    close / far             one dir per answer

<Name> is the fighter the row is about (the pairs are A vs B, two different characters, so the name is unique on the
screen). No note (state_text): the input is the pair's two frames (n - 4, n), HUD visible, and the question.

Splits: test = the build's test (crc32 of the match % 3 == 2, sf2.data.pairs_data.split_of_game), never touched;
val = whole TRAINING matches with crc32("val:<A>_vs_<B>:<game>") % VAL_MOD == VAL_REST (1 in 6 training matches);
train = the other training matches. So val and test never share a match with train or each other.

Round 2 (ask="side", scripts/build_mv_data.py --ask side -> test_data_mv2_<dataset>): the question names the fighter
by SCREEN SIDE instead ("the fighter on the left" / "on the right"); side = which fighter has the smaller x at the
displayed RAM row t (lag 1: the row the labels come from; the image is captured at t + 1). Rows whose two x are equal
at t are dropped and counted (build.json dropped_equal_x). Same rows, labels, splits and dirs otherwise.

Answer dirs (air, dist): every row of <dir>/ has the answer <dir>, so scripts/train.py --balance sampling (laya draws
every --data dir in equal shares) draws the two answers equally without copies. move and face are one dir each: the
build caps every (character, movement, facing) cell, so their answers are already (near) equal.
"""
import collections
import json
import os
import zlib
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import movement_collect_io as MIO
from . import pairs_data as D

VAL_MOD, VAL_REST = 6, 0
FILES = ("train", "val", "test")
NAMES = {"chunli": "Chun-Li", "ryu": "Ryu", "ken": "Ken", "guile": "Guile", "blanka": "Blanka", "dhalsim": "Dhalsim",
         "honda": "E. Honda", "zangief": "Zangief"}
CRITERIA: Dict[str, Dict[str, str]] = {
    "movement": {"stand": "standing still", "walk toward": "walking toward the other fighter",
                 "walk away": "walking away from the other fighter", "crouch": "crouching",
                 "jump": "jumping", "attack": "attacking (a normal, a jump attack or a throw)",
                 "special": "doing a special move", "block": "blocking", "hit": "being hit",
                 "down": "knocked down, on the ground"},
    "facing": {"left": "facing left", "right": "facing right"},
    "air": {"ground": "on the ground", "air": "in the air"},
    "distance": {"close": "close (throw or poke range)", "far": "far"},
}
INSTRUCTIONS = {"movement": "What is %s doing?", "facing": "Which way is %s facing?",
                "air": "Is %s on the ground or in the air?",
                "distance": "Is %s close to or far from the other fighter?"}
SIDES = ("left", "right")
SIDE_INSTRUCTIONS = {"movement": "What is the fighter on the %s doing?",
                     "facing": "Which way is the fighter on the %s facing?",
                     "air": "Is the fighter on the %s on the ground or in the air?",
                     "distance": "Is the fighter on the %s close to or far from the other fighter?"}
ASKS = ("name", "side")
# dataset name -> (pairs-build question, the dirs: None = one dir named after the dataset, else one per answer)
DATASETS: Dict[str, Tuple[str, bool]] = {"move": ("movement", False), "face": ("facing", False),
                                         "air": ("air", True), "dist": ("distance", True)}
OUT = "test_data_mv_%s"
OUT_BY_ASK = {"name": OUT, "side": "test_data_mv2_%s"}


def is_val_match(pair_name: str, game: int) -> bool:
    """1 in VAL_MOD training matches (whole match); never a test match."""
    if D.split_of_game(pair_name, game) != "train":
        return False
    return zlib.crc32(("val:%s:%d" % (pair_name, game)).encode()) % VAL_MOD == VAL_REST


def split3(pair_name: str, game: int) -> str:
    if D.split_of_game(pair_name, game) == "test":
        return "test"
    return "val" if is_val_match(pair_name, game) else "train"


def question(q: str, char: str) -> Dict:
    crit = CRITERIA[q]
    if tuple(crit) != D.QUESTION_ANSWERS[q]:
        raise AssertionError("criteria %s != answers %s" % (tuple(crit), D.QUESTION_ANSWERS[q]))
    return {"type": "choice", "instructions": INSTRUCTIONS[q] % NAMES[char], "criteria": dict(crit)}


def question_side(q: str, side: str) -> Dict:
    if side not in SIDES:
        raise ValueError("side %r is not one of %s" % (side, SIDES))
    crit = CRITERIA[q]
    if tuple(crit) != D.QUESTION_ANSWERS[q]:
        raise AssertionError("criteria %s != answers %s" % (tuple(crit), D.QUESTION_ANSWERS[q]))
    return {"type": "choice", "instructions": SIDE_INSTRUCTIONS[q] % side, "criteria": dict(crit)}


def side_of(ram_row: Dict[str, int], slot: int) -> Optional[str]:
    """'left' when ``slot``'s fighter has the smaller x in this RAM row, 'right' when the larger, None when equal."""
    me, other = ram_row["p%d_x" % slot], ram_row["p%d_x" % (3 - slot)]
    if me == other:
        return None
    return "left" if me < other else "right"


def side_at(ram: Sequence[Dict[str, int]], r: Dict) -> Optional[str]:
    """The row's side at its displayed RAM row t (never t - 4, never the capture row t + 1)."""
    return side_of(ram[r["t"]], r["slot"])


def _ram_root(src: str) -> str:
    with open(os.path.join(src, "build.json")) as fh:
        return json.load(fh)["root"]


def _ram_path(root: str, pair_name: str, game: int) -> str:
    return os.path.join(root, pair_name, "ram", "g%04d.json.gz" % game)


KEEP = ("decision", "pair", "pair_name", "game", "slot", "controller", "char", "opp", "t", "k_prev", "k_now",
        "movement", "direction", "facing", "air", "distance", "pressed", "pressed_class", "mv_source")


def record(q: str, r: Dict, side: Optional[str] = None) -> Dict:
    """A pairs-build row of question ``q`` -> a laya record (images relative to the dataset dir, via frames/). With
    ``side`` the question names the fighter by screen side (round 2) and the record keeps ``side``."""
    crit = CRITERIA[q]
    if r["question_key"] != q or r["answer"] not in crit:
        raise ValueError("%s: not a %s row with a known answer" % (r.get("id"), q))
    asked = question(q, r["char"]) if side is None else question_side(q, side)
    extra = {} if side is None else {"side": side}
    return dict({"id": r["id"], "images": list(r["images"]), "question": asked,
                 "label": list(crit).index(r["answer"]), "answer": r["answer"], "question_key": q,
                 "split": split3(r["pair_name"], r["game"])}, **{k: r[k] for k in KEEP}, **extra)


def dir_of(dataset: str, rec: Dict) -> str:
    return rec["answer"] if DATASETS[dataset][1] else dataset


def source_rows(src: str, q: str) -> List[Dict]:
    rows: List[Dict] = []
    for f in D.FILES:
        with open(os.path.join(src, q, f + ".jsonl")) as fh:
            rows += [json.loads(x) for x in fh if x.strip()]
    return rows


def records(src: str, dataset: str, ask: str = "name") -> Tuple[List[Dict], List[str]]:
    """(the records, the ids dropped because the two x are equal at t (ask='side' only))."""
    if ask not in ASKS:
        raise ValueError("ask %r is not one of %s" % (ask, ASKS))
    q = DATASETS[dataset][0]
    rows = source_rows(src, q)
    if ask == "name":
        return [record(q, r) for r in rows], []
    root = _ram_root(src)
    rams: Dict[Tuple[str, int], List[Dict[str, int]]] = {}
    out, dropped = [], []
    for r in rows:
        m = (r["pair_name"], r["game"])
        if m not in rams:
            rams[m] = MIO.read_ram(_ram_path(root, *m))
        side = side_at(rams[m], r)
        if side is None:
            dropped.append(r["id"])
        else:
            out.append(record(q, r, side))
    return out, dropped


def build_one(src: str, dataset: str, out: str, ask: str = "name") -> Dict:
    """Write <out>/<dir>/{train,val,test}.jsonl and <out>/<dir>/frames -> <src>/frames; refuses an existing out."""
    if os.path.exists(out):
        raise FileExistsError("%s exists: pick a new --out (never overwritten)" % out)
    recs, dropped = records(src, dataset, ask)
    by_dir: Dict[str, Dict[str, List[Dict]]] = collections.defaultdict(lambda: {f: [] for f in FILES})
    for r in recs:
        by_dir[dir_of(dataset, r)][r["split"]].append(r)
    frames = os.path.abspath(os.path.join(src, "frames"))
    os.makedirs(out)
    counts = {}
    for d, files in sorted(by_dir.items()):
        base = os.path.join(out, d)
        os.makedirs(base)
        os.symlink(frames, os.path.join(base, "frames"))
        for f, rows in files.items():
            D._write_jsonl(os.path.join(base, f + ".jsonl"), sorted(rows, key=lambda x: x["id"]))
        counts[d] = {f: dict(sorted(collections.Counter(r["answer"] for r in rows).items()))
                     for f, rows in files.items()}
    meta = {"source": os.path.abspath(src), "dataset": dataset, "question": DATASETS[dataset][0],
            "answer_dirs": DATASETS[dataset][1], "dirs": sorted(by_dir), "counts": counts,
            "val": "train matches with crc32('val:<pair>:<game>') %% %d == %d" % (VAL_MOD, VAL_REST),
            "matches": {f: len({(r["pair_name"], r["game"]) for r in recs if r["split"] == f}) for f in FILES},
            "ask": ask}
    if ask == "side":
        src_split = {r["id"]: r["split"] for r in source_rows(src, DATASETS[dataset][0])}
        meta["dropped_equal_x"] = {"total": len(dropped), "ids": sorted(dropped),
                                   "by_split": dict(sorted(collections.Counter(src_split[i] for i in dropped).items()))}
        meta["sides"] = {f: dict(sorted(collections.Counter(r["side"] for r in recs if r["split"] == f).items()))
                         for f in FILES}
    bad = problems(out, src, dataset, ask)
    meta["problems"] = bad
    with open(os.path.join(out, "build.json"), "w") as fh:
        json.dump(meta, fh, indent=1, sort_keys=True)
    if bad:
        raise ValueError("%d problems in %s, e.g. %s" % (len(bad), out, bad[:5]))
    return meta


def read_dataset(out: str) -> Dict[str, Dict[str, List[Dict]]]:
    res: Dict[str, Dict[str, List[Dict]]] = {}
    for d in sorted(os.listdir(out)):
        if os.path.isdir(os.path.join(out, d)):
            res[d] = {}
            for f in FILES:
                with open(os.path.join(out, d, f + ".jsonl")) as fh:
                    res[d][f] = [json.loads(x) for x in fh if x.strip()]
    return res


def _side_expected(src: str, q: str) -> Dict[str, Optional[str]]:
    """Written apart from side_of / records: every source row id -> 'left' / 'right' from the two x of RAM row t,
    None when they are equal."""
    root = _ram_root(src)
    want: Dict[str, Optional[str]] = {}
    cache: Dict[Tuple[str, int], List[Dict[str, int]]] = {}
    for r in source_rows(src, q):
        key = (r["pair_name"], r["game"])
        ram = cache.setdefault(key, MIO.read_ram(_ram_path(root, *key)))
        xs = (ram[r["t"]]["p1_x"], ram[r["t"]]["p2_x"])
        mine, theirs = (xs[0], xs[1]) if r["slot"] == 1 else (xs[1], xs[0])
        want[r["id"]] = None if mine == theirs else SIDES[int(mine > theirs)]
    return want


def problems(out: str, src: str, dataset: str, ask: str = "name") -> List[str]:
    """Independent of how the files were written: every source row exactly once (ask='side': except the equal-x
    rows, which must be absent); each row in the split of its match (val only from training matches); no match in two
    splits; answer dirs hold only their answer; label = the answer's criteria index; the question names the row's
    fighter (by name, or by its RAM side and no character name); images resolve through the dir's frames link."""
    q = DATASETS[dataset][0]
    data = read_dataset(out)
    src_ids = {r["id"] for r in source_rows(src, q)}
    sides = _side_expected(src, q) if ask == "side" else {}
    if ask == "side":
        src_ids = {i for i in src_ids if sides[i] is not None}
    out_ids = collections.Counter()
    split_of_match: Dict[Tuple[str, int], set] = collections.defaultdict(set)
    bad: List[str] = []
    for d, files in data.items():
        if DATASETS[dataset][1] and d not in D.QUESTION_ANSWERS[q]:
            bad.append("%s: dir %s is no answer" % (dataset, d))
        if not DATASETS[dataset][1] and d != dataset:
            bad.append("%s: unexpected dir %s" % (dataset, d))
        for f, rows in files.items():
            for r in rows:
                out_ids[r["id"]] += 1
                m = (r["pair_name"], r["game"])
                split_of_match[m].add(f)
                want = ("test" if D.split_of_game(*m) == "test" else
                        "val" if zlib.crc32(("val:%s:%d" % m).encode()) % VAL_MOD == VAL_REST else "train")
                if r["split"] != f or f != want:
                    bad.append("%s %s: in %s, split %s, match belongs to %s" % (d, r["id"], f, r["split"], want))
                if DATASETS[dataset][1] and r["answer"] != d:
                    bad.append("%s %s: answer %s in dir %s" % (dataset, r["id"], r["answer"], d))
                crit = r["question"]["criteria"]
                if list(crit)[r["label"]] != r["answer"] or tuple(crit) != D.QUESTION_ANSWERS[q]:
                    bad.append("%s %s: label %s is not %s" % (dataset, r["id"], r["label"], r["answer"]))
                text = r["question"]["instructions"]
                if ask == "name" and NAMES[r["char"]] not in text:
                    bad.append("%s %s: question does not name %s" % (dataset, r["id"], r["char"]))
                if ask == "side":
                    want_side = sides.get(r["id"])
                    if r.get("side") != want_side or text != SIDE_INSTRUCTIONS[q] % want_side:
                        bad.append("%s %s: side %s / %r, RAM says %s" % (dataset, r["id"], r.get("side"), text,
                                                                       want_side))
                    if any(n in text for n in NAMES.values()):
                        bad.append("%s %s: question names a character" % (dataset, r["id"]))
                if not all(os.path.exists(os.path.join(out, d, p)) for p in r["images"]):
                    bad.append("%s %s: image missing" % (dataset, r["id"]))
    bad += ["%s: id %s on %d rows" % (dataset, i, n) for i, n in out_ids.items() if n > 1]
    bad += ["%s: source row %s missing" % (dataset, i) for i in sorted(src_ids - set(out_ids))[:20]]
    bad += ["%s: row %s not in the source" % (dataset, i) for i in sorted(set(out_ids) - src_ids)[:20]]
    bad += ["%s: match %s:%d in %s" % (dataset, m[0], m[1], sorted(s)) for m, s in split_of_match.items()
            if len(s) > 1]
    return bad


def summary(meta: Dict) -> str:
    return "; ".join("%s %s" % (d, " ".join("%s %d" % (f, sum(c.values())) for f, c in fs.items()))
                     for d, fs in meta["counts"].items())


def all_rows(out: str, split: str) -> List[Dict]:
    return [r for files in read_dataset(out).values() for r in files[split]]


def answer_counts(rows: Iterable[Dict]) -> Dict[str, int]:
    return dict(sorted(collections.Counter(r["answer"] for r in rows).items()))


def datasets(names: Sequence[str] = tuple(DATASETS)) -> List[str]:
    bad = [n for n in names if n not in DATASETS]
    if bad:
        raise ValueError("unknown datasets %s (of %s)" % (bad, list(DATASETS)))
    return list(names)
