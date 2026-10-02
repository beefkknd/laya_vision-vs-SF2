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

Answer dirs (air, dist): every row of <dir>/ has the answer <dir>, so scripts/train.py --balance sampling (laya draws
every --data dir in equal shares) draws the two answers equally without copies. move and face are one dir each: the
build caps every (character, movement, facing) cell, so their answers are already (near) equal.
"""
import collections
import json
import os
import zlib
from typing import Dict, Iterable, List, Sequence, Tuple

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
# dataset name -> (pairs-build question, the dirs: None = one dir named after the dataset, else one per answer)
DATASETS: Dict[str, Tuple[str, bool]] = {"move": ("movement", False), "face": ("facing", False),
                                         "air": ("air", True), "dist": ("distance", True)}
OUT = "test_data_mv_%s"


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


KEEP = ("decision", "pair", "pair_name", "game", "slot", "controller", "char", "opp", "t", "k_prev", "k_now",
        "movement", "direction", "facing", "air", "distance", "pressed", "pressed_class", "mv_source")


def record(q: str, r: Dict) -> Dict:
    """A pairs-build row of question ``q`` -> a laya record (images relative to the dataset dir, via frames/)."""
    crit = CRITERIA[q]
    if r["question_key"] != q or r["answer"] not in crit:
        raise ValueError("%s: not a %s row with a known answer" % (r.get("id"), q))
    return dict({"id": r["id"], "images": list(r["images"]), "question": question(q, r["char"]),
                 "label": list(crit).index(r["answer"]), "answer": r["answer"], "question_key": q,
                 "split": split3(r["pair_name"], r["game"])}, **{k: r[k] for k in KEEP})


def dir_of(dataset: str, rec: Dict) -> str:
    return rec["answer"] if DATASETS[dataset][1] else dataset


def records(src: str, dataset: str) -> List[Dict]:
    q = DATASETS[dataset][0]
    rows: List[Dict] = []
    for f in D.FILES:
        with open(os.path.join(src, q, f + ".jsonl")) as fh:
            rows += [json.loads(x) for x in fh if x.strip()]
    return [record(q, r) for r in rows]


def build_one(src: str, dataset: str, out: str) -> Dict:
    """Write <out>/<dir>/{train,val,test}.jsonl and <out>/<dir>/frames -> <src>/frames; refuses an existing out."""
    if os.path.exists(out):
        raise FileExistsError("%s exists: pick a new --out (never overwritten)" % out)
    recs = records(src, dataset)
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
            "matches": {f: len({(r["pair_name"], r["game"]) for r in recs if r["split"] == f}) for f in FILES}}
    bad = problems(out, src, dataset)
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


def problems(out: str, src: str, dataset: str) -> List[str]:
    """Independent of how the files were written: every source row exactly once; each row in the split of its match
    (val only from training matches); no match in two splits; answer dirs hold only their answer; label = the answer's
    criteria index; the question names the row's fighter; images resolve through the dir's frames link."""
    q = DATASETS[dataset][0]
    data = read_dataset(out)
    src_ids = set()
    for f in D.FILES:
        with open(os.path.join(src, q, f + ".jsonl")) as fh:
            src_ids |= {json.loads(x)["id"] for x in fh if x.strip()}
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
                if NAMES[r["char"]] not in r["question"]["instructions"]:
                    bad.append("%s %s: question does not name %s" % (dataset, r["id"], r["char"]))
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
