"""The movement dataset v2 (docs/prereg_movement_data.md; scripts/build_movement_data2.py -> test_data_mv2/<opp>/).

One dir per opponent (Chun-Li vs him), from the collector's committed games (sf2.data.movement_collect_io; pairs of a
game without its games.jsonl line are dropped and counted):
- the pictures: <out>/<opp>/frames is a symlink to the collection's <opp>/images (hud_frame PNGs, no copies); a row's
  images are frames/g<game>_k<k_prev>.png and frames/g<game>_k<k_now>.png (k_now = t + LAG, k_prev = t - 4 + LAG);
- the note: v3, "me=chunli" (u_data.eye_note); "perception": true, as the U arm's and test_data_mv's rows;
- the question: sf2.data.movement's one question, label = the answer's index; the answer as the collector stored it
  (re-derived independently from the stored RAM by the dataset gate, sf2.data.movement_gate);
- the split by whole games: game % 10 in 2,5,8 -> test_real.jsonl, 0 -> val.jsonl, the rest -> train.jsonl;
- natural counts: no repetition copies (balance is the collector's quotas). The sampling fields (stage, stage_bin,
  pos, length, episode, long, cut_end, t, k_prev, k_now) are kept on every row.
"""
import collections
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

from . import movement as M
from . import movement_collect as C
from . import movement_collect_io as IO
from . import u_data as U

OUT = "test_data_mv2"
ME = "chunli"
FILES = {"train": "train", "val": "val", "test": "test_real"}
NOTE_VERSION, FRAMES_VERSION = U.NOTE_VERSION, U.FRAMES_VERSION
KEEP = ("t", "k_prev", "k_now", "stage", "stage_bin", "pos", "length", "episode", "long", "cut_end")


def image_name(game: int, k: int) -> str:
    return "g%04d_k%05d.png" % (game, k)


def row_of(opp: str, p: Dict) -> Dict:
    q = M.movement_question()
    did = "%s-mv2_%s_g%04d_t%05d" % (ME, opp, p["game"], p["t"])
    return dict({"decision": did, "char": ME, "opp": opp, "sub": opp, "game": p["game"], "frame": p["k_now"],
                 "split": p["split"], "images": ["frames/" + n for n in p["images"]], "state_text": U.eye_note(ME),
                 "perception": True, "note_version": NOTE_VERSION, "frames_version": FRAMES_VERSION,
                 "id": "%s-%s" % (did, M.KEY), "task": M.KEY, "key": M.KEY, "question": q,
                 "label": list(q["criteria"]).index(p["answer"]), "answer": p["answer"], "copy": 0},
                **{k: p[k] for k in KEEP})


def pair_problems(opp: str, p: Dict, img_dir: str) -> List[str]:
    where = "%s g%s t%s" % (opp, p.get("game"), p.get("t"))
    if p.get("answer") not in M.ANSWERS:
        return ["%s: answer %r is not one of the movement answers" % (where, p.get("answer"))]
    out = []
    if p["split"] != C.split_of_game(p["game"]):
        out.append("%s: split %s but game %d belongs to %s" % (where, p["split"], p["game"],
                                                              C.split_of_game(p["game"])))
    want = [image_name(p["game"], p["t"] - 4 + C.LAG), image_name(p["game"], p["t"] + C.LAG)]
    if p["images"] != want or (p["k_prev"], p["k_now"]) != (p["t"] - 4 + C.LAG, p["t"] + C.LAG):
        out.append("%s: images %s are not frames (t - 4, t) shown at lag %d %s" % (where, p["images"], C.LAG, want))
    out += ["%s: missing image %s" % (where, n) for n in p["images"] if not os.path.exists(os.path.join(img_dir, n))]
    return out


def counts_of(rows: List[Dict]) -> Dict[str, int]:
    c = collections.Counter(r["answer"] for r in rows)
    return {a: c[a] for a in M.ANSWERS}


def opp_build(job: Tuple[str, str, str, Dict[str, int]]) -> Tuple[str, Dict, List[str]]:
    opp, root, out, quota = job
    src, base = os.path.join(root, opp), os.path.join(out, opp)
    torn: List[int] = []
    pairs, dropped = IO.committed_pairs(src)
    IO.read_jsonl(os.path.join(src, "pairs.jsonl"), torn)
    img_dir = os.path.join(src, "images")
    problems: List[str] = []
    files: Dict[str, List[Dict]] = {n: [] for n in FILES.values()}
    for p in pairs:
        bad = pair_problems(opp, p, img_dir)
        if bad:
            problems += bad
            continue
        files[FILES[p["split"]]].append(row_of(opp, p))
    ids = collections.Counter(r["id"] for rows in files.values() for r in rows)
    problems += ["%s: id %s on %d rows" % (opp, i, n) for i, n in ids.items() if n > 1]
    counts = {s: counts_of(files[FILES[s]]) for s in FILES}
    problems += ["%s %s: %d %s pairs above the quota %d" % (opp, s, n, a, quota[s])
                 for s, c in counts.items() for a, n in c.items() if n > quota[s]]
    files = {k: sorted(v, key=lambda r: r["id"]) for k, v in files.items()}
    os.makedirs(base)
    os.symlink(os.path.abspath(img_dir), os.path.join(base, "frames"))
    for name, rows in files.items():
        with open(os.path.join(base, name + ".jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
    games = IO.read_games(src)
    all_rows = [r for rows in files.values() for r in rows]
    stats = {"counts": counts, "rows": {k: len(v) for k, v in files.items()},
             "games": dict(collections.Counter(C.split_of_game(g["game"]) for g in games)),
             "dropped_uncommitted": dropped, "torn_lines": len(torn),
             "stage_bins": {a: dict(collections.Counter(r["stage_bin"] for r in all_rows if r["answer"] == a))
                            for a in M.ANSWERS},
             "long": sum(r["long"] for r in all_rows), "cut_end": sum(r["cut_end"] for r in all_rows)}
    with open(os.path.join(base, "stats.json"), "w") as f:
        json.dump(stats, f, indent=1, sort_keys=True)
    return opp, stats, problems


def quota_of(root: str) -> Dict[str, int]:
    path = os.path.join(root, "run.json")
    if os.path.exists(path):
        return dict(json.load(open(path)).get("quota") or C.SPLIT_QUOTA)
    return dict(C.SPLIT_QUOTA)


def build(root: str, out: str = OUT, opps: Optional[Sequence[str]] = None, workers: int = 1) -> Dict:
    """Every opponent of the collection ``root`` into ``out`` (which must not exist or be empty)."""
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit("%s exists and is not empty: remove it first" % out)
    want = sorted(opps or (o for o in os.listdir(root) if os.path.exists(os.path.join(root, o, "games.jsonl"))))
    quota = quota_of(root)
    os.makedirs(out, exist_ok=True)
    jobs = [(o, root, out, quota) for o in want]
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
            done = list(ex.map(opp_build, jobs))
    else:
        done = [opp_build(j) for j in jobs]
    problems = [p for _, _, ps in done for p in ps]
    meta = {"root": os.path.abspath(root), "me": ME, "opps": want, "quota": quota, "lag": C.LAG,
            "walk_px": M.WALK_PX, "question": M.movement_question(), "answers": list(M.ANSWERS),
            "split": {"test": list(C.TEST_INDEX), "val": list(C.VAL_INDEX)}, "copies": 0,
            "frames_version": FRAMES_VERSION, "note_version": NOTE_VERSION}
    with open(os.path.join(out, "build.json"), "w") as f:
        json.dump(dict(meta, problems=len(problems)), f, indent=1, sort_keys=True)
    return {"problems": problems, "counts": {o: s for o, s, _ in done}, "meta": meta}
