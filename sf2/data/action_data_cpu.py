"""The CPU Chun-Li action dataset (docs/prereg_movement_data.md, "Addition: CPU Chun-Li games";
scripts/build_action_data_cpu.py): from the collection rollouts/mv4 (one dir per player-1 character, CPU Chun-Li as
player 2), Chun-Li's action rows only, in the act set's format (sf2.data.action_data):

  key "act"    "What move is Chun-Li (him) doing?"   options: every Chun-Li code on any pair of this collection
  key "stage"  "Which part of Chun-Li's (him) move is this?"   stg1 start / stg2 middle / stg3 end

Each row's note (state_text) is "me=<player-1 character>": the model sees the game from player 1's side, Chun-Li is
"him". Every row carries the provenance (sf2.data.cpu_chunli.PROVENANCE_KEYS) and "char" = the player-1 character,
"opp" = "chunli". A pair that is not Chun-Li's (player 2), whose provenance is not its dir's, or whose collector label
the RAM does not give, is a problem (exit 1). Split by whole games as the act set; natural counts, no copies.
"""
import collections
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

from . import action_codes as A
from . import action_collect_io as IO
from . import cpu_chunli as K
from . import u_data as U
from .action_data import FILES, HEAD_MAX, KEEP, THRESHOLDS, act_question, head_tokens, pair_problems, \
    stage_question, _read_ram
from .movement_collect import LAG

PLAYER = 2                       # Chun-Li's slot
OUT = "test_data_act_cpu"
LAYOUT = "cpu_chunli"


def label_p1(job: Tuple[str, str]) -> Tuple[str, List[Dict], Dict, List[str]]:
    """One player-1 character's committed pairs -> (p1, labelled pairs, stats, problems). A labelled pair is the
    collector's pair plus "chunli": (code, stg, [s, e], cut_end) read from the RAM at t."""
    p1, root = job
    src = os.path.join(root, p1)
    pairs, dropped = IO.committed_pairs(src)
    img_dir = os.path.join(src, "images")
    want = K.provenance(p1)
    problems: List[str] = []
    seen = collections.Counter((p["game"], p["t"]) for p in pairs)
    problems += ["%s g%d t%d: %d pairs at one displayed row" % (p1, g, t, n) for (g, t), n in seen.items() if n > 1]
    by_game: Dict[int, List[Dict]] = collections.defaultdict(list)
    for p in pairs:
        by_game[p["game"]].append(p)
    out: List[Dict] = []
    stats = collections.Counter()
    for game in sorted(by_game):
        rows = _read_ram(src, game)
        if rows is None:
            problems.append("%s g%d: no RAM file" % (p1, game))
            continue
        labs = A.labels_at(rows, PLAYER)
        for p in by_game[game]:
            where = "%s g%d t%d" % (p1, game, p["t"])
            bad = ["%s: %s" % (where, x) for x in K.provenance_problems(p, want)]
            if (p.get("actor"), p.get("player")) != (K.CPU, PLAYER):
                bad.append("%s: not a Chun-Li pair (actor %s, player %s)" % (where, p.get("actor"), p.get("player")))
            bad += pair_problems(p1, p, img_dir)
            lab = labs.get(p["t"])
            if lab is None or (lab[0].code, lab[1]) != (p["code"], p["stg"]):
                bad.append("%s: collector label %s stg%s, RAM gives %s" % (
                    where, p["code"], p["stg"], None if lab is None else (lab[0].code, lab[1])))
            if bad:
                problems += bad
                continue
            ep, stg = lab
            out.append(dict(p, chunli=(ep.code, stg, [ep.start, ep.end], ep.cut_end)))
    stats["dropped_uncommitted"] = dropped
    stats["pairs"] = len(out)
    return p1, out, dict(stats), problems


def act_rows(p1: str, p: Dict, codes: Sequence[int]) -> List[Dict]:
    code, stg, ep, cut_end = p["chunli"]
    did = "%s-%s_%s_g%04d_t%05d" % (K.CPU, K.COLLECTION, p1, p["game"], p["t"])
    note = K.note(p1)
    if note != U.eye_note(p1):
        raise AssertionError("the provenance note %r is not the model's note %r" % (note, U.eye_note(p1)))
    base = dict({"decision": did, "char": p1, "opp": K.CPU, "sub": p1, "game": p["game"], "frame": p["k_now"],
                 "split": p["split"], "images": ["frames/" + n for n in p["images"]], "state_text": note,
                 "perception": True, "note_version": U.NOTE_VERSION, "frames_version": U.FRAMES_VERSION,
                 "copy": 0, "src_actor": p["actor"], "src_code": p["code"], "src_stg": p["stg"],
                 "actor": K.CPU, "player": PLAYER, "code": code, "stg": stg,
                 "label_full": A.label(K.CPU, code, stg), "actor_episode": ep, "actor_cut_end": cut_end},
                **{k: p[k] for k in KEEP}, **K.provenance(p1))
    out = []
    for key, q, answer in (("act", act_question(K.CPU, PLAYER, codes), A.code_name(code)),
                           ("stage", stage_question(K.CPU, PLAYER), "stg%d" % stg)):
        out.append(dict(base, id="%s-%s-him" % (did, key), task=key, key=key, question=q,
                        label=list(q["criteria"]).index(answer), answer=answer))
    return out


def build(root: str, out: str = OUT, p1s: Optional[Sequence[str]] = None, workers: int = 1,
          thresholds: str = THRESHOLDS, tokenizer=None) -> Dict:
    """Chun-Li's act set of the collection ``root`` into ``out`` (it may not exist and be non-empty)."""
    if os.path.exists(out) and os.listdir(out):
        raise SystemExit("%s exists and is not empty: remove it first" % out)
    want = sorted(p1s or (o for o in os.listdir(root) if os.path.exists(os.path.join(root, o, "games.jsonl"))))
    jobs = [(o, root) for o in want]
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
            done = list(ex.map(label_p1, jobs))
    else:
        done = [label_p1(j) for j in jobs]
    labelled = {o: pairs for o, pairs, _, _ in done}
    problems = [x for _, _, _, ps in done for x in ps]
    codes = sorted({p["chunli"][0] for pairs in labelled.values() for p in pairs})
    heads = {}
    if tokenizer is not None:
        heads = {"act chunli him": head_tokens(act_question(K.CPU, PLAYER, codes), tokenizer),
                 "stage chunli him": head_tokens(stage_question(K.CPU, PLAYER), tokenizer)}
        problems += ["question %s: %d head tokens > %d (laya cuts it)" % (k, n, HEAD_MAX)
                     for k, n in heads.items() if n > HEAD_MAX]
    os.makedirs(out, exist_ok=True)
    stats = {}
    for p1, pairs in sorted(labelled.items()):
        base = os.path.join(out, p1)
        os.makedirs(base)
        os.symlink(os.path.abspath(os.path.join(root, p1, "images")), os.path.join(base, "frames"))
        files: Dict[str, List[Dict]] = {n: [] for n in FILES.values()}
        for p in pairs:
            files[FILES[p["split"]]] += act_rows(p1, p, codes)
        ids = collections.Counter(r["id"] for rows in files.values() for r in rows)
        problems += ["%s: id %s on %d rows" % (p1, i, n) for i, n in ids.items() if n > 1]
        for name, rows in files.items():
            with open(os.path.join(base, name + ".jsonl"), "w") as f:
                f.write("".join(json.dumps(r) + "\n" for r in sorted(rows, key=lambda r: r["id"])))
        stats[p1] = {"rows": {k: len(v) for k, v in files.items()},
                     "answers": {k: dict(collections.Counter("%s %s" % (r["key"], r["answer"]) for r in v))
                                 for k, v in files.items()}}
    meta = {"root": os.path.abspath(root), "dataset": "act", "layout": LAYOUT, "me": None, "cpu": K.CPU,
            "players": [PLAYER], "opps": want, "lag": LAG, "options": {K.CPU: [A.code_name(c) for c in codes]},
            "head_tokens": heads, "split": {"test": [2, 5, 8], "val": [0]}, "copies": 0, "codes": A.table(),
            "provenance_keys": list(K.PROVENANCE_KEYS), "run": K.read_run(root),
            "frames_version": U.FRAMES_VERSION, "note_version": U.NOTE_VERSION, "problems": len(problems)}
    with open(os.path.join(out, "build.json"), "w") as f:
        json.dump(meta, f, indent=1, sort_keys=True)
    return {"problems": problems, "stats": stats, "label_stats": {o: s for o, _, s, _ in done}, "codes": codes,
            "head_tokens": heads}
