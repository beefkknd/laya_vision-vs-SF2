"""The lesson loop, Chun-Li vs one opponent: Qwen proposes what to learn, code verifies and keeps the books
(sf2.system2.lessons), text laya plays with the lessons in play. Headless, paired with a no-advice arm (same savestate,
same seed, same start delays).

    python scripts/qwen_lessons.py --opp ken --games 10

After every game: code reviews the registry (claims in test judged or stopped when her rounds tanked since they
began, lessons whose evidence stopped holding retired),
then Qwen proposes at most 2 claims, code judges them. Starts from her play data against him (--no-history: nothing).
From a lock (sf2.eval.lock): --lock NAME --run N repeats the lock's run N with its checkpoints, savestate and play
data only (refused when a locked file changed); output under rollouts/locked/NAME/.
--prompt character: Qwen studies this one opponent - his threats and "if you see X" (sf2.system2.character_prompt);
the default "views" is the two-view prompt the lock lesson_loop_v1 ran with (sf2.system2.lesson_prompt).
--book lessons/book.json (scripts/book.py; default: none, the loop as before): the loop arm starts with this opponent's
verified players' tips in the registry ("verified": never retired, in play first; sf2.system2.lessons.from_book); the
run file, every ledger row and the verdict name the book and its sha256.
--oracle lessons/value_oracle_v1.json (default: none, laya-vision's P(hit) as before): System 1 ranks by the lookup
table (sf2.data.value_oracle; no laya-vision loaded), text laya rates its expected nets on the net scale
(sf2.system1.advice.rating); the run file and the verdict name the table and its sha256, the run name ends "+table"
(docs/prereg_2x2.md). --shared-text-laya: both arms ask the one shared text laya server (sf2.system1.shared_laya, which
reserves its own memory once) and each arm reserves config.RUN_JOB_SHARED_GB instead of MODEL_JOB_GB.
Since 2026-09-30 claims naming forward are refused and the prompts do not offer it for lessons (text laya was never
trained on such lessons, docs/component_boundaries.md); --forward-lessons restores both, for a replay of an old run.
Verdict (rollouts/qwen_lessons/<stamp>_<opp>/verdict.json):
    invariant     no registered lesson ever contradicts its own evidence (code-enforced; a violation is a bug)
    hypotheses    Qwen's valid claims that hold on all the data at the end (qwen_hold_rate), vs 500 random claims
                  judged the same way (random_hold_rate): a random kind on a random POPULATED cell - a move, range
                  and situation where the move has MIN_TRIES+ tries and the rest MIN_TRIES+ (random_cells of them);
                  a claim on an empty cell can never hold, so drawing over every cell (random_hold_rate_uniform, the
                  baseline before 2026-09-29) flatters Qwen. Qwen is shown the verifier's classes, so the verdict
                  also says how much of its hold rate was on the table already: registered_at_proposal(_share) of
                  the valid claims were registered the moment they were proposed; refused_duplicate(_share) of all
                  proposals were refused as "already registered / testing / rejected" (it repeated itself)
    outcome       hit points per round vs the no-advice arm, paired (reported, not a gate)
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Dict, List

import _path  # noqa: F401
from sf2.config import LAYA_VISION, MODEL_JOB_GB, PORTS, RUN_JOB_SHARED_GB, TEXT_LAYA
from sf2.data import value_oracle
from sf2.data.dataset import read
from sf2.eval import lock as lk
from sf2.eval.logs import load_actions, mark_run, play_dirs
from sf2.eval.runner import exit_on_sigterm, fan_out, open_fight, open_logs
from sf2.eval.stats import ci, paired
from sf2.system1.advisor import Advisor
from sf2.system1.system1 import System1, choices, play_round
from sf2.system2 import lessons as L
from sf2.system2 import character_prompt, lesson_prompt
from sf2.system2 import track_record as T
from sf2.system2.lesson_prompt import streak
from sf2.system2.move_coach import MIN_TRIES
from sf2.system2.qwen import chat, json_reply
from sf2.vocab import RANGES

ME = "chunli"
ROOT = os.path.join("rollouts", "qwen_lessons")
BASELINE = 500
PROMPTS = {"views": lesson_prompt, "character": character_prompt,
           "character_fgc": SimpleNamespace(messages=character_prompt.messages_fgc,
                                            parse_claims=character_prompt.parse_claims, VIEWS=character_prompt.VIEWS)}


def decisions(opp: str, lock: str = None) -> List[Dict]:
    """Every decision of hers against ``opp`` in her play data (attacks, walks, blocks); from a lock: its snapshot."""
    if lock:
        return lk.play_rows(lock, ME, opp)
    return [a for d in play_dirs() for a in load_actions(d) if a.get("me") == ME and a.get("opp") == opp]


MOVES = choices(ME)          # what System 1 can pick: advice about anything else cannot be followed


def apply_lock(args) -> None:
    """--lock NAME [--run N]: check the lock, then play from its copies (and with run N's opponent, seed, games)."""
    args.state = None
    if not args.lock:
        return
    bad = lk.verify(args.lock)
    if bad:
        raise SystemExit("lock %s does not hold: %s" % (args.lock, "; ".join(bad)))
    if args.run is not None:
        runs = lk.manifest(args.lock)["runs"]
        if not 0 <= args.run < len(runs):
            raise SystemExit("lock %s has runs 0..%d" % (args.lock, len(runs) - 1))
        for k in ("opp", "seed", "games", "rounds", "history"):
            if k in runs[args.run]:
                setattr(args, k, runs[args.run][k])
    p = lk.paths(args.lock)
    args.model, args.advisor = p["model"], p["advisor"]
    if args.opp:
        args.state = os.path.join(p["states"], "p1_%s_vs_%s.state" % (ME, args.opp))


def load_track(path, opp: str):
    """(this opponent's track record, sha256 of the file); (None, None) without a file (sf2.system2.track_record)."""
    if not path:
        return None, None
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw)["opponents"].get(opp, {}), hashlib.sha256(raw).hexdigest()


def load_book(path, opp: str):
    """(this opponent's verified lines, sha256 of the book file); ([], None) without a book (scripts/book.py)."""
    if not path:
        return [], None
    with open(path, "rb") as f:
        raw = f.read()
    return json.loads(raw)["opponents"].get(opp, {}).get("lines", []), hashlib.sha256(raw).hexdigest()


def book_meta(args) -> Dict:
    """What the run file, the ledger rows and the verdict record about the book ({} without one)."""
    if not getattr(args, "book", None):
        return {}
    return {"book": args.book, "book_sha256": load_book(args.book, args.opp)[1]}


def load_oracle(path):
    """(the lookup table, sha256 of its file); (None, None) without one (sf2.data.value_oracle)."""
    if not path:
        return None, None
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return value_oracle.load(path), digest


def oracle_meta(args) -> Dict:
    """What the run file and the verdict record about the table ({} without one: laya-vision's P(hit) ranking)."""
    path = getattr(args, "oracle", None)
    return {"oracle": path, "oracle_sha256": load_oracle(path)[1]} if path else {}


def advisor_options(args) -> Dict:
    """--shared-text-laya: the one shared server; else nothing (the default, a helper per run)."""
    return {"shared": True} if getattr(args, "shared_text_laya", False) else {}


def job_gb(args) -> float:
    """Memory each arm reserves: with the shared server only the game (+ laya-vision), the server reserves its own."""
    return RUN_JOB_SHARED_GB if getattr(args, "shared_text_laya", False) else MODEL_JOB_GB


def code_commit() -> str:
    """The commit the run's code is at ("" outside git): recorded so a report pairs only runs of the same code."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
    except OSError:
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def play_settings(args, table) -> Dict:
    """What the run file records about how the arm played (scripts/factorial_report.py refuses pairs that differ)."""
    return {"games": args.games, "rounds": args.rounds, "advisor": args.advisor,
            "model": None if table is not None else args.model, "commit": code_commit()}


def run_name(args, stamp: str) -> str:
    """<stamp>_<opp>[_<prompt>][_track][_book][+table]: the ranking is part of the name (the reports pair by it)."""
    return "%s_%s%s" % (stamp, args.opp, ("" if args.prompt == "views" else "_" + args.prompt)
                        + ("_track" if args.track else "") + ("_book" if args.book else "")
                        + ("+table" if getattr(args, "oracle", None) else ""))


def start_registry(args, arm: str) -> L.Registry:
    """The registry the arm starts with: the loop arm, the book's verified lines for this opponent; else nothing."""
    if arm != "loop" or not getattr(args, "book", None):
        return []
    return L.from_book(load_book(args.book, args.opp)[0], args.book)


def ask(opp: str, reg: L.Registry, rows: List[Dict], last: List[Dict], all_rounds: List[Dict],
        last_rounds: List[Dict], refused: List[Dict], stable=None, prompt: str = "views", track=None,
        forward_lessons: bool = False):
    p = PROMPTS[prompt]
    try:
        raw = chat(p.messages(ME, opp, reg, rows, last, all_rounds, last_rounds, MOVES, refused, stable, track=track,
                              forward_lessons=forward_lessons),
                   "lessons_%s" % opp)
        claims, problems = p.parse_claims(json_reply(raw))
    except Exception as e:                   # Qwen down, cut off or not JSON: no claims this game
        return [], ["%s: %s" % (type(e).__name__, e)], None
    return claims, problems, raw


def play_arm(args, arm: str, port: int, out: str) -> int:
    base = decisions(args.opp, args.lock) if args.history else []
    reg: L.Registry = start_registry(args, arm)
    book = book_meta(args)
    unfollowable = () if args.forward_lessons else L.UNFOLLOWABLE
    acts: List[Dict] = []
    played: List[Dict] = []
    refused: List[Dict] = []
    games: List[Dict] = []             # per game: rounds won / lost
    game_hp: List[float] = []          # per game: her mean hp (dealt - taken) per round, for the early stop
    changed: List[bool] = []           # per update: did the registered lessons change
    os.makedirs(out, exist_ok=True)
    track, digest = load_track(args.track, args.opp)
    table = load_oracle(getattr(args, "oracle", None))[0]
    mark_run(out, test=True, arm=arm, opp=args.opp, seed=args.seed, lock=args.lock,
             prompt=args.prompt, track=args.track, track_sha256=digest, forward_lessons=args.forward_lessons,
             **book, **oracle_meta(args), **play_settings(args, table))      # never play data (sf2.eval.logs)
    with Advisor(args.advisor, **advisor_options(args)) as advisor, \
            open_fight(ME, args.opp, port, state=args.state) as (b, state), \
            open_logs(out, ("actions", "rounds", "ledger")) as logs:
        # the table ranks without laya-vision: no model is loaded (System 1 refuses a model with an oracle)
        s1 = System1(None if table is not None else args.model, ME, advisor=advisor, oracle=table)
        s1.advice_on = arm == "loop"
        rng = random.Random(args.seed)

        def learn(game: int, last: List[Dict], last_rounds: List[Dict]) -> None:
            nonlocal reg, refused
            rows = base + acts
            before = {r["line"] for r in reg if r["state"] == "registered"}
            reg = L.review(reg, rows, game, game_hp)
            stable = streak(games, changed)
            claims, problems, raw = ask(args.opp, reg, rows, last, played, last_rounds, refused, stable,
                                        args.prompt, track, args.forward_lessons)
            reg, outcome = L.propose(reg, claims, rows, game, moves=MOVES, unfollowable=unfollowable)
            changed.append({r["line"] for r in reg if r["state"] == "registered"} != before)
            refused = [o for o in outcome if o["state"] == "refused"]
            logs["ledger"].write(json.dumps({"game": game, "prompt": args.prompt, "stable": stable, "claims": claims, "outcome": outcome,
                                             "problems": problems,
                                             "registry": reg, "in_play": L.in_play(reg),
                                             "violations": L.violations(reg, rows),
                                             "reply": raw, **book}) + "\n")
            logs["ledger"].flush()
            print("after game %d: %s | in play %s" % (game, ["%s -> %s" % (o.get("line", o["claim"]), o["state"])
                                                            for o in outcome], L.in_play(reg)), flush=True)

        if arm == "loop" and base:
            learn(-1, base, [])                   # before the first game: from her play data
        for game in range(args.games):
            lines = L.in_play(reg)
            s1.short = {"me": ME, "opp": args.opp, "lessons": [{"text": t} for t in lines]}
            this, this_rounds = [], []
            for r in range(args.rounds):
                i = game * args.rounds + r
                rnd = play_round(b, s1, args.opp, state, rng, None, i)
                where = {"round": i, "game": game, "arm": arm}
                this += [dict(e, **where) for e in rnd.log]
                logs["actions"].write("".join(json.dumps(dict(e, **where)) + "\n" for e in rnd.log))
                logs["rounds"].write(json.dumps(dict(rnd.summary, **where, lines=lines)) + "\n")
                this_rounds.append(rnd.summary)
                for f in logs.values():
                    f.flush()
                print("%s game %d round %d: %s hp %+d" % (arm, game, r, rnd.result,
                                                          rnd.summary["dealt"] - rnd.summary["taken"]), flush=True)
            acts += this
            played += this_rounds
            game_hp.append(sum(r["dealt"] - r["taken"] for r in this_rounds) / len(this_rounds))
            games.append({"won": sum(r["result"] == "win" for r in this_rounds),
                          "lost": sum(r["result"] != "win" for r in this_rounds)})
            if arm == "loop":
                learn(game, this, this_rounds)
    return 0


def populated_cells(rows: List[Dict], need_when: bool = False) -> List[Dict]:
    """Every (move, range, when) cell with a judgeable class: the move MIN_TRIES+ times there, the rest MIN_TRIES+
    (sf2.system2.lessons.condition_evidence). ``need_when``: only cells that name what he is doing."""
    moves = sorted({a["action"] for a in rows})
    whens = tuple(L.WHEN_WORDS) if need_when else (None,) + tuple(L.WHEN_WORDS)
    out = []
    for m in moves:
        for rng in (None,) + RANGES:
            for w in whens:
                ev = L.condition_evidence(rows, {"move": m, "range": rng, "when": w})
                if ev["tries"] >= MIN_TRIES and ev["others"] >= MIN_TRIES:
                    out.append({"move": m, "range": rng, "when": w})
    return out


def random_claims(rows: List[Dict], n: int, seed: int = 0, need_when: bool = False) -> List[Dict]:
    """The chance baseline: ``n`` claims, each a random kind on a random populated cell (``populated_cells``); none
    when no cell is populated. ``need_when``: every claim names what he is doing, like a character lesson."""
    r = random.Random(seed)
    cells = populated_cells(rows, need_when)
    return [dict(r.choice(cells), kind=r.choice(L.KINDS)) for _ in range(n)] if cells else []


def uniform_claims(rows: List[Dict], n: int, seed: int = 0, need_when: bool = False) -> List[Dict]:
    """The old baseline (before 2026-09-29): uniform over every move x range x when, mostly empty cells."""
    r = random.Random(seed)
    moves = sorted({a["action"] for a in rows})
    whens = tuple(L.WHEN_WORDS) if need_when else (None,) + tuple(L.WHEN_WORDS)
    return [{"kind": r.choice(L.KINDS), "move": r.choice(moves), "range": r.choice((None,) + RANGES),
             "when": r.choice(whens)} for _ in range(n)] if moves else []


DUPLICATE = ("already registered", "already testing", "already rejected")     # sf2.system2.lessons._refusal


def _rate(k: int, n: int):
    return k / n if n else None


def holds(c: Dict, rows: List[Dict]) -> bool:
    return L.condition_evidence(rows, c)["cls"] == L.RIGHT[c["kind"]]


def verdict(root: str, opp: str, history: bool, lock: str = None, prompt: str = "views") -> Dict:
    led = read(os.path.join(root, "loop", "ledger.jsonl"), missing_ok=True)
    acts = read(os.path.join(root, "loop", "actions.jsonl"), missing_ok=True)
    rows = (decisions(opp, lock) if history else []) + acts
    valid = [o["claim"] for r in led for o in r["outcome"] if o["state"] != "refused"]
    views = {v: [c for c in valid if c.get("view") == v] for v in PROMPTS[prompt].VIEWS if v != "what_if"}
    need_when = prompt.startswith("character")
    cells = populated_cells(rows, need_when)
    rand = random_claims(rows, BASELINE, need_when=need_when)
    uniform = [c for c in uniform_claims(rows, BASELINE, need_when=need_when)
               if L.condition_evidence(rows, c)["tries"] > 0]
    outcomes = [o for r in led for o in r["outcome"]]
    at_proposal = sum(o["state"] == "registered" for o in outcomes)
    dup = sum(o["state"] == "refused" and str(o.get("why", "")).startswith(DUPLICATE) for o in outcomes)
    loop, none = (read(os.path.join(root, a, "rounds.jsonl"), missing_ok=True) for a in ("loop", "none"))
    d = paired(loop, none) if loop and len(loop) == len(none) else []
    states = {}
    for r in led:
        for o in r["outcome"]:
            states[o["state"]] = states.get(o["state"], 0) + 1
    final = led[-1]["registry"] if led else []
    return {"prompt": prompt, "updates": len(led), "violations": sum(len(r["violations"]) for r in led),
            "proposed": sum(len(r["outcome"]) for r in led), "outcomes": states,
            "qwen_hold_rate": sum(holds(c, rows) for c in valid) / len(valid) if valid else None,
            "qwen_hold_rate_by_view": {v: (sum(holds(c, rows) for c in cs) / len(cs) if cs else None, len(cs))
                                       for v, cs in views.items()},
            "what_if_asked": sum(bool(r.get("stable")) for r in led),
            "what_if": [(o.get("line") or o["claim"], o["state"]) for r in led for o in r["outcome"]
                        if isinstance(o["claim"], dict) and o["claim"].get("view") == "what_if"],
            "kinds_registered": {k: sum(r["state"] == "registered" and r["claim"]["kind"] == k for r in final)
                                 for k in L.KINDS},
            "random_hold_rate": sum(holds(c, rows) for c in rand) / len(rand) if rand else None,
            "random_cells": len(cells),
            "random_hold_rate_uniform": sum(holds(c, rows) for c in uniform) / len(uniform) if uniform else None,
            "valid": len(valid),
            "registered_at_proposal": at_proposal, "registered_at_proposal_share": _rate(at_proposal, len(valid)),
            "refused_duplicate": dup, "refused_duplicate_share": _rate(dup, len(outcomes)),
            "registered_at_end": [r["line"] for r in final if r["state"] == "registered"],
            "verified_at_end": [r["line"] for r in final if r["state"] == "verified"],
            "retired": [r["line"] for r in final if r["state"] == "retired"],
            "won": {"loop": sum(r["result"] == "win" for r in loop), "none": sum(r["result"] == "win" for r in none)},
            "hp_vs_none": ci(d) if d else None,
            "taken_vs_none": ci([b["taken"] - a["taken"] for a, b in zip(loop, none)]) if d else None}


def export(out: str = os.path.join("lessons", "chunli.json")) -> int:
    """Every lesson registered at the end of a run, per opponent, with the runs that registered it and its evidence
    there (a lesson registered in every run of that opponent is marked "all_runs")."""
    by_opp: Dict[str, Dict] = {}
    for root in sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d))):
        led = read(os.path.join(ROOT, root, "loop", "ledger.jsonl"), missing_ok=True)
        v = os.path.join(ROOT, root, "verdict.json")
        if not led or not os.path.exists(v) or json.load(open(v)).get("updates", 0) < 10:
            continue                              # smoke runs and unfinished runs are left out
        opp = root.split("_", 1)[1]
        runs = by_opp.setdefault(opp, {"runs": [], "lessons": {}})
        runs["runs"].append(root)
        for r in led[-1]["registry"]:
            if r["state"] == "registered":
                e = runs["lessons"].setdefault(r["line"], {"claim": r["claim"], "runs": {}})
                e["runs"][root] = {k: v for k, v in r["evidence"].items()      # absolute (old runs) or relative
                                   if k in ("tries", "net", "base", "diff", "lo", "hi", "total")}
    doc = {"me": ME, "made_by": "scripts/qwen_lessons.py --export: Qwen proposed, code verified (sf2.system2.lessons)",
           "opponents": {o: {"runs": d["runs"], "lessons": [dict(line=line, all_runs=len(e["runs"]) == len(d["runs"]),
                                                                  **e) for line, e in sorted(d["lessons"].items())]}
                         for o, d in sorted(by_opp.items())}}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(doc, f, indent=1)
    for o, d in doc["opponents"].items():
        print("%-8s %d runs, %d lessons (%d in every run)" % (o, len(d["runs"]), len(d["lessons"]),
                                                             sum(x["all_runs"] for x in d["lessons"])))
    print("saved", out)
    return 0


def log_dir(root: str) -> str:
    """The arms' console logs, one folder per run (two runs against one opponent must not share a file)."""
    return os.path.join("logs", "qwen_lessons", os.path.basename(root))


def arm_cmds(args, root: str) -> List:
    """The two arms' command lines (loop, none): the same options, each its own port and folder."""
    common = (["--opp", args.opp, "--games", str(args.games), "--rounds", str(args.rounds), "--seed", str(args.seed),
               "--model", args.model, "--advisor", args.advisor] + ([] if args.history else ["--no-history"])
              + (["--lock", args.lock] if args.lock else []) + ["--prompt", args.prompt]
              + (["--track", args.track] if args.track else [])
              + (["--book", args.book] if args.book else [])
              + (["--oracle", args.oracle] if getattr(args, "oracle", None) else [])
              + (["--shared-text-laya"] if getattr(args, "shared_text_laya", False) else [])
              + (["--forward-lessons"] if args.forward_lessons else []))
    return [((args.opp, arm), [sys.executable, os.path.abspath(__file__)] + common
             + ["--one", arm, str(args.base_port + i), os.path.join(root, arm)])
            for i, arm in enumerate(("loop", "none"))]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opp")
    ap.add_argument("--export", action="store_true", help="write lessons/chunli.json from every finished run")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3, help="rounds per game (Qwen proposes after each game)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-history", dest="history", action="store_false")
    ap.add_argument("--base-port", type=int, default=PORTS["qwen_moves"][0], help="two ports: loop and none")
    ap.add_argument("--model", default=LAYA_VISION)
    ap.add_argument("--advisor", default=TEXT_LAYA)
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="views", help="what Qwen is asked after each game")
    ap.add_argument("--track", help="a track record file (scripts/track_record.py): shown to Qwen (never a reason to "
                                    "refuse a lesson)")
    ap.add_argument("--book", help="a book of verified players' tips (scripts/book.py build): the loop starts from "
                                   "this opponent's lines (default: none)")
    ap.add_argument("--oracle", help="System 1 ranks by this lookup table (sf2.data.value_oracle, e.g. "
                                     "lessons/value_oracle_v1.json) instead of laya-vision's P(hit)")
    ap.add_argument("--shared-text-laya", action="store_true",
                    help="text laya from the one shared server; each arm reserves config.RUN_JOB_SHARED_GB")
    ap.add_argument("--forward-lessons", action="store_true",
                    help="offer and accept lessons naming forward again (the loop before 2026-09-30, for replays)")
    ap.add_argument("--lock", help="play from this lock's copies only (sf2.eval.lock)")
    ap.add_argument("--run", type=int, help="with --lock: repeat the lock's run N (its opponent, seed, games)")
    ap.add_argument("--one", nargs=3, metavar=("ARM", "PORT", "OUT"), help=argparse.SUPPRESS)
    args = ap.parse_args()
    exit_on_sigterm()
    if args.export:
        return export()
    apply_lock(args)
    if not args.opp:
        ap.error("--opp is required")
    if args.one:
        return play_arm(args, args.one[0], int(args.one[1]), args.one[2])
    args.seed = int(time.time()) % 100000 if args.seed is None else args.seed
    root = os.path.join(os.path.join("rollouts", "locked", args.lock) if args.lock else ROOT,
                        run_name(args, time.strftime("%Y%m%d-%H%M%S")))
    cmds = arm_cmds(args, root)
    print("Chun-Li vs %s: %d games x %d rounds, seed %d; %s/" % (
        args.opp, args.games, args.rounds, args.seed, log_dir(root)), flush=True)
    failed = fan_out(cmds, log_dir(root), job_gb=job_gb(args))
    v = dict(verdict(root, args.opp, args.history, args.lock, args.prompt), seed=args.seed, lock=args.lock, track=args.track,
             track_sha256=load_track(args.track, args.opp)[1], forward_lessons=args.forward_lessons,
             failed_jobs=[list(k) for k in failed], **book_meta(args), **oracle_meta(args))
    with open(os.path.join(root, "verdict.json"), "w") as f:
        json.dump(v, f, indent=1)
    print(json.dumps(v, indent=1))
    print("saved", root)
    return 1 if failed or v["violations"] else 0


if __name__ == "__main__":
    sys.exit(main())
