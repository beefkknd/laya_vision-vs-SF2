"""The learning loop: System 1 plays arcade mode from power-on in a Mesen window you can watch, System 2 (Qwen on omlx)
learns from the game log, until Ctrl-C.

    python scripts/learn_loop.py                      # Chun-Li, window at normal speed
    python scripts/learn_loop.py --char ryu

From power-on: GAME START, pick the character, then the arcade ladder as it comes. A GAME is one opponent until it is
decided: win the match (best of 3 rounds) and the next opponent comes; lose it and the continue starts a new game
against the same opponent. System 2 steps in:
    new opponent  -> the short memory for him (memory/short/<me>_vs_<opp>.json, goes into laya's prompt); one kept
                     from an earlier session is reused at once if it still meets today's rules (system2.fits_laya)
    lost round    -> refreshes the short memory, so the next round plays with what just happened
    end of a game -> rewrites the playbook (memory/playbook/<me>.json) from every round so far, then refreshes the
                     short memory
It learns from the earlier logs of this character too (rollouts/games_v*/<me>, rollouts/learn/<me>).

Session log: rollouts/learn/<me>/<session>/{games.jsonl (one line per game), rounds.jsonl (per round), actions.jsonl
(per action, with its game and round), images/}; console copy in logs/learn_<me>.log; every Qwen prompt and reply in
logs/system2/; every memory version in logs/system2/memory/.
"""
import argparse
import glob
import json
import os
import random
import shutil
import sys
import time
from typing import Dict, List, Tuple

import _path  # noqa: F401
from sf2.boot import CHARACTERS, next_fight, start_arcade
from sf2.headless import launch_argv, window_argv
from sf2.memory import load, playbook_path, short_path
from sf2.mesen import MesenBridge
from sf2.system1 import System1, play_round
from sf2.system2 import fits_laya, populate, review
from sf2.vs import NAMES, VARS
from sf2.vs_sweep import actions

LOG = None


def say(msg: str) -> None:
    line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def history(me: str) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
    """Every earlier log of ``me``, by opponent: {opp: (actions, rounds)}; each action tagged with its log dir. Older
    logs (games_v*) have one line per round in games.jsonl; learning sessions keep rounds in rounds.jsonl."""
    by_opp: Dict[str, Tuple[List[Dict], List[Dict]]] = {}
    dirs = sorted(glob.glob("rollouts/games_v*/%s" % me)) + sorted(glob.glob("rollouts/learn/%s/*" % me))
    for d in dirs:
        if not os.path.exists(os.path.join(d, "actions.jsonl")):
            continue
        tag = os.path.relpath(d, "rollouts")
        acts = [dict(json.loads(x), log=tag) for x in open(os.path.join(d, "actions.jsonl"))]
        rfile = os.path.join(d, "rounds.jsonl")
        rounds = [json.loads(x) for x in open(rfile if os.path.exists(rfile) else os.path.join(d, "games.jsonl"))]
        opps = {a["opp"] for a in acts}
        for o in opps:
            a0, r0 = by_opp.setdefault(o, ([], []))
            a0 += [a for a in acts if a["opp"] == o]
            r0 += [r for r in rounds if r.get("opp", next(iter(opps))) == o]
    return by_opp


def save_memory(path: str, mem: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(mem, f, indent=1)
    keep = os.path.join("logs", "system2", "memory", time.strftime("%Y%m%d-%H%M%S_") + os.path.basename(path))
    os.makedirs(os.path.dirname(keep), exist_ok=True)
    shutil.copy(path, keep)


def show(title: str, lessons: List[Dict]) -> None:
    say(title)
    for x in lessons:
        say("    %-15s %s  [%s %d/%d]" % (x["kind"], x["text"], x.get("claim", "-"), x["evidence"]["count"],
                                          x["evidence"]["tries"]))


def run_system2_short(me: str, opp: str, by_opp) -> Dict:
    say("System 2: writing the short memory vs %s ..." % opp)
    t = time.time()
    everything = [a for acts, _ in by_opp.values() for a in acts]
    lessons, rep = populate(me, opp, load(playbook_path(me), list(actions(me))), by_opp.get(opp, ([], [])), everything)
    if lessons is None:
        say("System 2: no usable short memory (%s); keeping the previous one" % rep["attempts"])
    else:
        save_memory(short_path(me, opp), {"me": me, "opp": opp, "source": sorted({a.get("log") for a in everything}),
                                           "lessons": lessons})
        show("System 2 (%.0f s): short memory vs %s - %s" % (time.time() - t, opp, rep.get("notes")), lessons)
        for att in rep["attempts"]:
            for r in att.get("rejected", []):
                say("    rejected: %s (%s)" % (r["lesson"].get("text"), r["why"]))
    return load(short_path(me, opp), list(actions(me)))


def run_system2_review(me: str, by_opp) -> None:
    say("System 2: reviewing %d rounds, rewriting the playbook ..." % sum(len(r) for _, r in by_opp.values()))
    t = time.time()
    lessons, rep = review(me, load(playbook_path(me), list(actions(me))), by_opp)
    if lessons is None:
        say("System 2: no usable playbook (%s); keeping the previous one" % rep["attempts"])
        return
    save_memory(playbook_path(me), {"me": me, "source": sorted({a.get("log") for acts, _ in by_opp.values()
                                                                for a in acts}), "lessons": lessons})
    show("System 2 (%.0f s): playbook - %s" % (time.time() - t, rep.get("notes")), lessons)


def short_for(me: str, opp: str, by_opp) -> Dict:
    """The short memory to play ``opp`` with: a kept one if it meets today's rules, else a fresh one from System 2."""
    kept = load(short_path(me, opp), list(actions(me)))
    if fits_laya(kept):
        show("new opponent: %s - short memory kept from before" % opp, kept["lessons"])
        return kept
    say("new opponent: %s%s" % (opp, " (the kept short memory breaks today's rules: rewriting it)" if kept else ""))
    return run_system2_short(me, opp, by_opp)


def main() -> int:
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--char", default="chunli")
    ap.add_argument("--model", default="runs/all8/best")
    ap.add_argument("--headless", action="store_true", help="no window, full speed")
    ap.add_argument("--port", type=int, default=47990)
    args = ap.parse_args()
    me = args.char
    session = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join("rollouts", "learn", me, session)
    tag = os.path.relpath(out, "rollouts")
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    LOG = open(os.path.join("logs", "learn_%s.log" % me), "a")
    by_opp = history(me)
    say("session %s: %s, System 1 %s, earlier rounds: %s" % (session, me, args.model,
                                                             {o: len(r) for o, (_, r) in by_opp.items()}))
    s1 = System1(args.model, me)
    argv = launch_argv(args.port, None) if args.headless else window_argv(args.port, None)
    b = MesenBridge(args.port, launch=argv)
    files = {k: open(os.path.join(out, k + ".jsonl"), "a") for k in ("actions", "rounds", "games")}
    rng, n, game, opp = random.Random(0), 0, 0, None
    score, rounds = [0, 0], []           # this game's round wins (me, him) and its round summaries
    try:
        b.set_capture("raw")
        say("power-on: GAME START, picking %s ..." % me)
        start_arcade(b, me)
        while True:
            b.set_vars(VARS)
            row = dict(zip(NAMES, b.run([[]]).rams[-1]))
            now = CHARACTERS.get(row["p2_char"], "id%d" % row["p2_char"])
            if now != opp:
                opp = now
                s1.short = short_for(me, opp, by_opp)
            rnd = play_round(b, s1, opp, None, rng, os.path.join(out, "images"), n)
            where = {"game": game, "round": len(rounds), "opp": opp, "log": tag}
            summary = dict(rnd.summary, **where)
            files["actions"].write("".join(json.dumps(dict(e, **where)) + "\n" for e in rnd.log))
            files["rounds"].write(json.dumps(summary) + "\n")
            a0, r0 = by_opp.setdefault(opp, ([], []))
            a0 += [dict(e, **where) for e in rnd.log]
            r0.append(summary)
            rounds.append(summary)
            score[0] += rnd.result == "win"
            score[1] += rnd.result == "loss"
            say("game %d round %d vs %s: %s (%d-%d)  dealt %d taken %d  %d actions, attacks %s" % (
                game, len(rounds) - 1, opp, rnd.result, score[0], score[1], rnd.summary["dealt"],
                rnd.summary["taken"], len(rnd.log), rnd.summary["outcomes"]))
            if max(score) >= 2 or len(rounds) >= 4:        # the match is decided: the game is over
                result = "win" if score[0] > score[1] else "loss"
                files["games"].write(json.dumps({"game": game, "opp": opp, "result": result, "score": score,
                                                 "rounds": [r["result"] for r in rounds], "log": tag}) + "\n")
                say("GAME %d vs %s: %s %d-%d" % (game, opp, result.upper(), score[0], score[1]))
                run_system2_review(me, by_opp)
                s1.short = run_system2_short(me, opp, by_opp)
                game, score, rounds = game + 1, [0, 0], []
            elif rnd.result == "loss":
                s1.short = run_system2_short(me, opp, by_opp)
            for f in files.values():
                f.flush()
            n += 1
            next_fight(b, me=me)
    except KeyboardInterrupt:
        say("Ctrl-C: stopping after %d games (%d rounds)" % (game, n))
    finally:
        b.close()
        for f in files.values():
            f.close()
        say("session %s saved in %s" % (session, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
