"""The learning loop: System 1 plays arcade mode from power-on in a Mesen window you can watch, System 2 (Qwen on omlx)
learns from the game log, until Ctrl-C.

    python scripts/learn_loop.py                      # Chun-Li, window at 1.5x speed
    python scripts/learn_loop.py --speed 100          # the real game speed
    python scripts/learn_loop.py --char ryu
    python scripts/learn_loop.py --minutes 10         # stop by itself after 10 minutes (between rounds)
    python scripts/learn_loop.py --advisor off        # the old System 1: laya-vision's threshold rule only
    python scripts/learn_loop.py --ab-advice          # A/B: every other game without the short memory's advice

From power-on: GAME START, pick the character, then the arcade ladder as it comes. A GAME is one opponent until it is
decided: win the match (best of 3 rounds) and the next opponent comes; lose it and the continue starts a new game
against the same opponent. System 2 steps in:
System 1: laya-vision rates every move from the screen; text laya (--advisor) picks from its best-rated moves and the
moves the short memory names, following the memory (sf2.system1.advisor). The actions log keeps what text laya read and
picked, and whether that matches the label rule (follows_rule).
    new opponent  -> the short memory for him (memory/short/<me>_vs_<opp>.json, goes into laya's prompt); one kept
                     from an earlier session is reused at once if it still meets today's rules (system2.fits_laya)
System 2 runs in the background (class System2): the game never waits for Qwen; a new memory is used from the
round after it is ready.
    lost round    -> REVISES the short memory from that round: Qwen sees the memory it was played with, what System 1
                     did with each lesson (used / ignored, how it went) and the round's digest
    end of a game -> rewrites the playbook (memory/playbook/<me>.json) from every round so far, then revises the short
                     memory from the whole game the same way
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
import re
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import Future
from typing import Dict, List, Optional, Tuple

import _path  # noqa: F401
from sf2.config import LAYA_VISION, LIVE, MEMORY_RUNS, PORTS, TEXT_LAYA
from sf2.emu.boot import next_fight, start_arcade
from sf2.eval.logs import is_test, load_actions, load_rounds, mark_run
from sf2.emu.headless import KeepMesenSettings, launch_argv, window_argv
from sf2.demo import demo_cheat
from sf2.system1.advisor import Advisor
from sf2.system2.memory import OutsideWatch, load, playbook_path, save, short_path
from sf2.system2.memory_churn import diff as diff_memory
from sf2.emu.mesen import MesenBridge
from sf2.system1.system1 import System1, play_round
from sf2.system2.system2 import fits_laya, populate, review
from sf2.vocab import CHARACTERS
from sf2.emu.vs import NAMES, VARS
from sf2.data.vs_sweep import actions

LOG = None
MEM_ROOT = "memory"        # memory_runs/<name> with --fresh


SAY_LOCK = threading.Lock()


def _stop(signum, frame) -> None:
    raise KeyboardInterrupt


def say(msg: str) -> None:
    line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
    with SAY_LOCK:                      # System 2 speaks from its own thread
        print(line, flush=True)
        LOG.write(line + "\n")
        LOG.flush()


def history(me: str) -> Dict[str, Tuple[List[Dict], List[Dict]]]:
    """Every earlier log of ``me``, by opponent: {opp: (actions, rounds)}; each action tagged with its log dir. Older
    logs (games_v*) have one line per round in games.jsonl; learning sessions keep rounds in rounds.jsonl."""
    by_opp: Dict[str, Tuple[List[Dict], List[Dict]]] = {}
    dirs = sorted(glob.glob("rollouts/games_v*/%s" % me)) + sorted(glob.glob("rollouts/learn/%s/*" % me))
    for d in dirs:
        if not os.path.exists(os.path.join(d, "actions.jsonl")) or is_test(d):
            continue            # --fresh demo sessions are not play data (sf2.eval.logs)
        acts, rounds = load_actions(d), load_rounds(d)
        opps = {a["opp"] for a in acts}
        for o in opps:
            a0, r0 = by_opp.setdefault(o, ([], []))
            a0 += [a for a in acts if a["opp"] == o]
            r0 += [r for r in rounds if r.get("opp", next(iter(opps))) == o]
    return by_opp


def save_memory(path: str, mem: Dict) -> None:
    """Write System 2's new version, saying how much it changed from the one it replaces (sf2.system2.memory_churn)."""
    if os.path.exists(path):
        change = diff_memory(json.load(open(path)), mem, list(actions(mem["me"])) + ["forward"])
        say("    change vs the previous %s: %s" % ("playbook" if "opp" not in mem else "short memory", change.line()))
        for old, new in change.flipped:
            say("        FLIP: %r -> %r" % (old, new))
    save(path, mem)
    keep = os.path.join("logs", "system2", "memory", time.strftime("%Y%m%d-%H%M%S_") + os.path.basename(path))
    os.makedirs(os.path.dirname(keep), exist_ok=True)
    shutil.copy(path, keep)


def show(title: str, lessons: List[Dict]) -> None:
    say(title)
    for x in lessons:
        say("    %-15s %s  [%s %d/%d]" % (x["kind"], x["text"], x.get("claim", "-"), x["evidence"]["count"],
                                          x["evidence"]["tries"]))


def run_system2_short(me: str, opp: str, by_opp, recent: Optional[List[Dict]] = None, what: str = "the last round",
                      current: Optional[Dict] = None) -> Dict:
    """Write (or, with ``recent`` and ``current``, revise from what just happened) the short memory vs ``opp``."""
    say("System 2: %s the short memory vs %s ..." % ("revising" if recent else "writing", opp))
    t = time.time()
    everything = [a for acts, _ in by_opp.values() for a in acts]
    lessons, rep = populate(me, opp, load(playbook_path(me, MEM_ROOT), list(actions(me))), by_opp.get(opp, ([], [])), everything,
                            recent=recent, current=current, what=what)
    if lessons is None:
        say("System 2: no usable short memory (%s); keeping the previous one" % rep["attempts"])
    else:
        save_memory(short_path(me, opp, MEM_ROOT), {"me": me, "opp": opp, "source": sorted({a.get("log") for a in everything}),
                                           "lessons": lessons})
        show("System 2 (%.0f s): short memory vs %s - %s" % (time.time() - t, opp, rep.get("notes")), lessons)
        for att in rep["attempts"]:
            for r in att.get("rejected", []):
                say("    rejected: %s (%s)" % (r["lesson"].get("text"), r["why"]))
    return load(short_path(me, opp, MEM_ROOT), list(actions(me)))


def run_system2_review(me: str, by_opp) -> None:
    say("System 2: reviewing %d rounds, rewriting the playbook ..." % sum(len(r) for _, r in by_opp.values()))
    t = time.time()
    lessons, rep = review(me, load(playbook_path(me, MEM_ROOT), list(actions(me))), by_opp)
    if lessons is None:
        say("System 2: no usable playbook (%s); keeping the previous one" % rep["attempts"])
        return
    save_memory(playbook_path(me, MEM_ROOT), {"me": me, "source": sorted({a.get("log") for acts, _ in by_opp.values()
                                                                for a in acts}), "lessons": lessons})
    show("System 2 (%.0f s): playbook - %s" % (time.time() - t, rep.get("notes")), lessons)


def snapshot(by_opp) -> Dict:
    """A copy System 2 can read while the game goes on adding to ``by_opp``."""
    return {o: (list(a), list(r)) for o, (a, r) in by_opp.items()}


class System2:
    """System 2 in the background: Chun-Li keeps playing while Qwen reviews. One job at a time; while one runs,
    only the newest request waits (an end-of-game job, which also rewrites the playbook, is not replaced by a
    lost-round one). A finished job's short memory is picked up by ``poll`` before the next round."""

    def __init__(self, me: str):
        self.me = me
        self.future, self.waiting = None, None

    def _start(self, job) -> Future:
        """Run a job on a daemon thread: stopping the loop never waits for a Qwen call to return."""
        fut: Future = Future()

        def run():
            try:
                fut.set_result(self._job(*job))
            except BaseException as e:          # reported by poll(); never kills the game
                fut.set_exception(e)
        threading.Thread(target=run, name="system2", daemon=True).start()
        return fut

    def _job(self, kind: str, opp: str, by_opp, recent, current, what) -> Tuple[str, Dict, Optional[float]]:
        if kind == "game":
            run_system2_review(self.me, by_opp)
        mem = run_system2_short(self.me, opp, by_opp, recent=recent, what=what, current=current)
        path = short_path(self.me, opp, MEM_ROOT)          # stamped as System 2 left it (sf2.memory.OutsideWatch)
        return opp, mem, os.path.getmtime(path) if os.path.exists(path) else None

    def ask(self, kind: str, opp: str, by_opp, recent: List[Dict], current: Optional[Dict], what: str) -> None:
        job = (kind, opp, snapshot(by_opp), list(recent), current, what)
        if self.future is None or self.future.done():
            self.future = self._start(job)
        elif self.waiting is None or self.waiting[0] != "game" or kind == "game":
            if self.waiting:
                say("System 2 busy: %s review replaced by the newer %s review" % (self.waiting[0], kind))
            self.waiting = job

    def busy(self) -> bool:
        """A review is running or waiting (so a memory file may be about to change under System 2's own hand)."""
        return self.waiting is not None or (self.future is not None and not self.future.done())

    def poll(self) -> Optional[Tuple[str, Dict, Optional[float]]]:
        """(opp, short memory, its file's stamp) of a finished job, or None; starts the waiting job."""
        if self.future is None or not self.future.done():
            return None
        try:
            done = self.future.result()
        except Exception as e:                       # System 2 failing must not stop the game
            say("System 2 failed: %s: %s" % (type(e).__name__, e))
            done = None
        self.future = None
        if self.waiting:
            self.future, self.waiting = self._start(self.waiting), None
        return done

    def close(self, wait: bool) -> None:
        """``wait``: let a running review finish (a timed stop); on Ctrl-C / kill it is abandoned (its memory file
        is only written at the end of the review, so nothing is left half-written)."""
        if self.future is not None and not self.future.done():
            if wait:
                say("waiting for System 2 to finish its review ...")
                try:
                    self.future.result(timeout=300)
                except Exception as e:
                    say("System 2's last review failed: %s: %s" % (type(e).__name__, e))
            else:
                say("stopping: System 2's running review is abandoned")
        self.future, self.waiting = None, None


def short_for(me: str, opp: str, by_opp) -> Dict:
    """The short memory to play ``opp`` with: a kept one if it meets today's rules, else a fresh one from System 2."""
    kept = load(short_path(me, opp, MEM_ROOT), list(actions(me)))
    if fits_laya(kept):
        show("new opponent: %s - short memory kept from before" % opp, kept["lessons"])
        return kept
    say("new opponent: %s%s" % (opp, " (the kept short memory breaks today's rules: rewriting it)" if kept else ""))
    return run_system2_short(me, opp, by_opp)


def main() -> int:
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--char", default="chunli")
    ap.add_argument("--model", default=LAYA_VISION)
    ap.add_argument("--headless", action="store_true", help="no window, full speed")
    ap.add_argument("--speed", type=int, default=150, help="window speed in percent (100 = the real game)")
    ap.add_argument("--port", type=int, default=PORTS["learn"][0])
    ap.add_argument("--advisor", default=TEXT_LAYA,
                    help="text laya checkpoint that turns the short memory into the move ('off': laya-vision's "
                         "threshold rule, the memory in its prompt)")
    ap.add_argument("--minutes", type=float, default=0, help="stop after this long (0: until Ctrl-C)")
    ap.add_argument("--games", type=int, default=0, help="stop after this many games (matches); 0: no limit")
    ap.add_argument("--quiet", action="store_true", help="do not print every decision to the console")
    ap.add_argument("--fresh", default=None, metavar="NAME",
                    help="start from a blank memory: its own memory folder memory_runs/NAME and no earlier logs "
                         "(memory/ is not touched); for demos and learning-from-zero tests")
    ap.add_argument("--seed", default=None, metavar="DIR",
                    help="with --fresh: start from this seed memory (scripts/seed_memory.py) instead of blank")
    ap.add_argument("--live", default=os.path.join(LIVE, "decision.json"),
                    help="the latest decision, for scripts/brain_panel.py ('' to turn off)")
    ap.add_argument("--ab-advice", action="store_true",
                    help="A/B test: odd games are played with 'Advice: none' (text laya still picks); logged as advice")
    args = ap.parse_args()
    if args.fresh and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.fresh):
        raise SystemExit("--fresh %r: use letters, digits, - and _ only (it names a folder in memory_runs/)" % args.fresh)
    signal.signal(signal.SIGTERM, _stop)     # `kill` stops it like Ctrl-C: Mesen, text laya and settings cleaned up
    me = args.char
    session = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join("rollouts", "learn", me, session)
    tag = os.path.relpath(out, "rollouts")
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    LOG = open(os.path.join("logs", "learn_%s.log" % me), "a")
    global MEM_ROOT
    MEM_ROOT = os.path.join(MEMORY_RUNS, args.fresh) if args.fresh else "memory"
    if args.fresh and os.path.exists(MEM_ROOT):
        raise SystemExit("%s exists: pick a new --fresh name (a blank start must be blank)" % MEM_ROOT)
    if args.seed:
        if not args.fresh:
            raise SystemExit("--seed needs --fresh (a seed never overwrites memory/)")
        shutil.copytree(args.seed, MEM_ROOT)
    mark_run(out, memory=MEM_ROOT, fresh=args.fresh, seed=args.seed)
    by_opp = {} if args.fresh else history(me)
    pushed = demo_cheat.apply_pending(MEM_ROOT, me, args.seed or demo_cheat.SEED) if args.fresh else []
    if args.live:                               # the brain panel: a new session, nothing left from the last one
        os.makedirs(os.path.dirname(args.live), exist_ok=True)
        if os.path.exists(args.live):
            os.remove(args.live)
        with open(os.path.join(os.path.dirname(args.live), "session.json"), "w") as f:
            json.dump({"session": out, "memory": MEM_ROOT, "me": me, "log": os.path.join("logs", "learn_%s.log" % me),
                       "log_offset": os.path.getsize(os.path.join("logs", "learn_%s.log" % me)),
                       "seed": args.seed, "started": time.time(), "running": True, "pid": os.getpid()}, f)
    for line in pushed:                         # the brain panel's buttons pressed before this game
        say(line)
    say("session %s: %s, System 1 %s, earlier rounds: %s" % (session, me, args.model,
                                                             {o: len(r) for o, (_, r) in by_opp.items()}))
    deadline = time.time() + 60 * args.minutes if args.minutes else None
    s2 = System2(me)
    argv = launch_argv(args.port, None) if args.headless else window_argv(args.port, None, speed=args.speed)
    advisor, b, keep, files, timed_out = None, None, None, {}, False
    rng, n, game, opp = random.Random(0), 0, 0, None
    outside = OutsideWatch()                         # the brain panel's buttons change a short memory from outside
    score, rounds, game_acts = [0, 0], [], []    # this game's round wins (me, him), round summaries, actions
    try:                                         # everything started from here on is cleaned up in `finally`
        advisor = None if args.advisor == "off" else Advisor(args.advisor)
        say("System 1 picks with %s" % ("text laya %s + short memory" % args.advisor if advisor else
                                        "laya-vision's threshold rule"))
        s1 = System1(args.model, me, advisor=advisor)
        keep = KeepMesenSettings().__enter__()   # the window saves its overrides on exit: put the settings back after
        b = MesenBridge(args.port, launch=argv)
        files = {k: open(os.path.join(out, k + ".jsonl"), "a") for k in ("actions", "rounds", "games")}
        b.set_capture("raw")
        say("power-on: GAME START, picking %s ..." % me)
        start_arcade(b, me)
        while True:
            b.set_vars(VARS)
            row = dict(zip(NAMES, b.run([[]]).rams[-1]))
            now = CHARACTERS.get(row["p2_char"], "id%d" % row["p2_char"])
            done = s2.poll()
            if now != opp:
                opp = now
                s1.short = short_for(me, opp, by_opp)
                outside.seen(opp, short_path(me, opp, MEM_ROOT))
            elif done and done[0] == opp:
                s1.short = done[1]
                outside.seen(opp, short_path(me, opp, MEM_ROOT), stamp=done[2])
                say("System 2's new short memory vs %s is in play from this round" % opp)
            else:
                got = outside.check(opp, short_path(me, opp, MEM_ROOT), s2.busy(), list(actions(me)))
                if got:
                    if got[0] is not OutsideWatch.KEEP:
                        s1.short = got[0]
                    say(got[1])
            s1.advice_on = not (args.ab_advice and game % 2 == 1)
            rnd = play_round(b, s1, opp, None, rng, os.path.join(out, "images"), n, live_path=args.live or None,
                             echo=not args.quiet)
            where = {"game": game, "round": len(rounds), "opp": opp, "log": tag,
                     "advice": "on" if s1.advice_on else "off"}
            summary = dict(rnd.summary, **where)
            files["actions"].write("".join(json.dumps(dict(e, **where)) + "\n" for e in rnd.log))
            files["rounds"].write(json.dumps(summary) + "\n")
            a0, r0 = by_opp.setdefault(opp, ([], []))
            a0 += [dict(e, **where) for e in rnd.log]
            r0.append(summary)
            rounds.append(summary)
            game_acts += rnd.log
            score[0] += rnd.result == "win"
            score[1] += rnd.result == "loss"
            say("game %d round %d vs %s: %s (%d-%d)  dealt %d taken %d  %d actions, attacks %s" % (
                game, len(rounds) - 1, opp, rnd.result, score[0], score[1], rnd.summary["dealt"],
                rnd.summary["taken"], len(rnd.log), rnd.summary["outcomes"]))
            if max(score) >= 2 or len(rounds) >= 4:        # the match is decided: the game is over
                result = "win" if score[0] > score[1] else "loss"
                files["games"].write(json.dumps({"game": game, "opp": opp, "result": result, "score": score,
                                                 "rounds": [r["result"] for r in rounds], "log": tag,
                                                 "advice": "on" if s1.advice_on else "off"}) + "\n")
                say("GAME %d vs %s: %s %d-%d  (advice %s)" % (game, opp, result.upper(), score[0], score[1],
                                                             "on" if s1.advice_on else "off"))
                s2.ask("game", opp, by_opp, game_acts, s1.short, "the last game")
                game, score, rounds, game_acts = game + 1, [0, 0], [], []
            elif rnd.result == "loss":
                s2.ask("round", opp, by_opp, rnd.log, s1.short, "the last round")
            for f in files.values():
                f.flush()
            n += 1
            if args.games and game >= args.games:
                say("%d game%s played: stopping (after System 2's review of it)" % (game, "" if game == 1 else "s"))
                timed_out = True                  # let the running review (the playbook) finish on screen
                break
            if deadline and time.time() >= deadline:
                say("%.0f minutes up: stopping after %d games (%d rounds)" % (args.minutes, game, n))
                timed_out = True
                break
            next_fight(b, me=me)
    except KeyboardInterrupt:
        say("Ctrl-C: stopping after %d games (%d rounds)" % (game, n))
    finally:                                     # each step runs even if an earlier one fails
        for step in ((lambda: b.close()) if b else None, lambda: s2.close(wait=timed_out),
                     (lambda: advisor.close()) if advisor else None,
                     (lambda: keep.__exit__(None, None, None)) if keep else None,
                     lambda: [f.close() for f in files.values()]):
            if step is None:
                continue
            try:
                step()
            except Exception as e:
                say("cleanup step failed: %s: %s" % (type(e).__name__, e))
        if args.live:                           # the panel's buttons stop working once the run is over
            path = os.path.join(os.path.dirname(args.live), "session.json")
            try:
                done = dict(json.load(open(path)), running=False)
                with open(path, "w") as f:
                    json.dump(done, f)
            except (OSError, ValueError) as e:
                say("could not mark the session finished for the panel: %s" % e)
        say("session %s saved in %s" % (session, out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
