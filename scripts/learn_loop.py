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
import json
import os
import random
import re
import shutil
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import _path  # noqa: F401
from sf2.config import LAYA_VISION, LIVE, MEMORY_RUNS, PORTS, TEXT_LAYA
from sf2.demo import demo_cheat
from sf2.emu.boot import next_fight, start_arcade
from sf2.emu.headless import KeepMesenSettings, launch_argv, window_argv
from sf2.emu.mesen import MesenBridge
from sf2.emu.vs import NAMES, VARS
from sf2.eval.logs import history, mark_run
from sf2.system1.advisor import Advisor
from sf2.system1.system1 import System1, play_round
from sf2.system2.async_runner import Reviewer, Speaker, System2
from sf2.system2.memory import OutsideWatch
from sf2.vocab import CHARACTERS

FRESH_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
MATCH_WON = 2          # rounds to win a match
MAX_ROUNDS = 4         # a match is over after this many rounds whatever the score (draws)


def _stop(signum, frame) -> None:
    raise KeyboardInterrupt


def parse_args() -> argparse.Namespace:
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
    if args.fresh and not FRESH_NAME.fullmatch(args.fresh):
        raise SystemExit("--fresh %r: use letters, digits, - and _ only (it names a folder in memory_runs/)" % args.fresh)
    if args.seed and not args.fresh:
        raise SystemExit("--seed needs --fresh (a seed never overwrites memory/)")
    return args


@dataclass
class Session:
    """One run of the loop: where it logs, which memory it writes, what it learns from."""
    me: str
    name: str
    out: str
    tag: str
    mem_root: str
    log_path: str
    log: object
    say: Speaker
    by_opp: Dict = field(default_factory=dict)


def open_session(args) -> Session:
    """Folders, the memory root (a --fresh one is created, from --seed if given), the run's marker, the earlier logs
    to learn from, and the brain panel's view of the session."""
    me, name = args.char, time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join("rollouts", "learn", me, name)
    mem_root = os.path.join(MEMORY_RUNS, args.fresh) if args.fresh else "memory"
    if args.fresh and os.path.exists(mem_root):
        raise SystemExit("%s exists: pick a new --fresh name (a blank start must be blank)" % mem_root)
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    log_path = os.path.join("logs", "learn_%s.log" % me)
    log = open(log_path, "a")
    say = Speaker(lambda line: print(line, flush=True), log)
    if args.seed:
        shutil.copytree(args.seed, mem_root)
    mark_run(out, memory=mem_root, fresh=args.fresh, seed=args.seed)
    sess = Session(me, name, out, os.path.relpath(out, "rollouts"), mem_root, log_path, log, say,
                   {} if args.fresh else history(me))
    pushed = demo_cheat.apply_pending(mem_root, me, args.seed or demo_cheat.SEED) if args.fresh else []
    if args.live:                               # the brain panel: a new session, nothing left from the last one
        os.makedirs(os.path.dirname(args.live), exist_ok=True)
        if os.path.exists(args.live):
            os.remove(args.live)
        with open(os.path.join(os.path.dirname(args.live), "session.json"), "w") as f:
            json.dump({"session": out, "memory": mem_root, "me": me, "log": log_path,
                       "log_offset": os.path.getsize(log_path), "seed": args.seed, "started": time.time(),
                       "running": True, "pid": os.getpid()}, f)
    for line in pushed:                         # the brain panel's buttons pressed before this game
        say(line)
    say("session %s: %s, System 1 %s, earlier rounds: %s" % (name, me, args.model,
                                                             {o: len(r) for o, (_, r) in sess.by_opp.items()}))
    return sess


def mark_finished(args, sess: Session) -> None:
    """The panel's buttons stop working once the run is over."""
    if not args.live:
        return
    path = os.path.join(os.path.dirname(args.live), "session.json")
    try:
        with open(path) as f:
            done = dict(json.load(f), running=False)
        with open(path, "w") as f:
            json.dump(done, f)
    except (OSError, ValueError) as e:
        sess.say("could not mark the session finished for the panel: %s" % e)


@dataclass
class Match:
    """The game in progress: one opponent until the match is decided."""
    number: int = 0
    score: List[int] = field(default_factory=lambda: [0, 0])      # round wins (me, him)
    rounds: List[Dict] = field(default_factory=list)
    acts: List[Dict] = field(default_factory=list)

    def over(self) -> bool:
        return max(self.score) >= MATCH_WON or len(self.rounds) >= MAX_ROUNDS

    def next(self) -> "Match":
        return Match(self.number + 1)


def short_in_play(sess: Session, s1, s2: System2, reviewer: Reviewer, outside: OutsideWatch, opp: Optional[str],
                  now: str) -> str:
    """Before a round: the short memory System 1 plays with. A new opponent gets his (kept or new); else System 2's
    newest if a job just finished; else the file if the brain panel changed it. Returns the opponent."""
    done = s2.poll()
    path = reviewer.short_path(now)
    if now != opp:
        s1.short = reviewer.short_for(now, sess.by_opp)
        outside.seen(now, path)
    elif done and done[0] == now:
        s1.short = done[1]
        outside.seen(now, path, stamp=done[2])
        sess.say("System 2's new short memory vs %s is in play from this round" % now)
    else:
        got = outside.check(now, path, s2.busy(), reviewer.moves)
        if got:
            if got[0] is not OutsideWatch.KEEP:
                s1.short = got[0]
            sess.say(got[1])
    return now


def record_round(sess: Session, files: Dict, match: Match, opp: str, rnd, advice_on: bool) -> Dict:
    """Log the round, add it to what System 2 learns from, and count it in the match."""
    where = {"game": match.number, "round": len(match.rounds), "opp": opp, "log": sess.tag,
             "advice": "on" if advice_on else "off"}
    summary = dict(rnd.summary, **where)
    files["actions"].write("".join(json.dumps(dict(e, **where)) + "\n" for e in rnd.log))
    files["rounds"].write(json.dumps(summary) + "\n")
    a0, r0 = sess.by_opp.setdefault(opp, ([], []))
    a0 += [dict(e, **where) for e in rnd.log]
    r0.append(summary)
    match.rounds.append(summary)
    match.acts += rnd.log
    match.score[0] += rnd.result == "win"
    match.score[1] += rnd.result == "loss"
    sess.say("game %d round %d vs %s: %s (%d-%d)  dealt %d taken %d  %d actions, attacks %s" % (
        match.number, len(match.rounds) - 1, opp, rnd.result, match.score[0], match.score[1], rnd.summary["dealt"],
        rnd.summary["taken"], len(rnd.log), rnd.summary["outcomes"]))
    return summary


def end_game(sess: Session, files: Dict, match: Match, opp: str, s1, s2: System2) -> Match:
    result = "win" if match.score[0] > match.score[1] else "loss"
    advice = "on" if s1.advice_on else "off"
    files["games"].write(json.dumps({"game": match.number, "opp": opp, "result": result, "score": match.score,
                                     "rounds": [r["result"] for r in match.rounds], "log": sess.tag,
                                     "advice": advice}) + "\n")
    sess.say("GAME %d vs %s: %s %d-%d  (advice %s)" % (match.number, opp, result.upper(), match.score[0],
                                                       match.score[1], advice))
    s2.ask("game", opp, sess.by_opp, match.acts, s1.short, "the last game")
    return match.next()


def fight(args, sess: Session, b: MesenBridge, s1, s2: System2, reviewer: Reviewer, files: Dict) -> bool:
    """Arcade mode from power-on until --games / --minutes (returns True: let System 2 finish) or Ctrl-C."""
    deadline = time.time() + 60 * args.minutes if args.minutes else None
    rng, n, opp, match = random.Random(0), 0, None, Match()
    outside = OutsideWatch()
    b.set_capture("raw")
    sess.say("power-on: GAME START, picking %s ..." % sess.me)
    start_arcade(b, sess.me)
    try:
        while True:
            b.set_vars(VARS)
            row = dict(zip(NAMES, b.run([[]]).rams[-1]))
            now = CHARACTERS.get(row["p2_char"], "id%d" % row["p2_char"])
            opp = short_in_play(sess, s1, s2, reviewer, outside, opp, now)
            s1.advice_on = not (args.ab_advice and match.number % 2 == 1)
            rnd = play_round(b, s1, opp, None, rng, os.path.join(sess.out, "images"), n,
                             live_path=args.live or None, echo=not args.quiet)
            record_round(sess, files, match, opp, rnd, s1.advice_on)
            if match.over():
                match = end_game(sess, files, match, opp, s1, s2)
            elif rnd.result == "loss":
                s2.ask("round", opp, sess.by_opp, rnd.log, s1.short, "the last round")
            for f in files.values():
                f.flush()
            n += 1
            if args.games and match.number >= args.games:
                sess.say("%d game%s played: stopping (after System 2's review of it)" % (
                    match.number, "" if match.number == 1 else "s"))
                return True                       # let the running review (the playbook) finish on screen
            if deadline and time.time() >= deadline:
                sess.say("%.0f minutes up: stopping after %d games (%d rounds)" % (args.minutes, match.number, n))
                return True
            next_fight(b, me=sess.me)
    except KeyboardInterrupt:
        sess.say("Ctrl-C: stopping after %d games (%d rounds)" % (match.number, n))
        return False


def shutdown(sess: Session, steps) -> None:
    """Each cleanup step runs even if an earlier one fails."""
    for step in steps:
        if step is None:
            continue
        try:
            step()
        except Exception as e:
            sess.say("cleanup step failed: %s: %s" % (type(e).__name__, e))


def main() -> int:
    args = parse_args()
    signal.signal(signal.SIGTERM, _stop)     # `kill` stops it like Ctrl-C: Mesen, text laya and settings cleaned up
    sess = open_session(args)
    reviewer = Reviewer(sess.me, sess.mem_root, sess.say)
    s2 = System2(reviewer.job, sess.say)
    argv = launch_argv(args.port, None) if args.headless else window_argv(args.port, None, speed=args.speed)
    advisor, b, keep, files, finish = None, None, None, {}, False
    try:                                         # everything started from here on is cleaned up in `finally`
        advisor = None if args.advisor == "off" else Advisor(args.advisor)
        sess.say("System 1 picks with %s" % ("text laya %s + short memory" % args.advisor if advisor else
                                             "laya-vision's threshold rule"))
        s1 = System1(args.model, sess.me, advisor=advisor)
        keep = KeepMesenSettings().__enter__()   # the window saves its overrides on exit: put the settings back after
        b = MesenBridge(args.port, launch=argv)
        files = {k: open(os.path.join(sess.out, k + ".jsonl"), "a") for k in ("actions", "rounds", "games")}
        finish = fight(args, sess, b, s1, s2, reviewer, files)
    except KeyboardInterrupt:
        sess.say("Ctrl-C: stopping")
    finally:
        shutdown(sess, ((lambda: b.close()) if b else None, lambda: s2.close(wait=finish),
                        (lambda: advisor.close()) if advisor else None,
                        (lambda: keep.__exit__(None, None, None)) if keep else None,
                        lambda: [f.close() for f in files.values()]))
        mark_finished(args, sess)
        sess.say("session %s saved in %s" % (sess.name, sess.out))
        sess.log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
