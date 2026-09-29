"""System 2 beside the game (scripts/learn_loop.py): the game never waits for Qwen.

    Speaker    one line to the console and the session log, with the time; safe from System 2's thread
    Reviewer   writes the memories into one memory root (memory/, or memory_runs/<name> for --fresh):
               the short memory vs an opponent (write or revise) and the playbook, each version kept in
               logs/system2/memory/ with how much it changed (sf2.system2.memory_churn)
    System2    runs a Reviewer job on a daemon thread, one at a time; while one runs only the newest request
               waits (an end-of-game job, which also rewrites the playbook, is not replaced by a lost-round one);
               ``poll`` hands a finished job's short memory to the loop before the next round
"""
import json
import os
import shutil
import threading
import time
from concurrent.futures import Future
from typing import Callable, Dict, IO, List, Optional, Tuple

from ..data.vs_sweep import actions
from .memory import load, playbook_path, save, short_path
from .memory_churn import diff as diff_memory
from .system2 import fits_laya, populate, review

KEEP = os.path.join("logs", "system2", "memory")     # every memory version System 2 wrote
CLOSE_WAIT = 300                                     # seconds a timed stop waits for a running review

ByOpp = Dict[str, Tuple[List[Dict], List[Dict]]]      # opponent -> (actions, rounds)
Done = Tuple[str, Optional[Dict], Optional[float]]    # (opponent, short memory, its file's stamp at the save)


class Speaker:
    """``say(msg)``: "HH:MM:SS  msg" to ``echo`` (print) and ``log``; one line at a time from any thread."""

    def __init__(self, echo: Callable[[str], None] = print, log: Optional[IO] = None):
        self.echo, self.log, self.lock = echo, log, threading.Lock()

    def __call__(self, msg: str) -> None:
        line = "%s  %s" % (time.strftime("%H:%M:%S"), msg)
        with self.lock:
            self.echo(line)
            if self.log:
                self.log.write(line + "\n")
                self.log.flush()


def snapshot(by_opp: ByOpp) -> ByOpp:
    """A copy System 2 can read while the game goes on adding to ``by_opp``."""
    return {o: (list(a), list(r)) for o, (a, r) in by_opp.items()}


class Reviewer:
    """System 2's writing, into ``root``. Every method speaks through ``say``."""

    def __init__(self, me: str, root: str, say: Speaker):
        self.me, self.root, self.say = me, root, say
        self.moves = list(actions(me))

    def short_path(self, opp: str) -> str:
        return short_path(self.me, opp, self.root)

    def _save(self, path: str, mem: Dict) -> None:
        """Write the new version, saying how much it changed from the one it replaces; keep a copy."""
        if os.path.exists(path):
            with open(path) as f:
                change = diff_memory(json.load(f), mem, self.moves + ["forward"])
            self.say("    change vs the previous %s: %s" % ("playbook" if "opp" not in mem else "short memory",
                                                            change.line()))
            for a, b in change.flipped:
                self.say("        FLIP: %r -> %r" % (a, b))
        save(path, mem)
        keep = os.path.join(KEEP, time.strftime("%Y%m%d-%H%M%S_") + os.path.basename(path))
        os.makedirs(os.path.dirname(keep), exist_ok=True)
        shutil.copy(path, keep)

    def _show(self, title: str, lessons: List[Dict]) -> None:
        self.say(title)
        for x in lessons:
            self.say("    %-15s %s  [%s %d/%d]" % (x["kind"], x["text"], x.get("claim", "-"), x["evidence"]["count"],
                                                   x["evidence"]["tries"]))

    def short(self, opp: str, by_opp: ByOpp, recent: Optional[List[Dict]] = None, what: str = "the last round",
              current: Optional[Dict] = None) -> Optional[Dict]:
        """Write (or, with ``recent`` and ``current``, revise from what just happened) the short memory vs ``opp``."""
        self.say("System 2: %s the short memory vs %s ..." % ("revising" if recent else "writing", opp))
        t = time.time()
        everything = [a for acts, _ in by_opp.values() for a in acts]
        playbook = load(playbook_path(self.me, self.root), self.moves)
        lessons, rep = populate(self.me, opp, playbook, by_opp.get(opp, ([], [])), everything,
                                recent=recent, current=current, what=what)
        if lessons is None:
            self.say("System 2: no usable short memory (%s); keeping the previous one" % rep["attempts"])
        else:
            self._save(self.short_path(opp), {"me": self.me, "opp": opp,
                                              "source": sorted({a.get("log") for a in everything}), "lessons": lessons})
            self._show("System 2 (%.0f s): short memory vs %s - %s" % (time.time() - t, opp, rep.get("notes")),
                       lessons)
            for att in rep["attempts"]:
                for r in att.get("rejected", []):
                    self.say("    rejected: %s (%s)" % (r["lesson"].get("text"), r["why"]))
        return load(self.short_path(opp), self.moves)

    def playbook(self, by_opp: ByOpp) -> None:
        self.say("System 2: reviewing %d rounds, rewriting the playbook ..." % sum(len(r) for _, r in by_opp.values()))
        t = time.time()
        path = playbook_path(self.me, self.root)
        lessons, rep = review(self.me, load(path, self.moves), by_opp)
        if lessons is None:
            self.say("System 2: no usable playbook (%s); keeping the previous one" % rep["attempts"])
            return
        self._save(path, {"me": self.me, "source": sorted({a.get("log") for acts, _ in by_opp.values() for a in acts}),
                          "lessons": lessons})
        self._show("System 2 (%.0f s): playbook - %s" % (time.time() - t, rep.get("notes")), lessons)

    def short_for(self, opp: str, by_opp: ByOpp) -> Optional[Dict]:
        """The short memory to play ``opp`` with: a kept one if it meets today's rules, else a new one."""
        kept = load(self.short_path(opp), self.moves)
        if fits_laya(kept):
            self._show("new opponent: %s - short memory kept from before" % opp, kept["lessons"])
            return kept
        self.say("new opponent: %s%s" % (opp, " (the kept short memory breaks today's rules: rewriting it)"
                                         if kept else ""))
        return self.short(opp, by_opp)

    def job(self, kind: str, opp: str, by_opp: ByOpp, recent: List[Dict], current: Optional[Dict],
            what: str) -> Done:
        """One background job: the playbook first after a game, then the short memory."""
        if kind == "game":
            self.playbook(by_opp)
        mem = self.short(opp, by_opp, recent=recent, what=what, current=current)
        path = self.short_path(opp)                    # stamped as System 2 left it (sf2.system2.memory.OutsideWatch)
        return opp, mem, os.path.getmtime(path) if os.path.exists(path) else None


class System2:
    """Runs ``job`` (Reviewer.job) in the background; see the module doc."""

    def __init__(self, job: Callable[..., Done], say: Speaker):
        self.job, self.say = job, say
        self.future: Optional[Future] = None
        self.waiting: Optional[tuple] = None

    def _start(self, args: tuple) -> Future:
        """A daemon thread: stopping the loop never waits for a Qwen call to return."""
        fut: Future = Future()

        def run():
            try:
                fut.set_result(self.job(*args))
            except BaseException as e:          # reported by poll(); never kills the game
                fut.set_exception(e)
        threading.Thread(target=run, name="system2", daemon=True).start()
        return fut

    def ask(self, kind: str, opp: str, by_opp: ByOpp, recent: List[Dict], current: Optional[Dict],
            what: str) -> None:
        args = (kind, opp, snapshot(by_opp), list(recent), current, what)
        if self.future is None or self.future.done():
            self.future = self._start(args)
        elif self.waiting is None or self.waiting[0] != "game" or kind == "game":
            if self.waiting:
                self.say("System 2 busy: %s review replaced by the newer %s review" % (self.waiting[0], kind))
            self.waiting = args

    def busy(self) -> bool:
        """A review is running or waiting (so a memory file may be about to change under System 2's own hand)."""
        return self.waiting is not None or (self.future is not None and not self.future.done())

    def poll(self) -> Optional[Done]:
        """A finished job's (opp, short memory, stamp), or None; starts the waiting job."""
        if self.future is None or not self.future.done():
            return None
        try:
            done = self.future.result()
        except Exception as e:                       # System 2 failing must not stop the game
            self.say("System 2 failed: %s: %s" % (type(e).__name__, e))
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
                self.say("waiting for System 2 to finish its review ...")
                try:
                    self.future.result(timeout=CLOSE_WAIT)
                except Exception as e:
                    self.say("System 2's last review failed: %s: %s" % (type(e).__name__, e))
            else:
                self.say("stopping: System 2's running review is abandoned")
        self.future, self.waiting = None, None
