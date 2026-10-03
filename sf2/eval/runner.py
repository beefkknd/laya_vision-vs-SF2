"""Running headless fights (scripts/ab_memory.py, notebook_run.py, play_system1.py).

    open_fight   one headless Mesen with states/p1_<me>_vs_<opp>.state loaded and checked, always closed
    open_logs    the run's JSONL files, always closed
    fan_out      one child process per job, as many at once as the machine-wide memory budget allows
                 (sf2.eval.budget), each with its own log; Ctrl-C, SIGTERM or an error stops every child, releases its
                 memory and closes every log, so nothing is left running
"""
import contextlib
import os
import signal
import subprocess
import sys
import time
from typing import Dict, IO, Iterator, List, Optional, Sequence, Tuple

from ..config import JOBS_WAIT_S
from ..emu.headless import launch_argv
from ..emu.mesen import MesenBridge
from ..vocab import IDS
from .budget import POLL, Budget, NoRoom
from ..emu.vs import NAMES, VARS

STOP_WAIT = 20          # seconds a child gets to shut Mesen down before it is killed

Key = Tuple[str, ...]


def savestate(me: str, opp: str) -> str:
    return os.path.join("states", "p1_%s_vs_%s.state" % (me, opp))


def read_state(me: str, opp: str, state: "Optional[str | bytes]" = None) -> bytes:
    """The savestate bytes to load. ``state`` may be the savestate CONTENT itself (bytes - used as is) or a PATH to it
    (str, read from disk; default states/p1_<me>_vs_<opp>.state). A path that does not exist is a SystemExit naming the
    path. Accepting bytes keeps a caller that already has the savestate in memory (the screen loop's in-play replay
    scorer) from passing it where a path is expected - which read os.path.exists() on the raw bytes, failed, and raised
    'no savestate <172 KB of bytes>', aborting the loop before Mesen was ever launched."""
    if isinstance(state, (bytes, bytearray)):
        return bytes(state)
    path = state or savestate(me, opp)
    if not os.path.exists(path):
        raise SystemExit("no savestate %s" % path)
    with open(path, "rb") as f:
        return f.read()


@contextlib.contextmanager
def open_fight(me: str, opp: str, port: int, rom: Optional[str] = None,
               state: "Optional[str | bytes]" = None) -> Iterator[Tuple[MesenBridge, bytes]]:
    """(bridge, savestate bytes): headless Mesen on ``port`` with raw capture and the fight VARS, the savestate
    (``state`` - a path, or the savestate bytes; default states/p1_<me>_vs_<opp>.state) loaded and checked to hold
    ``me`` (player 1) vs ``opp``. Mesen is launched headless (launch_argv) and closed on the way out, whatever happens;
    no manual Mesen is ever waited on."""
    state = read_state(me, opp, state)
    b = MesenBridge(port, launch=launch_argv(port, rom))
    try:
        b.set_capture("raw")
        b.set_vars(VARS)
        r = dict(zip(NAMES, b.load_state(state).rams[-1]))
        if (r["p1_char"], r["p2_char"]) != (IDS[me], IDS[opp]):
            raise SystemExit("savestate holds characters %s, expected %s" % (
                (r["p1_char"], r["p2_char"]), (IDS[me], IDS[opp])))
        yield b, state
    finally:
        b.close()


@contextlib.contextmanager
def open_logs(out: str, names: Sequence[str]) -> Iterator[Dict[str, IO]]:
    """<out>/<name>.jsonl for each name, opened for writing and closed on the way out."""
    os.makedirs(out, exist_ok=True)
    with contextlib.ExitStack() as stack:
        yield {n: stack.enter_context(open(os.path.join(out, n + ".jsonl"), "w")) for n in names}


def exit_on_sigterm() -> None:
    """SIGTERM (``kill``, or ``fan_out`` stopping a child) exits like Ctrl-C does: every ``finally`` runs, so Mesen
    and text laya are closed. Call it first thing in a runner's main and in each child."""
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))


def fan_out(cmds: Sequence[Tuple[Key, List[str]]], log_dir: str, job_gb: float,
            budget: Optional[Budget] = None) -> List[Key]:
    """Run every command, as many at once as the machine-wide memory budget allows (``job_gb`` each, sf2.eval.budget);
    the keys of those that exited non-zero. Logs: <log_dir>/<key joined by _>.log"""
    budget = budget or Budget()
    if job_gb > budget.total:
        raise NoRoom("a %.1f GB job never fits the %.0f GB budget" % (job_gb, budget.total))
    os.makedirs(log_dir, exist_ok=True)
    queue, running, failed, logs = list(cmds), [], [], []
    waited = time.time()
    try:
        while queue or running:
            for job in [j for j in running if j[1].poll() is not None]:
                running.remove(job)
                budget.release(job[2])
                if job[1].returncode != 0:
                    failed.append(job[0])
            rid = budget.try_reserve(job_gb) if queue else None
            if rid:
                key, cmd = queue.pop(0)
                logs.append(open(os.path.join(log_dir, "_".join(key) + ".log"), "w"))
                running.append((key, subprocess.Popen(cmd, stdout=logs[-1], stderr=subprocess.STDOUT), rid))
                waited = time.time()
                continue
            if queue and not running and time.time() - waited > JOBS_WAIT_S:
                raise NoRoom("waited %.0f s for room for a %.1f GB job" % (JOBS_WAIT_S, job_gb))
            if running and not queue:
                running[0][1].wait()               # nothing left to start: block on a child instead of polling
            else:
                time.sleep(POLL)
        return [k for k, _ in cmds if k in failed]
    finally:
        for _, p, _ in running:
            if p.poll() is None:
                p.terminate()
        for _, p, rid in running:
            try:
                p.wait(timeout=STOP_WAIT)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            budget.release(rid)
        for f in logs:
            f.close()
