"""Running headless fights (scripts/ab_memory.py, notebook_run.py, play_system1.py).

    open_fight   one headless Mesen with states/p1_<me>_vs_<opp>.state loaded and checked, always closed
    open_logs    the run's JSONL files, always closed
    fan_out      one child process per job side by side, each with its own log; Ctrl-C, SIGTERM or an error stops
                 every child and closes every log, so nothing is left running
"""
import contextlib
import os
import signal
import subprocess
import sys
from typing import Dict, IO, Iterator, List, Optional, Sequence, Tuple

from ..emu.headless import launch_argv
from ..emu.mesen import MesenBridge
from ..vocab import IDS
from ..emu.vs import NAMES, VARS

STOP_WAIT = 20          # seconds a child gets to shut Mesen down before it is killed

Key = Tuple[str, ...]


def savestate(me: str, opp: str) -> str:
    return os.path.join("states", "p1_%s_vs_%s.state" % (me, opp))


@contextlib.contextmanager
def open_fight(me: str, opp: str, port: int, rom: Optional[str] = None) -> Iterator[Tuple[MesenBridge, bytes]]:
    """(bridge, savestate bytes): headless Mesen on ``port`` with raw capture and the fight VARS, the savestate
    loaded and checked to hold ``me`` (player 1) vs ``opp``. Mesen is closed on the way out, whatever happens."""
    path = savestate(me, opp)
    if not os.path.exists(path):
        raise SystemExit("no savestate %s" % path)
    with open(path, "rb") as f:
        state = f.read()
    b = MesenBridge(port, launch=launch_argv(port, rom))
    try:
        b.set_capture("raw")
        b.set_vars(VARS)
        r = dict(zip(NAMES, b.load_state(state).rams[-1]))
        if (r["p1_char"], r["p2_char"]) != (IDS[me], IDS[opp]):
            raise SystemExit("%s holds characters %s, expected %s" % (
                path, (r["p1_char"], r["p2_char"]), (IDS[me], IDS[opp])))
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


def fan_out(cmds: Sequence[Tuple[Key, List[str]]], log_dir: str) -> List[Key]:
    """Start every command, wait for all; the keys of those that exited non-zero. Logs: <log_dir>/<key joined by _>.log"""
    os.makedirs(log_dir, exist_ok=True)
    jobs, logs = [], []
    try:
        for key, cmd in cmds:
            logs.append(open(os.path.join(log_dir, "_".join(key) + ".log"), "w"))
            jobs.append((key, subprocess.Popen(cmd, stdout=logs[-1], stderr=subprocess.STDOUT)))
        return [k for k, p in jobs if p.wait() != 0]
    finally:
        for _, p in jobs:
            if p.poll() is None:
                p.terminate()
        for _, p in jobs:
            try:
                p.wait(timeout=STOP_WAIT)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for f in logs:
            f.close()
