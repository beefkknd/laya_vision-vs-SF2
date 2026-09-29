"""Running headless jobs side by side (scripts/ab_memory.py, scripts/notebook_run.py): one child process per job, each
with its own log. Ctrl-C, SIGTERM or an error stops every child and closes every log; nothing is left running."""
import os
import signal
import subprocess
import sys
from typing import List, Sequence, Tuple

STOP_WAIT = 20          # seconds a child gets to shut Mesen down before it is killed

Key = Tuple[str, ...]


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
