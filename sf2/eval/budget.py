"""The machine-wide memory budget for headless jobs. On 2026-09-29 two A/B runs started side by side (38 jobs, ~208 GB
of Python next to Qwen's 39 GB) ran the Mac out of memory and swap, and a watchdog panic rebooted it.

Every fan-out (sf2.eval.runner.fan_out), in any process or checkout, reserves each job's memory in one ledger
(config.JOBS_LEDGER, a JSON list locked with flock) before starting it:
    - the reservations of live processes together stay within ``total_gb`` (config.JOBS_BUDGET_GB);
    - the machine must really have the job's memory plus ``margin_gb`` free (other programs use memory the ledger does
      not know about), counting jobs started less than ``load_s`` ago as not loaded yet (their models arrive later);
    - a reservation whose owner process is gone is dropped, so a killed run frees its share.
A job that could never fit raises NoRoom instead of waiting forever.
"""
import fcntl
import json
import os
import re
import subprocess
import time
import uuid
from typing import Callable, Dict, List, Optional

from ..config import JOBS_BUDGET_GB, JOBS_LEDGER, JOBS_LOAD_S, JOBS_MARGIN_GB

POLL = 0.2              # seconds between tries while waiting for room


class NoRoom(RuntimeError):
    pass


def available_gb() -> float:
    """Memory the Mac can hand out now: free + inactive + speculative + purgeable pages (vm_stat)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    pages = {k: int(v) for k, v in re.findall(r"Pages (\w+):\s+(\d+)\.", out)}
    return sum(pages.get(k, 0) for k in ("free", "inactive", "speculative", "purgeable")) * page / 1e9


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class Budget:
    def __init__(self, path: str = JOBS_LEDGER, total_gb: float = JOBS_BUDGET_GB, margin_gb: float = JOBS_MARGIN_GB,
                 load_s: float = JOBS_LOAD_S, available_gb: Callable[[], float] = available_gb):
        self.path, self.total, self.margin, self.load_s, self.free = path, total_gb, margin_gb, load_s, available_gb
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _update(self, change: Callable[[List[Dict]], Optional[str]]) -> Optional[str]:
        """Run ``change`` on the live reservations under the lock and save them."""
        with open(self.path + ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                with open(self.path) as f:
                    rows = json.load(f)
                rows = rows if isinstance(rows, list) else []
            except (OSError, ValueError):
                rows = []
            rows = [r for r in rows if isinstance(r, dict) and _alive(r.get("pid", -1))]
            out = change(rows)
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(rows, f)
            os.replace(tmp, self.path)
            return out

    def try_reserve(self, gb: float) -> Optional[str]:
        """A reservation id, or None if there is no room now."""
        def change(rows):
            now = time.time()
            loading = sum(r["gb"] for r in rows if now - r["t"] < self.load_s)
            if sum(r["gb"] for r in rows) + gb > self.total or self.free() - loading < gb + self.margin:
                return None
            rid = uuid.uuid4().hex
            rows.append({"pid": os.getpid(), "gb": gb, "t": now, "id": rid})
            return rid
        return self._update(change)

    def reserve(self, gb: float, timeout: Optional[float] = None) -> str:
        """Wait for room (up to ``timeout`` s); NoRoom if the job can never fit or the wait runs out."""
        if gb > self.total:
            raise NoRoom("a %.1f GB job never fits the %.0f GB budget" % (gb, self.total))
        end = None if timeout is None else time.time() + timeout
        while True:
            rid = self.try_reserve(gb)
            if rid:
                return rid
            if end is not None and time.time() >= end:
                raise NoRoom("no room for a %.1f GB job after %.0f s (budget %.0f GB, %.0f GB free on the machine)"
                             % (gb, timeout, self.total, self.free()))
            time.sleep(POLL)

    def release(self, rid: str) -> None:
        def change(rows):
            rows[:] = [r for r in rows if r.get("id") != rid]
        self._update(change)
