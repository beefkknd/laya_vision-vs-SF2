"""The machine-wide memory budget for headless jobs (sf2.eval.budget). On 2026-09-29 two A/B runs started side by side
(38 jobs, ~208 GB of Python) ran the Mac out of memory and swap; a watchdog panic rebooted it. Every fan-out now
reserves each job's memory in one ledger shared by all processes, and checks what the machine really has left."""
import json
import os
import subprocess
import sys
import threading
import time

import pytest

from sf2.eval.budget import Budget, NoRoom

PY = sys.executable


def budget(tmp_path, total=10.0, free=1000.0, margin=0.0, load_s=0.0):
    return Budget(str(tmp_path / "jobs.json"), total_gb=total, margin_gb=margin, load_s=load_s,
                  available_gb=lambda: free)


def test_reservations_stay_within_the_budget(tmp_path):
    b = budget(tmp_path, total=10)
    got = [b.try_reserve(4) for _ in range(3)]
    assert [g is not None for g in got] == [True, True, False]           # 4 + 4 fit, a third 4 does not
    b.release(got[0])
    assert b.try_reserve(4) is not None


def test_two_processes_share_one_ledger(tmp_path):
    """A second fan-out (another run, another checkout) sees the first one's reservations."""
    a, b = budget(tmp_path, total=10), budget(tmp_path, total=10)
    assert a.try_reserve(6) is not None
    assert b.try_reserve(6) is None and b.try_reserve(4) is not None


def test_a_dead_owner_frees_its_share(tmp_path):
    path = str(tmp_path / "jobs.json")
    dead = subprocess.Popen([PY, "-c", "pass"])
    dead.wait()
    with open(path, "w") as f:
        json.dump([{"pid": dead.pid, "gb": 9, "t": time.time(), "id": "x"}], f)
    assert budget(tmp_path, total=10).try_reserve(9) is not None


def test_the_machine_must_really_have_the_memory(tmp_path):
    """Other programs (Qwen, a browser) use memory the ledger does not know about."""
    b = budget(tmp_path, total=100, free=20, margin=16)
    assert b.try_reserve(3) is not None                                  # 3 + 16 <= 20
    assert b.try_reserve(6) is None                                      # 6 + 16 > 20


def test_a_job_still_loading_counts_against_free_memory(tmp_path):
    """Its models arrive ~30 s after the start: free memory does not show it yet."""
    b = budget(tmp_path, total=100, free=26, margin=10, load_s=60)
    assert b.try_reserve(6) is not None and b.try_reserve(6) is not None   # 26 - 6 >= 6 + 10
    assert b.try_reserve(6) is None                                          # 26 - 12 < 6 + 10


def test_a_job_that_can_never_fit_is_refused(tmp_path):
    with pytest.raises(NoRoom):
        budget(tmp_path, total=10).reserve(12, timeout=0.2)


def test_reserve_waits_for_a_release(tmp_path):
    b = budget(tmp_path, total=10)
    first = b.reserve(8, timeout=1)
    threading.Timer(0.3, b.release, args=(first,)).start()
    t = time.time()
    assert b.reserve(8, timeout=5) is not None and time.time() - t >= 0.25


def test_fan_out_never_runs_more_than_the_budget(tmp_path):
    from sf2.eval.runner import fan_out
    log = tmp_path / "events"
    child = ("import os, time; p=%r\n"
             "open(p, 'a').write('+%%d\\n' %% os.getpid()); time.sleep(0.4); open(p, 'a').write('-\\n')") % str(log)
    cmds = [(("job", str(i)), [PY, "-c", child]) for i in range(6)]
    assert fan_out(cmds, str(tmp_path / "logs"), job_gb=4, budget=budget(tmp_path, total=10)) == []
    running, peak = 0, 0
    for line in log.read_text().split():
        running += 1 if line.startswith("+") else -1
        peak = max(peak, running)
    assert peak == 2                                                     # 10 GB / 4 GB each
    assert json.load(open(tmp_path / "jobs.json")) == []                 # everything released


def test_fan_out_releases_on_interrupt(tmp_path, monkeypatch):
    from sf2.eval.runner import fan_out
    real_wait = subprocess.Popen.wait

    def interrupted(self, timeout=None):
        if timeout is None:
            raise KeyboardInterrupt
        return real_wait(self, timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", interrupted)
    b = budget(tmp_path, total=10)
    with pytest.raises(KeyboardInterrupt):
        fan_out([(("a",), [PY, "-c", "import time; time.sleep(30)"])], str(tmp_path / "logs"), job_gb=4, budget=b)
    assert json.load(open(tmp_path / "jobs.json")) == []


def test_available_memory_reads_the_machine():
    from sf2.eval.budget import available_gb
    gb = available_gb()
    assert 0 < gb < os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9 + 1
