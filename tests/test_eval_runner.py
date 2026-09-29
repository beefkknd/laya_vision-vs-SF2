"""The fan-out of headless runs: failures reported, and an interrupt never leaves a child running."""
import os
import subprocess
import sys
import time

import pytest

from sf2.eval.runner import fan_out

PY = sys.executable


def test_reports_the_runs_that_failed(tmp_path):
    cmds = [(("ryu", "none"), [PY, "-c", "print('ok')"]), (("ken", "none"), [PY, "-c", "raise SystemExit(3)"])]
    assert fan_out(cmds, str(tmp_path)) == [("ken", "none")]
    assert open(tmp_path / "ryu_none.log").read() == "ok\n"


def test_an_interrupt_stops_every_child(tmp_path, monkeypatch):
    started = []
    real = subprocess.Popen
    real_wait = real.wait

    def popen(*a, **k):
        p = real(*a, **k)
        started.append(p)
        return p

    def interrupted(self, timeout=None):
        if self is started[0] and timeout is None:
            raise KeyboardInterrupt
        return real_wait(self, timeout)

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(real, "wait", interrupted)
    cmds = [((o, "none"), [PY, "-c", "import time; time.sleep(60)"]) for o in ("ryu", "ken")]
    with pytest.raises(KeyboardInterrupt):
        fan_out(cmds, str(tmp_path))
    assert len(started) == 2 and all(p.poll() is not None for p in started)


def test_a_stopped_child_runs_its_cleanup(tmp_path, monkeypatch):
    """terminate() sends SIGTERM: a child that called exit_on_sigterm() still runs its `finally` (closes Mesen)."""
    mark = tmp_path / "closed"
    child = ("import sys, time; sys.path.insert(0, %r)\n"
             "from sf2.eval.runner import exit_on_sigterm\nexit_on_sigterm()\n"
             "open(%r, 'w').close()\n"
             "try:\n    time.sleep(60)\nfinally:\n    open(%r, 'w').write('yes')\n") % (
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), str(tmp_path / "up"), str(mark))
    real_wait = subprocess.Popen.wait

    def interrupted(self, timeout=None):
        if timeout is None:
            end = time.time() + 10
            while not (tmp_path / "up").exists() and time.time() < end and self.poll() is None:
                time.sleep(0.05)
            raise KeyboardInterrupt
        return real_wait(self, timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", interrupted)
    with pytest.raises(KeyboardInterrupt):
        fan_out([(("ryu", "none"), [PY, "-c", child])], str(tmp_path))
    assert mark.read_text() == "yes"
