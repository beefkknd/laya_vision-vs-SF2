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


class FakeBridge:
    """Stands in for MesenBridge: records calls; the savestate holds characters ``chars``."""
    opened = []

    def __init__(self, port, launch=None, chars=(5, 0)):
        self.port, self.closed, self.chars = port, False, chars
        FakeBridge.opened.append(self)

    def set_capture(self, mode):
        pass

    def set_vars(self, vars_):
        self.vars = vars_

    def load_state(self, state):
        from types import SimpleNamespace
        from sf2.vs import NAMES
        row = [0] * len(NAMES)
        row[NAMES.index("p1_char")], row[NAMES.index("p2_char")] = self.chars
        return SimpleNamespace(rams=[row])

    def close(self):
        self.closed = True


@pytest.fixture
def fake_fight(tmp_path, monkeypatch):
    from sf2.eval import runner
    monkeypatch.chdir(tmp_path)
    (tmp_path / "states").mkdir()
    (tmp_path / "states" / "p1_chunli_vs_ryu.state").write_bytes(b"state")
    FakeBridge.opened = []
    monkeypatch.setattr(runner, "MesenBridge", FakeBridge)
    monkeypatch.setattr(runner, "launch_argv", lambda port, rom: "mesen")
    return runner


def test_open_fight_checks_the_savestate_and_always_closes(fake_fight):
    with fake_fight.open_fight("chunli", "ryu", 49000) as (b, state):
        assert state == b"state" and not b.closed
    assert FakeBridge.opened[0].closed
    with pytest.raises(RuntimeError):
        with fake_fight.open_fight("chunli", "ryu", 49000):
            raise RuntimeError("a round crashed")
    assert FakeBridge.opened[1].closed


def test_open_fight_refuses_the_wrong_characters(fake_fight, monkeypatch):
    monkeypatch.setattr(FakeBridge, "__init__", lambda self, port, launch=None: FakeBridge.opened.append(self) or
                        setattr(self, "chars", (5, 4)) or setattr(self, "closed", False))
    with pytest.raises(SystemExit, match="expected"):
        with fake_fight.open_fight("chunli", "ryu", 49000):
            pass
    assert FakeBridge.opened[0].closed
    with pytest.raises(SystemExit, match="no savestate"):
        with fake_fight.open_fight("chunli", "ken", 49000):
            pass


def test_open_logs_closes_every_file(tmp_path):
    from sf2.eval.runner import open_logs
    with open_logs(str(tmp_path / "run"), ("actions", "rounds")) as f:
        f["actions"].write("{}\n")
        files = list(f.values())
    assert all(x.closed for x in files) and (tmp_path / "run" / "actions.jsonl").read_text() == "{}\n"


def test_the_advisor_closes_its_server_when_the_block_ends():
    from sf2.advisor import Advisor
    a = Advisor.__new__(Advisor)                   # no server started: only the context protocol is tested
    closed = []
    a.close = lambda: closed.append(True)
    with pytest.raises(RuntimeError):
        with a as got:
            assert got is a
            raise RuntimeError
    assert closed == [True]
