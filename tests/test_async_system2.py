"""System 2 in the background (sf2.system2.async_runner): one job at a time, only the newest request waits (a game
review is never replaced by a round review), a failing job never stops the game, and a stop never hangs."""
import threading

from sf2.system2.async_runner import Speaker, System2, snapshot


class FakeReviewer:
    """Records jobs; each waits for ``go`` so the test decides when it finishes."""

    def __init__(self):
        self.done, self.go, self.fail = [], {}, set()

    def job(self, kind, opp, by_opp, recent, current, what):
        gate = self.go.setdefault((kind, opp), threading.Event())
        gate.wait(5)
        if (kind, opp) in self.fail:
            raise RuntimeError("Qwen down")
        self.done.append((kind, opp))
        return opp, {"lessons": [kind]}, 1.0


def wait_done(s2):
    for _ in range(200):
        if not s2.busy() or s2.future.done():
            return
        threading.Event().wait(0.01)


def make():
    said = []
    r = FakeReviewer()
    return r, System2(r.job, Speaker(said.append)), said


def test_one_job_at_a_time_and_the_newest_waits():
    r, s2, said = make()
    s2.ask("round", "ryu", {}, [], None, "the last round")
    s2.ask("round", "ryu", {}, [], None, "the last round")          # waits
    s2.ask("game", "ryu", {}, [], None, "the last game")            # replaces the waiting round review
    s2.ask("round", "ryu", {}, [], None, "the last round")          # never replaces a waiting game review
    assert s2.busy() and s2.waiting[0] == "game"
    assert any("replaced" in x for x in said)
    r.go[("round", "ryu")].set()
    wait_done(s2)
    assert s2.poll() == ("ryu", {"lessons": ["round"]}, 1.0)        # and the waiting game review starts
    r.go.setdefault(("game", "ryu"), threading.Event()).set()
    wait_done(s2)
    assert s2.poll()[1] == {"lessons": ["game"]} and not s2.busy()
    assert r.done == [("round", "ryu"), ("game", "ryu")]


def test_a_failed_job_is_reported_not_raised():
    r, s2, said = make()
    r.fail.add(("round", "ken"))
    s2.ask("round", "ken", {}, [], None, "the last round")
    r.go[("round", "ken")].set()
    wait_done(s2)
    assert s2.poll() is None and any("System 2 failed" in x and "Qwen down" in x for x in said)


def test_close_without_wait_abandons_a_running_review():
    r, s2, said = make()
    s2.ask("game", "ryu", {}, [], None, "the last game")
    s2.close(wait=False)                                             # returns at once; the thread is a daemon
    assert not s2.busy() and any("abandoned" in x for x in said)
    r.go[("game", "ryu")].set()


def test_snapshot_is_a_copy():
    by_opp = {"ryu": ([1], [2])}
    snap = snapshot(by_opp)
    by_opp["ryu"][0].append(3)
    assert snap == {"ryu": ([1], [2])}


def test_speaker_writes_the_log_with_a_time(tmp_path):
    lines = []
    log = open(tmp_path / "l.log", "a")
    Speaker(lines.append, log)("hello")
    log.close()
    assert lines[0].endswith("  hello") and (tmp_path / "l.log").read_text() == lines[0] + "\n"
