"""scripts/qwen_lessons.py --book: the loop starts from the opponent's verified tips (default off: no book, nothing
changes - the lock replays run as before); the run file, every ledger row and the verdict name the book and its
sha256."""
import hashlib
import json
from types import SimpleNamespace

from tests.test_lock import qwen_lessons
from tests.test_verified_lessons import HP, THROW


def book_file(tmp_path):
    p = tmp_path / "book.json"
    p.write_text(json.dumps({"me": "chunli", "opponents": {"ken": {"lines": [THROW, HP], "not_verified": []}}}))
    return str(p), hashlib.sha256(p.read_bytes()).hexdigest()


def test_the_book_is_off_by_default():
    q = qwen_lessons()
    args = SimpleNamespace(book=None, opp="ken")
    assert q.load_book(None, "ken") == ([], None)
    assert q.book_meta(args) == {}
    assert q.start_registry(args, "loop") == []


def test_the_loop_arm_starts_from_the_opponents_verified_lines(tmp_path):
    q = qwen_lessons()
    path, digest = book_file(tmp_path)
    tips, got = q.load_book(path, "ken")
    assert got == digest and [t["line"] for t in tips] == [THROW["line"], HP["line"]]
    args = SimpleNamespace(book=path, opp="ken")
    reg = q.start_registry(args, "loop")
    assert [(r["state"], r["line"]) for r in reg] == [("verified", THROW["line"]), ("verified", HP["line"])]
    assert reg[0]["evidence"]["book"] == path
    assert q.start_registry(args, "none") == []                      # the no-advice arm plays with nothing
    assert q.book_meta(args) == {"book": path, "book_sha256": digest}


def test_an_opponent_missing_from_the_book_starts_empty_but_is_still_recorded(tmp_path):
    q = qwen_lessons()
    path, digest = book_file(tmp_path)
    assert q.load_book(path, "ryu") == ([], digest)
    assert q.book_meta(SimpleNamespace(book=path, opp="ryu")) == {"book": path, "book_sha256": digest}


def test_the_verdict_counts_verified_lines_and_never_as_violations(tmp_path):
    q = qwen_lessons()
    path, _ = book_file(tmp_path)
    reg = q.start_registry(SimpleNamespace(book=path, opp="ken"), "loop")
    root = tmp_path / "run"
    (root / "loop").mkdir(parents=True)
    (root / "loop" / "ledger.jsonl").write_text(json.dumps({"game": 0, "claims": [], "outcome": [], "problems": [],
                                                           "registry": reg, "in_play": [], "violations": []}) + "\n")
    v = q.verdict(str(root), "ken", False)
    assert v["verified_at_end"] == [THROW["line"], HP["line"]] and v["violations"] == 0


def test_a_headless_loop_arm_plays_with_the_book_and_logs_it(tmp_path, monkeypatch):
    """play_arm with the game, the models and Qwen stubbed: the verified lines are in play from game 0, every ledger
    row keeps them verified and names the book; Qwen being down changes nothing."""
    import contextlib
    q = qwen_lessons()
    path, digest = book_file(tmp_path)
    seen = []

    class S1:
        def __init__(self, *a, **k):
            self.short = None

    def play_round(b, s1, opp, state, rng, _, i):
        seen.append([x["text"] for x in s1.short["lessons"]])
        return SimpleNamespace(log=[], result="win", summary={"result": "win", "dealt": 10, "taken": 0})

    def qwen_down(*a):
        raise RuntimeError("qwen down")

    monkeypatch.setattr(q, "Advisor", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(q, "open_fight", lambda *a, **k: contextlib.nullcontext((None, None)))
    monkeypatch.setattr(q, "System1", S1)
    monkeypatch.setattr(q, "play_round", play_round)
    monkeypatch.setattr(q, "chat", qwen_down)
    args = SimpleNamespace(opp="ken", seed=1, lock=None, prompt="character_fgc", track=None, book=path,
                           forward_lessons=False, history=False, games=2, rounds=1, advisor="a", model="m", state=None)
    out = tmp_path / "loop"
    assert q.play_arm(args, "loop", 0, str(out)) == 0
    assert seen == [[THROW["line"], HP["line"]]] * 2
    rows = [json.loads(x) for x in (out / "ledger.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and all(r["book"] == path and r["book_sha256"] == digest for r in rows)
    assert all([x["state"] for x in r["registry"]] == ["verified", "verified"] for r in rows)
    assert json.loads((out / "run.json").read_text())["book_sha256"] == digest
