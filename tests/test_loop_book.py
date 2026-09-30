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
