"""Which logs count as play data: test runs (A/B, notebook, --fresh demo sessions) never feed a coach or System 2."""
import json
import os

from sf2.code_coach import attacks
from sf2.eval.logs import mark_run, play_dirs, sources


def write_log(root, rel, opp="ryu", n=1, meta=None):
    d = os.path.join(root, rel)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "actions.jsonl"), "w") as f:
        for _ in range(n):
            f.write(json.dumps({"me": "chunli", "opp": opp, "kind": "attack", "action": "sweep", "range": "mid",
                                "dealt": 10, "taken": 0, "actual": "hit"}) + "\n")
    if meta is not None:
        mark_run(d, **meta)
    return d


def test_test_runs_are_not_play_data(tmp_path):
    root = str(tmp_path / "rollouts")
    write_log(root, "games_v1/chunli")
    write_log(root, "learn/chunli/s1", meta={"memory": "memory", "fresh": None})
    write_log(root, "learn/chunli/s0")                                     # older session, no marker: normal play
    write_log(root, "learn/chunli/s2", meta={"memory": "memory_runs/video1", "fresh": "video1"})
    write_log(root, "ab/20260928-091745/ryu_none")
    write_log(root, "notebook/20260928-000000/ryu_learn")
    got = [os.path.relpath(d, root) for d in play_dirs(root)]
    assert got == ["games_v1/chunli", "learn/chunli/s0", "learn/chunli/s1"]


def test_coach_counts_only_play_data_and_says_where_from(tmp_path):
    root = str(tmp_path / "rollouts")
    write_log(root, "games_v1/chunli", n=2)
    write_log(root, "ab/x/ryu_none", n=5)
    write_log(root, "learn/chunli/s2", n=7, meta={"fresh": "video1"})
    rows = attacks("chunli", root)
    assert len(rows) == 2 and sources(rows) == ["games_v1/chunli"]


def test_a_broken_marker_is_a_test_run(tmp_path):                         # unreadable -> never trusted as play data
    root = str(tmp_path / "rollouts")
    d = write_log(root, "learn/chunli/s3")
    with open(os.path.join(d, "run.json"), "w") as f:
        f.write("{half")
    assert play_dirs(root) == []


def test_rounds_come_from_rounds_jsonl_or_the_older_games_jsonl(tmp_path):
    from sf2.eval.logs import load_rounds
    new, old = tmp_path / "new", tmp_path / "old"
    new.mkdir()
    old.mkdir()
    (new / "rounds.jsonl").write_text('{"round": 0}\n\n{"round": 1}\n')
    (new / "games.jsonl").write_text('{"game": 0}\n')
    (old / "games.jsonl").write_text('{"game": 0}\n')
    assert load_rounds(str(new)) == [{"round": 0}, {"round": 1}]
    assert load_rounds(str(old)) == [{"game": 0}] and load_rounds(str(tmp_path / "none")) == []


def test_read_jsonl_missing():
    import pytest
    from sf2.data.dataset import read
    assert read("/no/such/file.jsonl", missing_ok=True) == []
    with pytest.raises(FileNotFoundError):
        read("/no/such/file.jsonl")
