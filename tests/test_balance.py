"""The balance gate: no character (or opponent) may dominate a training mix (owner rule: no Chun-Li focus)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import balance  # noqa: E402

NOTE = "me=%s stand hp=100 opp=%s stand hp=100 dist=mid facing=right corner=none time=early last=idle fireball=none"


def _data(tmp_path, name, me, opp, n):
    d = tmp_path / name
    d.mkdir()
    with open(d / "train.jsonl", "w") as f:
        for _ in range(n):
            f.write(json.dumps({"state_text": NOTE % (me, opp)}) + "\n")
    return str(d)


def test_counts_rows_per_character_and_opponent(tmp_path):
    ds = [_data(tmp_path, "a", "ryu", "ken", 100), _data(tmp_path, "b", "honda", "ryu", 100)]
    c = balance.count(ds)
    assert c["me"] == {"ryu": 100, "honda": 100} and c["opp"] == {"ken": 100, "ryu": 100}


def test_equal_characters_pass(tmp_path):
    ds = [_data(tmp_path, m, m, "ryu", 100 + i) for i, m in enumerate(["ryu", "honda", "chunli"])]
    assert balance.check(balance.count(ds), tol=0.10) == []


def test_a_dominant_character_fails_with_its_name(tmp_path):
    ds = [_data(tmp_path, "c", "chunli", "ryu", 300), _data(tmp_path, "h", "honda", "ryu", 100),
          _data(tmp_path, "g", "guile", "ryu", 100)]
    problems = balance.check(balance.count(ds), tol=0.10)
    assert any("chunli" in p for p in problems)


def test_opponents_are_checked_only_when_asked(tmp_path):
    ds = [_data(tmp_path, "a", "ryu", "ken", 100), _data(tmp_path, "b", "honda", "ryu", 100),
          _data(tmp_path, "c", "guile", "ryu", 100)]
    assert balance.check(balance.count(ds), tol=0.10) == []
    assert any("opponent" in p for p in balance.check(balance.count(ds), tol=0.10, opponents=True))


def test_cli_exits_nonzero_on_imbalance(tmp_path, monkeypatch):
    ds = [_data(tmp_path, "c", "chunli", "ryu", 300), _data(tmp_path, "h", "honda", "ryu", 100)]
    monkeypatch.setattr(sys, "argv", ["balance.py", *ds])
    with pytest.raises(SystemExit) as e:
        balance.main()
    assert e.value.code == 1
