"""A lock (sf2.eval.lock): the frozen inputs of a result - checkpoints, savestates, the play data used as history - copied
and hashed, so the result can be re-run exactly; a changed or missing file is refused."""
import argparse
import importlib.util
import json
import os
import sys

import pytest

from sf2.eval import lock


def tree(root, files):
    for rel, text in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tree(str(tmp_path), {"runs/v/best/model.safetensors": "w1", "runs/t/a/adapter.safetensors": "w2",
                         "states/p1_chunli_vs_ken.state": "s1",
                         "rollouts/games_v1/chunli/actions.jsonl": json.dumps({"me": "chunli", "opp": "ken"}) + "\n",
                         "rollouts/games_v1/chunli/images/x.png": "big"})
    return tmp_path


def make(**kw):
    args = dict(name="v1", model="runs/v/best", advisor="runs/t/a", states=["states/p1_chunli_vs_ken.state"],
                play=["rollouts/games_v1/chunli"], runs=[{"opp": "ken", "seed": 5}], extra={"qwen": "q"})
    args.update(kw)
    return lock.make(**args)


def test_a_lock_copies_and_hashes_the_inputs_but_not_images(repo):
    m = make()
    assert m["files"]["model/model.safetensors"] and m["runs"] == [{"opp": "ken", "seed": 5}]
    art = os.path.join("locks", "v1", "artifacts")
    assert os.path.exists(os.path.join(art, "play", "games_v1/chunli/actions.jsonl"))
    assert not os.path.exists(os.path.join(art, "play", "games_v1/chunli/images"))
    assert lock.verify("v1") == []
    assert lock.paths("v1")["model"] == os.path.join(art, "model")


def test_a_changed_or_missing_file_is_refused(repo):
    make()
    art = os.path.join("locks", "v1", "artifacts")
    with open(os.path.join(art, "advisor", "adapter.safetensors"), "w") as f:
        f.write("changed")
    os.remove(os.path.join(art, "states", "p1_chunli_vs_ken.state"))
    bad = lock.verify("v1")
    assert any("advisor/adapter.safetensors" in b and "changed" in b for b in bad)
    assert any("states/p1_chunli_vs_ken.state" in b and "missing" in b for b in bad)


def test_a_lock_is_never_overwritten(repo):
    make()
    with pytest.raises(SystemExit, match="exists"):
        make()


def test_play_rows_come_from_the_snapshot_only(repo):
    make()
    tree(str(repo), {"rollouts/learn/chunli/new/actions.jsonl": json.dumps({"me": "chunli", "opp": "ken"}) + "\n"})
    rows = lock.play_rows("v1", "chunli", "ken")
    assert len(rows) == 1 and rows[0]["log"] == "games_v1/chunli"


def qwen_lessons():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.join(here, "scripts") not in sys.path:
        sys.path.insert(0, os.path.join(here, "scripts"))
    spec = importlib.util.spec_from_file_location("qwen_lessons", os.path.join(here, "scripts", "qwen_lessons.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_run_from_a_lock_takes_everything_from_the_snapshot(repo):
    make()
    q = qwen_lessons()
    args = argparse.Namespace(lock="v1", run=0, opp=None, seed=None, games=10, rounds=3, history=True,
                              model="runs/other", advisor="runs/other")
    q.apply_lock(args)
    p = lock.paths("v1")
    assert (args.opp, args.seed, args.model, args.advisor) == ("ken", 5, p["model"], p["advisor"])
    assert args.state == os.path.join(p["states"], "p1_chunli_vs_ken.state")
    assert q.decisions("ken", "v1") == lock.play_rows("v1", "chunli", "ken")


def test_a_run_from_a_changed_lock_is_refused(repo):
    make()
    with open(os.path.join("locks", "v1", "artifacts", "states", "p1_chunli_vs_ken.state"), "w") as f:
        f.write("other")
    args = argparse.Namespace(lock="v1", run=0, opp=None, seed=None, games=10, rounds=3, history=True,
                              model="m", advisor="a")
    with pytest.raises(SystemExit, match="p1_chunli_vs_ken.state: changed"):
        qwen_lessons().apply_lock(args)


def test_lock_reruns_are_never_play_data(repo):
    from sf2.eval.logs import is_test
    assert is_test(os.path.join("rollouts", "locked", "v1", "x_ken", "loop"))
