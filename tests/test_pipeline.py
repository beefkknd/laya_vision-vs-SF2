"""collect (fake emulator) -> dataset on disk -> relabel -> laya-vision's own loader."""
import os
import random
import subprocess
import sys

import pytest

from sf2 import actions as A
from sf2 import dataset as D
from sf2.loop import play
from sf2.rollout import gate

from fake_mesen import make_env

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def teacher_choose(rng):
    def choose(env, prev, cur, text, t_dist):
        acts = list(t_dist)
        return rng.choices(acts, weights=[t_dist[a] for a in acts])[0], {"actor": "teacher"}
    return choose


def test_collect_relabel_load(tmp_path):
    env = make_env()
    w = D.Writer(str(tmp_path / "data"), "seed", source="seed", val_every=2)
    rows, rounds = play(env, teacher_choose(random.Random(0)), matches=2, writer=w, log_every=0)
    w.close()
    assert rounds and all(r["winner"] in ("me", "opp", "draw") for r in rounds)
    assert w.n["train"] > 0 and w.n["val"] > 0
    recs = D.read(str(tmp_path / "data/seed/train.jsonl"))
    r = recs[0]
    assert len(r["target"]) == len(A.ACTIONS) and abs(sum(r["target"]) - 1) < 1e-6
    assert r["state_text"].startswith("me=ryu ") and " opp=ken " in r["state_text"] and " dist=" in r["state_text"]
    for im in r["images"]:
        assert os.path.exists(tmp_path / "data/seed" / im)
    assert all("dmg_for_next" in x["meta"] and "round_result" in x["meta"] for x in recs)
    g = gate(rows, rounds)
    assert 0 <= g["round_win_rate"] <= 1 and g["decisions"] == len(rows)
    assert 0 <= g["damage_score"] <= 100
    assert g["net_damage_per_round"] == g["dmg_dealt_per_round"] - g["dmg_taken_per_round"]

    # pretend the seed data is a student rollout, then DAgger-relabel it
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/relabel.py"), "--rollout",
                          str(tmp_path / "data/seed"), "--name", "dagger_r1", "--out", str(tmp_path / "data")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    dag = D.read(str(tmp_path / "data/dagger_r1/train.jsonl"))
    for im in dag[0]["images"]:
        assert os.path.exists(os.path.normpath(tmp_path / "data/dagger_r1" / im))

    vt = pytest.importorskip("laya.vlm_train")
    for name in ("seed", "dagger_r1"):
        ex = vt.load_jsonl_examples(str(tmp_path / "data"), name, "train")
        assert len(ex) == len(D.read(str(tmp_path / "data" / name / "train.jsonl")))
        e = ex[0]
        assert e["q"]["t"] == "choice" and len(e["target"]) == len(A.ACTIONS)
        assert len(e["state"]["images"]) == 2 and all(os.path.exists(p) for p in e["state"]["images"])
        assert e["state"]["context"].startswith("me=ryu")


def test_relabel_leaves_out_decisions_where_the_stick_did_nothing(tmp_path):
    ro = tmp_path / "rollouts" / "r0"
    ro.mkdir(parents=True)
    meta = dict(action="idle", teacher_action="block", dmg_for_next=5, dmg_against_next=0, hot=False)
    recs = [{"id": "r%d" % i, "episode": 0, "step": i, "images": ["images/a.png", "images/a.png"], "label": 9,
             "target": D.one_hot("block"), "meta": dict(meta, controllable=c)}
            for i, c in enumerate([True, False, True])]
    D.write_jsonl(str(ro / "train.jsonl"), recs)
    for mode in ("dagger", "filter"):
        out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/relabel.py"), "--rollout", str(ro),
                              "--name", mode, "--mode", mode, "--out", str(tmp_path / "data")],
                             capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert [r["id"] for r in D.read(str(tmp_path / "data" / mode / "train.jsonl"))] == ["r0", "r2"]
