"""collect (fake emulator) -> dataset on disk -> relabel -> laya-vision's own loader."""
import os
import random
import subprocess
import sys

import pytest

from sf2 import dataset as D
from sf2.loop import play
from sf2.rollout import gate

from fake_retro import make_env

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
    assert len(r["target"]) == 12 and abs(sum(r["target"]) - 1) < 1e-6
    assert r["state_text"].startswith("me=ryu opp=guile dist=")
    for im in r["images"]:
        assert os.path.exists(tmp_path / "data/seed" / im)
    assert all("dmg_for_next" in x["meta"] and "round_result" in x["meta"] for x in recs)
    g = gate(rows, rounds)
    assert 0 <= g["round_win_rate"] <= 1 and g["decisions"] == len(rows)

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
        assert e["q"]["t"] == "choice" and len(e["target"]) == 12
        assert len(e["state"]["images"]) == 2 and all(os.path.exists(p) for p in e["state"]["images"])
        assert e["state"]["context"].startswith("me=ryu")
