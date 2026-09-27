"""play (fake emulator) -> dataset on disk -> laya-vision's own loader."""
import os
import random

import pytest

from sf2 import actions as A
from sf2 import dataset as D
from sf2.loop import play
from sf2.rollout import gate

from fake_mesen import make_env


def random_choose(rng):
    def choose(env, prev, cur, text):
        return rng.choice(A.ACTIONS), {"actor": "random"}
    return choose


def test_play_writes_a_dataset_laya_loads(tmp_path):
    env = make_env()
    w = D.Writer(str(tmp_path / "data"), "seed", source="seed", val_every=2)
    rows, rounds = play(env, random_choose(random.Random(0)), matches=2, writer=w, log_every=0)
    w.close()
    assert rounds and all(r["winner"] in ("me", "opp", "draw") for r in rounds)
    assert w.n["train"] > 0 and w.n["val"] > 0
    recs = D.read(str(tmp_path / "data/seed/train.jsonl"))
    r = recs[0]
    assert len(r["target"]) == len(A.ACTIONS) and abs(sum(r["target"]) - 1) < 1e-6
    assert r["target"] == D.one_hot(r["meta"]["action"])  # the label is the move played
    assert r["state_text"].startswith("me=ryu ") and " opp=ken " in r["state_text"] and " dist=" in r["state_text"]
    for im in r["images"]:
        assert os.path.exists(tmp_path / "data/seed" / im)
    assert all("dmg_for_next" in x["meta"] and "round_result" in x["meta"] for x in recs)
    g = gate(rows, rounds)
    assert 0 <= g["round_win_rate"] <= 1 and g["decisions"] == len(rows)
    assert 0 <= g["damage_score"] <= 100
    assert g["net_damage_per_round"] == g["dmg_dealt_per_round"] - g["dmg_taken_per_round"]

    vt = pytest.importorskip("laya.vlm_train")
    ex = vt.load_jsonl_examples(str(tmp_path / "data"), "seed", "train")
    assert len(ex) == len(recs)
    e = ex[0]
    assert e["q"]["t"] == "choice" and len(e["target"]) == len(A.ACTIONS)
    assert len(e["state"]["images"]) == 2 and all(os.path.exists(p) for p in e["state"]["images"])
    assert e["state"]["context"].startswith("me=ryu")


def test_rows_carry_her_action_state_at_decision_time():
    """So a rollout can count her Lightning Legs (state 0C) and other states after the fact."""
    env = make_env()
    rows, _ = play(env, lambda *a: ("hk", {}), matches=1, max_decisions=20, log_every=0)
    assert rows and all("my_state" in r["meta"] for r in rows)
