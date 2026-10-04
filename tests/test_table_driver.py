"""Stage 3a-ii: run_loop with policy='table' drives a career with the value table (no Qwen, no short memory),
credits it from each round's outcomes, and persists it so it GROWS across blocks. Pure (no emulator) via a
table-aware fake play_round that calls decide(moment)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from looptools import load_driver, make_moment            # noqa: E402

from sf2.system1 import value_table as VT                 # noqa: E402

OPP = "ryu"


class FakeEmu:
    def new_round(self):
        return self


def fake_table_play(emu, cat, move, me, opp, state, state_id, delay, lines, out, reader=None, decide=None):
    os.makedirs(out, exist_ok=True)
    assert decide is not None, "the table policy must pass a decide()"
    d = decide(make_moment(dx=36, doing="standing", my_bar=120 / 176.0, his_bar=150 / 176.0))   # close/standing
    rec = {"i": 0, "k": 12, "k_prev": 8, "action": d["action"],
           "situation": ["close", "standing", "half", "half"],
           "moment": {"my_life": 120, "his_life": 150, "doing": "standing", "his_air": False,
                      "his_label": "stand", "side": "left", "dx": 36, "fireball": False},
           **{x: d[x] for x in ("when", "explored", "category") if x in d}}
    with open(os.path.join(out, "decisions.jsonl"), "w") as f:
        f.write(json.dumps(rec) + "\n")
    return {"decisions": 1}


def fake_score(round_dir):
    return {"result": "loss", "dealt": 0, "taken": 20, "hp": -20, "my_life_end": 100, "opp_life_end": 150}


class BoomQwen:
    def __call__(self, *a, **k):
        raise AssertionError("the table policy must NOT call Qwen")


def test_table_policy_credits_and_persists_the_table(tmp_path):
    driver = load_driver()
    out = str(tmp_path / "block0")
    save = str(tmp_path / "table.json")
    verdict = driver.run_loop(OPP, None, None, BoomQwen(), games=2, rounds=2, seed_lines=[], out=out,
                              play_round_fn=fake_table_play, state=b"x", state_id={"path": "p", "sha256": "0"},
                              emu=FakeEmu(), score_fn=fake_score, seed_rng=1, policy="table", save_table=save)
    assert verdict["policy"] == "table" and verdict["table_cells"] >= 1      # the table learned at least one cell
    assert os.path.isfile(save)                                             # persisted
    t = json.load(open(save))
    assert "close|standing|0" in t["cells"]                                 # credited the cell she played in
    # table events were traced
    events = [json.loads(l) for l in open(os.path.join(out, "trace.jsonl"))]
    assert any(e.get("event") == "table" for e in events)
    assert not any(e.get("event") == "qwen" for e in events)                 # no Qwen churn on the table path


def test_table_grows_across_blocks_via_carry(tmp_path):
    driver = load_driver()
    save = str(tmp_path / "table.json")
    # block 0
    driver.run_loop(OPP, None, None, BoomQwen(), games=1, rounds=2, seed_lines=[], out=str(tmp_path / "b0"),
                    play_round_fn=fake_table_play, state=b"x", state_id={"path": "p", "sha256": "0"},
                    emu=FakeEmu(), score_fn=fake_score, seed_rng=1, policy="table", save_table=save)
    carried = json.load(open(save))
    n0 = carried["cells"]["close|standing|0"].get(next(iter(carried["cells"]["close|standing|0"])))[0]
    # block 1 carries the table in and keeps crediting -> the cell's count grows
    driver.run_loop(OPP, None, None, BoomQwen(), games=1, rounds=2, seed_lines=[], out=str(tmp_path / "b1"),
                    play_round_fn=fake_table_play, state=b"x", state_id={"path": "p", "sha256": "0"},
                    emu=FakeEmu(), score_fn=fake_score, seed_rng=2, policy="table", table=carried, save_table=save)
    grown = json.load(open(save))
    total1 = sum(s[0] for s in grown["cells"]["close|standing|0"].values())
    assert total1 > n0                                                      # pooled across blocks (grew)


class QuietQwen:
    def __call__(self, messages, task):
        return ""                                  # scout prose / no claims -> the rules update is a no-op this round


def test_hybrid_runs_text_laya_base_credits_the_table_and_churns_rules(tmp_path):
    from looptools import FollowerLaya
    driver = load_driver()
    out = str(tmp_path / "hyb")
    save = str(tmp_path / "table.json")
    verdict = driver.run_loop(OPP, FollowerLaya(), FollowerLaya(), QuietQwen(), games=2, rounds=2, seed_lines=[],
                              out=out, play_round_fn=fake_table_play, state=b"x",
                              state_id={"path": "p", "sha256": "0"}, emu=FakeEmu(), score_fn=fake_score,
                              seed_rng=1, policy="hybrid", save_table=save)
    assert verdict["policy"] == "hybrid" and verdict["table_cells"] >= 1      # the table was credited every round
    import os as _os
    assert _os.path.isfile(save)
    events = [json.loads(l) for l in open(_os.path.join(out, "trace.jsonl"))]
    assert any(e.get("event") == "table" for e in events)                    # table credited
    assert any(e.get("event") == "qwen" for e in events)                     # text-laya's rules update also ran
