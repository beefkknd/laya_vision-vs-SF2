"""scripts/eval_value.py's summary and gates on synthetic decisions (the model part runs on the real data)."""
import importlib.util
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
spec = importlib.util.spec_from_file_location("eval_value", os.path.join(HERE, "..", "scripts", "eval_value.py"))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def dec(i, throw_best, attacking=False, rng="close"):
    vals = {m: 0.0 for m in ("lp", "lk", "throw", "block_high", "block_low", "forward")}
    vals["throw"] = 10.0 if throw_best else -10.0
    return {"values": vals, "range": rng, "opp_state": "attack" if attacking else "stand", "action": "throw",
            "outcome": "hit", "pred_outcome": "hit", "net": i, "pred_net": float(i)}


def test_summary_and_passing_gates():
    ds = [dec(i, throw_best=True) for i in range(-20, 30)]
    s = ev.summarize(ds, "chunli", value=True)
    assert s["throw_top3_close"]["share"] == 1.0 and s["calibration_gate"]["pass"]
    assert ev.gates({"test_real": {"chunli": s}}) == []


def test_gates_fail_on_hidden_throw_and_flat_calibration():
    ds = [dict(dec(i, throw_best=False), net=0) for i in range(50)]
    s = ev.summarize(ds, "chunli", value=True)
    fails = ev.gates({"test_heldout_guile": {"chunli": s}})
    assert any("throw" in f for f in fails) and any("calibration" in f for f in fails)


def test_test_extra_calibration_is_reported_not_gated():
    ds = [dict(dec(i, throw_best=True), net=0) for i in range(50)]
    assert ev.gates({"test_extra": {"chunli": ev.summarize(ds, "chunli", value=True)}}) == []


def test_calibration_skips_moves_system1_never_picks():
    ds = [dec(i, throw_best=True) for i in range(-20, 30)] + [dict(dec(0, True), action="back", pred_net=None)] * 5
    s = ev.summarize(ds, "chunli", value=True)
    assert s["calibration_n"] == 50 and s["n"] == 55


def v2dec(i, throw_rank, explored=True, rng="close", att=False, air=False):
    d = dec(i, throw_best=throw_rank, attacking=att, rng=rng)
    return dict(d, explored=explored, opp_air=air)


TABLE = {("chunli", "close", 0, 0): {"throw": 10.0, "lp": -1.0}}


def test_v2_passes_when_model_matches_the_table():
    ds = ev.with_oracle([v2dec(i, True) for i in range(-20, 30)], "chunli", TABLE)
    s = {"v2": ev.summarize_v2(ds, "chunli")}
    assert s["v2"]["oracle_throw_top3_sweet"]["share"] == 1.0
    assert ev.gates_v2({"test_real": {"chunli": s}}) == []


def test_v2_fails_when_the_model_hides_the_throw_where_it_pays():
    ds = ev.with_oracle([v2dec(i, False) for i in range(-20, 30)], "chunli", TABLE)
    fails = ev.gates_v2({"test_heldout_guile": {"chunli": {"v2": ev.summarize_v2(ds, "chunli")}}})
    assert any("throw" in f for f in fails)


def test_v2_calibration_uses_explored_only_and_is_anchored():
    # policy rows (not explored) are perfectly calibrated, explored rows are noise: the gate must fail
    ds = [v2dec(i, True, explored=False) for i in range(-20, 30)] + \
         [dict(v2dec(i, True), net=0) for i in range(-20, 30)]
    s = ev.summarize_v2(ev.with_oracle(ds, "chunli", TABLE), "chunli")
    assert s["calibration_explored_n"] == 50 and not s["calibration_v2_gate"]["pass"]


def test_sweet_spot():
    assert ev.sweet({"range": "close", "opp_state": "stand", "opp_air": False})
    assert not ev.sweet({"range": "close", "opp_state": "attack", "opp_air": False})
    assert not ev.sweet({"range": "close", "opp_state": "stand", "opp_air": True})
    assert not ev.sweet({"range": "mid", "opp_state": "stand", "opp_air": False})


def test_sample_is_seeded_and_ordered():
    rows = [{"i": i} for i in range(100)]
    a, b = ev.sample(rows, 10), ev.sample(rows, 10)
    assert a == b and len(a) == 10 and [r["i"] for r in a] == sorted(r["i"] for r in a)
    assert ev.sample(rows, 0) == rows and ev.sample(rows, 500) == rows
