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
