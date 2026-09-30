"""Offline gates of the laya-vision value fine-tune (docs/prereg_lv_value.md), as pure functions over decisions."""
import pytest

from sf2.eval.value_eval import calibration, calibration_gate, top3_share


def test_calibration_quintiles_sorted_by_prediction():
    pairs = [(p, 2 * p) for p in range(100)]
    rows = calibration(pairs, k=5)
    assert [r["n"] for r in rows] == [20] * 5
    assert [r["mean_real"] for r in rows] == sorted(r["mean_real"] for r in rows)
    g = calibration_gate(rows, min_spread=15)
    assert g["monotone"] and g["spread"] == pytest.approx(rows[-1]["mean_real"] - rows[0]["mean_real"]) and g["pass"]


def test_calibration_gate_fails_on_noise_and_on_reversal():
    flat = calibration([(p, 0) for p in range(100)], k=5)
    assert not calibration_gate(flat, min_spread=15)["pass"]
    rev = calibration([(p, -p) for p in range(100)], k=5)
    g = calibration_gate(rev, min_spread=15)
    assert not g["monotone"] and not g["pass"]


def test_calibration_refuses_too_few():
    with pytest.raises(ValueError):
        calibration([(1, 1)] * 4, k=5)


def test_top3_share():
    ds = [{"values": {"throw": 9, "lk": 1, "mk": 2, "hk": 3}, "range": "close"},
          {"values": {"throw": 0, "lk": 1, "mk": 2, "hk": 3}, "range": "close"},
          {"values": {"throw": 9, "lk": 1, "mk": 2, "hk": 3}, "range": "far"}]
    assert top3_share(ds, "throw", lambda d: d["range"] == "close") == {"share": 0.5, "n": 2}
    assert top3_share(ds, "throw", lambda d: False) == {"share": None, "n": 0}


def test_top3_ties_do_not_favour_the_move():
    # a tie at the 3rd place: the move counts only if it is strictly among the best 3 by value then name order
    ds = [{"values": {"a": 5, "b": 5, "c": 5, "throw": 5}}]
    assert top3_share(ds, "throw", lambda d: True)["share"] == 0.0
