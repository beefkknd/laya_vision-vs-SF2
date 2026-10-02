"""sf2.data.eye_eval: the eye fine-tunes' scoring and the eye_all keep rule (docs/eye_questions_v1.md)."""
import pytest

from sf2.data import eye_eval as V


def _rows(spec):
    """spec: list of (answer, pred, match, extra)"""
    rows, preds = [], []
    for i, (a, p, m, extra) in enumerate(spec):
        rows.append(dict({"id": str(i), "answer": a, "pair_name": "p%d" % m, "game": 0}, **extra))
        preds.append(p)
    return rows, preds


def test_weighted_accuracy_q1_real_play():
    w = V.real_play_weights("q1")
    assert V.weighted_accuracy({"yes": 0.5, "no": 1.0}, w) == pytest.approx(0.035 * 0.5 + 0.965)


def test_q3_weights_split_attacking_by_the_pool():
    w = V.real_play_weights("q3", {"attack": 3, "special": 1, "moving": 99})
    assert w == pytest.approx({"moving": 0.715, "attack": 0.285 * 0.75, "special": 0.285 * 0.25})
    with pytest.raises(ValueError):
        V.real_play_weights("q3", None)


def test_q5_has_no_real_play_weight():
    assert V.real_play_weights("q5") is None and V.weighted_accuracy({"close": 1.0}, None) is None


def test_bad_weights_refused():
    with pytest.raises(ValueError):
        V.weighted_accuracy({"a": 1.0, "b": 1.0}, {"a": 0.5, "b": 0.6})


def test_summary_balanced_and_learned():
    spec = [("yes", "yes", m, {}) for m in range(10)] + [("no", "no", m, {}) for m in range(10)]
    spec[0] = ("yes", "no", 0, {})
    rows, preds = _rows(spec)
    res = V.summary(rows, preds, ["yes", "no"], ["yes"] * 5 + ["no"] * 5, "q1")
    assert res["model"]["balanced_accuracy"] == pytest.approx((0.9 + 1.0) / 2)
    assert res["majority"]["answer"] == "yes" and res["majority"]["balanced_accuracy"] == pytest.approx(0.5)
    assert res["chance"] == 0.5 and res["learned"] is True
    assert res["weighted_accuracy"] == pytest.approx(0.035 * 0.9 + 0.965 * 1.0)


def test_summary_not_learned_at_chance():
    rows, preds = _rows([("yes", "yes", m, {}) for m in range(10)] + [("no", "yes", m, {}) for m in range(10)])
    res = V.summary(rows, preds, ["yes", "no"], ["yes", "no"], "q1")
    assert res["model"]["balanced_accuracy"] == pytest.approx(0.5) and res["learned"] is False


def test_side_and_char_breakdowns_only_with_a_side():
    rows, preds = _rows([("air", "air", 0, {"side": "left", "char": "ryu"}),
                         ("ground", "air", 1, {"side": "right", "char": "ken"})])
    res = V.summary(rows, preds, ["ground", "air"], ["ground", "air"], "q4")
    assert res["by_side"]["left"]["accuracy"] == 1.0 and res["by_side"]["right"]["accuracy"] == 0.0
    assert set(res["by_char"]) == {"ryu", "ken"}
    rows, preds = _rows([("close", "close", 0, {}), ("far", "far", 1, {})])
    assert "by_side" not in V.summary(rows, preds, ["close", "far"], ["close", "far"], "q5")


def test_q1_tag_breakdowns():
    rows, preds = _rows([("yes", "no", 0, {"fire_frames": ["n-4"]}), ("yes", "yes", 1, {"fire_frames": ["n-4", "n"]}),
                         ("no", "yes", 2, {"hard": ["pose", "before_spawn"]}), ("no", "no", 3, {"hard": []})])
    res = V.summary(rows, preds, ["yes", "no"], ["yes", "no"], "q1")
    assert res["by_yes_frames"]["n-4 only"]["accuracy"] == 0.0 and res["by_yes_frames"]["both"]["accuracy"] == 1.0
    assert res["by_no_tag"]["pose"]["accuracy"] == 0.0 and res["by_no_tag"]["plain"]["accuracy"] == 1.0


def test_keep_rule_within_margin_keeps():
    r = V.keep_combined({"q1": 0.90, "q3": 0.80}, {"q1": 0.88, "q3": 0.85})
    assert r["keep"] is True and r["separate_adapters_for"] == []


def test_keep_rule_one_question_loses_more_drops():
    r = V.keep_combined({"q1": 0.90, "q3": 0.80}, {"q1": 0.879, "q3": 0.85})
    assert r["keep"] is False and r["separate_adapters_for"] == ["q1"]
    assert r["per_question"]["q1"]["diff"] == pytest.approx(-0.021)


def test_keep_rule_needs_the_same_questions():
    with pytest.raises(ValueError):
        V.keep_combined({"q1": 0.9}, {"q1": 0.9, "q3": 0.8})
