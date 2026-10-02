"""Scoring the movement fine-tunes (sf2.data.mv_eval): balanced accuracy, baselines, the match-cluster bootstrap
lower bound and the learned verdict, on hand-made predictions with known answers."""
import pytest

from sf2.data import mv_eval as E

ANS = ("ground", "air")


def rows_of(spec):
    """spec: [(pair, game, truth, facing, char)]"""
    return [{"pair_name": p, "game": g, "answer": t, "facing": f, "char": c} for p, g, t, f, c in spec]


def test_balanced_accuracy_is_the_mean_recall_and_majority_scores_one_over_k():
    truths = ["ground"] * 8 + ["air"] * 2
    preds = ["ground"] * 8 + ["ground", "air"]
    m = E.metrics(truths, preds, ANS)
    assert m["accuracy"] == 0.9 and m["recall"] == {"ground": 1.0, "air": 0.5} and m["balanced_accuracy"] == 0.75
    assert m["confusion"] == {"ground": {"ground": 8}, "air": {"ground": 1, "air": 1}}
    assert E.majority(["ground", "air", "ground"], ANS) == "ground"
    assert E.majority(["air", "ground"], ANS) == "ground"                  # a tie: the earlier answer
    with pytest.raises(ValueError):
        E.metrics(["ground"], ["flying"], ANS)


def _spec(n_matches, right):
    out = []
    for g in range(n_matches):
        for t in ANS:
            for _ in range(5):
                out.append(("ryu_vs_ken", g, t, "left" if g % 2 else "right", "ryu" if g % 3 else "ken"))
    rows = rows_of(out)
    preds = [r["answer"] if right(i, r) else ("air" if r["answer"] == "ground" else "ground")
             for i, r in enumerate(rows)]
    return rows, preds


def test_a_perfect_model_is_learned_and_a_constant_one_is_not():
    rows, preds = _spec(20, lambda i, r: True)
    res = E.evaluate(rows, preds, ANS, ["ground"] * 3 + ["air"])
    assert res["model"]["balanced_accuracy"] == 1.0 and res["balanced_lower_bound"] == 1.0 and res["learned"]
    assert res["majority"]["answer"] == "ground" and res["majority"]["balanced_accuracy"] == 0.5
    const = ["ground"] * len(rows)
    bad = E.evaluate(rows, const, ANS, ["ground"])
    assert bad["model"]["balanced_accuracy"] == 0.5 and not bad["learned"]


def test_the_bound_resamples_whole_matches_so_one_lucky_match_does_not_pass():
    # right only in match 0: balanced accuracy 0.55 overall, but the match-level bootstrap often drops match 0
    rows, preds = _spec(10, lambda i, r: r["game"] == 0)
    res = E.evaluate(rows, preds, ANS, ["ground"])
    assert res["model"]["balanced_accuracy"] == pytest.approx(0.1)
    rows, preds = _spec(10, lambda i, r: r["game"] == 0 or i % 2 == 0)
    res = E.evaluate(rows, preds, ANS, ["ground"])
    assert res["model"]["balanced_accuracy"] == pytest.approx(0.55)
    assert res["balanced_lower_bound"] < 0.55 and not res["learned"]


def test_breakdowns_per_character_and_facing():
    rows, preds = _spec(4, lambda i, r: r["facing"] == "right")
    res = E.evaluate(rows, preds, ANS, ["ground"])
    assert res["by_facing"]["right"]["balanced_accuracy"] == 1.0
    assert res["by_facing"]["left"]["balanced_accuracy"] == 0.0
    assert set(res["by_char"]) == {"ryu", "ken"} and res["matches"] == 4


def test_rows_of_one_match_move_together_in_the_bootstrap():
    # 20 matches, 12 entirely right and 8 entirely wrong: balanced 0.6 over 200 rows looks sure row by row
    # (lower bound ~0.53), but only 20 independent matches: the match-level bound is far lower
    rows, preds = _spec(20, lambda i, r: r["game"] < 12)
    res = E.evaluate(rows, preds, ANS, ["ground"])
    assert res["model"]["balanced_accuracy"] == pytest.approx(0.6)
    assert res["balanced_lower_bound"] < 0.45 and not res["learned"]
