"""Question 8's soft target (docs/prereg_u_perception.md): per (table cell, move), the share of bootstrap resamples where
the move's mean net is >= walking in + 3 ("likely works"), above it ("may work"), or not ("likely fails"). Opponents
are resampled as whole clusters, so a cell where opponents disagree never gets a confident answer."""
import random

import pytest

from sf2.data.perception import Q8_ANSWERS, q8_targets

CELL = ("chunli", "close", 0, 0)


def e(opp, action, net, game=0, explored=True, me="chunli", rng="close", att="stand", air=False):
    dealt, taken = (net, 0) if net >= 0 else (0, -net)
    return {"me": me, "opp": opp, "action": action, "dealt": dealt, "taken": taken, "game": game,
            "explored": explored, "range": rng, "opp_state": att, "opp_air": air}


def many(opp, action, net, k, **kw):
    return [e(opp, action, net, game=g % 10 if g % 10 not in (2, 5, 8) else 0, **kw) for g in range(k)]


def target(es, move, cell=CELL, **kw):
    out = q8_targets(es, resamples=200, seed=0, **kw)
    return out[cell][move]


def test_opponents_that_agree_give_a_confident_answer():
    es = many("ryu", "sweep", 20, 40) + many("ken", "sweep", 18, 40) + many("ryu", "forward", 0, 40) + \
        many("ken", "forward", 0, 40)
    t = target(es, "sweep")
    assert t["p"]["likely works"] >= 0.95 and t["n"] == 80 and t["n_by_opp"] == {"ken": 40, "ryu": 40}


def test_opponents_that_disagree_give_a_spread_answer():
    es = many("ryu", "sweep", 25, 60) + many("ken", "sweep", -25, 60) + many("ryu", "forward", 0, 60) + \
        many("ken", "forward", 0, 60)
    p = target(es, "sweep")["p"]
    assert max(p.values()) < 0.85 and p["likely works"] > 0.1 and p["likely fails"] > 0.1


def test_pooling_decisions_alone_would_have_been_confident_here():
    """The same disagreeing cell with the opponent label erased (one cluster): the decision-level bootstrap is
    sure - which is why the clusters are opponents."""
    es = many("x", "sweep", 25, 60) + many("x", "sweep", -24, 60) + many("x", "forward", -10, 120)
    assert target(es, "sweep")["p"]["likely works"] >= 0.95


def test_probabilities_sum_to_one_and_are_in_answer_order():
    es = many("ryu", "lp", 3, 10) + many("ryu", "forward", 1, 10)
    p = target(es, "lp")["p"]
    assert list(p) == list(Q8_ANSWERS) and abs(sum(p.values()) - 1) < 1e-9 and all(0 <= v <= 1 for v in p.values())


def test_the_filter_is_the_tables():
    """Explored decisions of training-split games only, never Chun-Li vs Guile."""
    es = (many("ryu", "lp", 10, 10) + [e("ryu", "lp", 10, game=2)] * 7 + [e("ryu", "lp", 10, explored=False)] * 5 +
          [e("guile", "lp", 10)] * 9 + many("ryu", "forward", 0, 10))
    t = target(es, "lp")
    assert t["n"] == 10 and t["n_by_opp"] == {"ryu": 10}


def test_seeded_and_reproducible():
    rng = random.Random(1)
    es = [e(rng.choice(["ryu", "ken", "honda"]), rng.choice(["lp", "sweep", "forward"]), rng.randint(-30, 30),
            game=rng.choice([0, 1, 3])) for _ in range(300)]
    assert q8_targets(es, resamples=50, seed=4) == q8_targets(es, resamples=50, seed=4)
    assert q8_targets(es, resamples=50, seed=4) != q8_targets(es, resamples=50, seed=5)


def test_a_move_never_tried_in_the_cell_is_reported_with_n_0():
    es = many("ryu", "forward", -2, 30)
    t = target(es, "lp", moves={"chunli": ["lp", "forward"]})
    assert t["n"] == 0 and t["p"]["may work"] == 1.0        # the table's 0 beats walking in at -2 (shrunk), by < 3
    assert t["p"]["likely works"] == 0.0 and all(0 <= v <= 1 for v in t["p"].values())


def test_forward_has_no_target_of_its_own():
    es = many("ryu", "forward", 0, 30)
    assert "forward" not in q8_targets(es, resamples=20, seed=0)[CELL]


def test_cells_follow_the_table():
    es = many("ryu", "lp", 5, 10, rng="far", att="attack", air=True) + many("ryu", "forward", 0, 10, rng="far",
                                                                            att="attack", air=True)
    assert ("chunli", "far", 1, 1) in q8_targets(es, resamples=20, seed=0)


@pytest.mark.parametrize("bad", [0, -1])
def test_resamples_must_be_positive(bad):
    with pytest.raises(ValueError):
        q8_targets([], resamples=bad, seed=0)
