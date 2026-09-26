"""The gate reports how many matches really differed and how noisy its damage numbers are."""
import math

from sf2.rollout import gate


def _match(ep, rounds):
    return [{"episode": ep, "round": i, "dmg_for": d, "dmg_against": 176, "winner": "opp", "end_frame": f}
            for i, (d, f) in enumerate(rounds)]


def test_identical_matches_count_once():
    rounds = _match(0, [(40, 2000), (0, 4100)]) + _match(1, [(40, 2000), (0, 4100)]) + _match(2, [(40, 2010), (0, 4100)])
    g = gate([], rounds)
    assert g["matches"] == 3 and g["distinct_matches"] == 2


def test_spread_and_standard_error_of_damage_per_round():
    dealt = [10, 50, 90, 130]
    g = gate([], _match(0, [(d, 1000 * (i + 1)) for i, d in enumerate(dealt)]))
    sd = math.sqrt(sum((d - 70) ** 2 for d in dealt) / 3)
    assert abs(g["dmg_dealt_sd"] - sd) < 1e-9
    assert abs(g["damage_score_se"] - 100 * sd / 176 / 2) < 1e-9


def test_standard_error_of_net_damage_per_round():
    rounds = [{"episode": 0, "round": i, "dmg_for": f, "dmg_against": a, "winner": "opp", "end_frame": i}
              for i, (f, a) in enumerate([(10, 176), (176, 40), (60, 176), (90, 176)])]
    net = [f - a for f, a in [(10, 176), (176, 40), (60, 176), (90, 176)]]
    mean = sum(net) / 4
    sd = math.sqrt(sum((x - mean) ** 2 for x in net) / 3)
    assert abs(gate([], rounds)["net_damage_se"] - sd / 2) < 1e-9


def test_no_special_move_whiffs_and_hot_means_hit():
    """No specials in the action set: nothing can whiff, and a decision is hot when she is hit soon after it."""
    from sf2.rollout import annotate

    meta = dict(episode=0, round=0, dmg_for=0, dmg_against=0)
    rows = annotate([{"meta": dict(meta, frame=0, action="hp")}, {"meta": dict(meta, frame=4, action="idle", dmg_against=12)},
                     {"meta": dict(meta, frame=200, action="lk")}])
    assert [r["meta"]["hot"] for r in rows] == [True, True, False]
    assert all("whiff" not in r["meta"] for r in rows)
    g = gate(rows, _match(0, [(0, 300)]))
    assert not [k for k in g if "whiff" in k]
