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
