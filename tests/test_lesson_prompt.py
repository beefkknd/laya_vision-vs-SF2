"""Qwen's one prompt in the lesson loop (sf2.system2.lesson_prompt): two views of each game (attack: what her attacks
did; defense: where the damage she took came from), her record, the registry; one claim per view back."""
from sf2.system2 import lessons as L
from sf2.system2.lesson_prompt import attack_view, defense_view, messages, parse_claims, record


def act(move, rng, net, air=False, rnd=0, kind="attack", actual=None, attacked=False):
    return {"action": move, "range": rng, "kind": kind, "dealt": max(net, 0), "taken": max(-net, 0),
            "opp_air": air, "opp_state": "attack" if attacked else "stand", "round": rnd, "opp_attacked": attacked,
            "actual": actual or ("hit" if net > 0 else "whiff")}


ROWS = ([act("sweep", "close", -12, attacked=True)] * 25 + [act("hp", "close", 9, air=True)] * 22 +
        [act("back", "close", -1, kind="movement", attacked=True)] * 25 + [act("c.mk", "mid", 2)] * 30)


def test_the_attack_view_is_relative_to_her_average_there():
    got = attack_view(ROWS)
    assert got[0].startswith("- hp up close when he jumps: 22 tries, net +9.0 per decision")
    assert all("vs her" in x for x in got)


def test_the_defense_view_says_where_the_damage_came_from():
    got = "\n".join(defense_view(ROWS))
    assert "punished after sweep up close when he attacks: 25 times, 300 damage" in got
    assert "when he attacked, she chose:" in got and "back 25" in got


def test_record():
    rounds = [{"result": "loss", "dealt": 50, "taken": 176}, {"result": "win", "dealt": 176, "taken": 90}]
    assert record(rounds, rounds[-1:]) == ("Last game: 1 round won, 0 lost (dealt 176, took 90 per round). "
                                          "So far: 1 won, 1 lost (dealt 113, took 133 per round).")


def test_the_prompt_holds_both_views_the_record_and_the_registry():
    reg, _ = L.propose([], [{"kind": "avoid", "move": "sweep", "range": "close", "when": "attacking"}], ROWS, 0)
    rounds = [{"result": "loss", "dealt": 50, "taken": 176}]
    msgs = messages("chunli", "ken", reg, ROWS, ROWS, rounds, rounds)
    user = msgs[1]["content"]
    for part in ("So far:", "registered: avoid sweep up close when he attacks", "ATTACK", "DEFENSE"):
        assert part in user
    assert '"attack"' in msgs[0]["content"] and '"defense"' in msgs[0]["content"]


def test_parse_claims_one_per_view():
    claims, problems = parse_claims({"attack": {"kind": "use_more", "move": "hp", "range": "close", "when": "jumping",
                                                "why": "lands"},
                                     "defense": {"kind": "always", "move": "block_low", "range": "", "when": "attacking"}})
    assert [c["view"] for c in claims] == ["attack", "defense"] and claims[1]["range"] is None and not problems
    claims, problems = parse_claims({"attack": None, "defense": {"kind": "avoid", "move": "sweep"}})
    assert len(claims) == 1 and claims[0]["view"] == "defense"
    for bad in (None, [], "x", {"attack": [1]}, {"claims": []}):
        assert parse_claims(bad)[0] == [] and parse_claims(bad)[1]


def test_record_before_any_game():
    assert record([], []).startswith("No game played yet in this session")


def test_the_prompt_lists_her_moves_with_the_defensive_ones_and_last_refusals():
    moves = ["lp", "sweep", "block_high", "block_low", "back", "jump_back"]
    refused = [{"claim": {"kind": "avoid", "move": "hp", "range": "close", "when": "jumping"}, "state": "refused",
                "why": "already registered"}]
    msgs = messages("chunli", "ken", [], ROWS, ROWS, [], [], moves=moves, refused=refused)
    assert "block_low" in msgs[0]["content"] and "never used" in msgs[0]["content"]
    assert "Refused last time" in msgs[1]["content"] and "already registered" in msgs[1]["content"]
