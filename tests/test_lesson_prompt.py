"""Qwen's one prompt in the lesson loop (sf2.system2.lesson_prompt): what it sees, and reading what it proposes."""

from sf2.system2 import lessons as L
from sf2.system2.lesson_prompt import MAX_CLAIMS, by_situation, messages, parse_claims


def act(move, rng, net, air=False, rnd=0):
    return {"action": move, "range": rng, "kind": "attack", "dealt": max(net, 0), "taken": max(-net, 0),
            "opp_air": air, "opp_state": "stand", "round": rnd}


def test_the_prompt_shows_the_registry_the_table_and_the_last_game_by_his_state():
    rows = [act("sweep", "close", -10)] * 30 + [act("hp", "close", 9, air=True, rnd=5)] * 6
    reg, _ = L.propose([], [{"kind": "avoid", "move": "sweep", "range": "close", "when": None}], rows, 0)
    reg, _ = L.propose(reg, [{"kind": "avoid", "move": "hp", "range": "close", "when": None}], rows, 0)
    user = messages("chunli", "ken", reg, rows, [a for a in rows if a["round"] == 5])[1]["content"]
    assert "registered: avoid sweep up close" in user and "rejected: avoid hp up close" in user
    assert "move sweep, range close" in user                       # the overall table
    assert "hp up close when he jumps: 6 tries" in user            # the last game, by his state


def test_by_situation_groups_and_orders_by_size():
    rows = [act("hp", "close", 9, air=True)] * 6 + [act("hp", "close", -2)] * 4 + [act("lp", "far", 1)] * 2
    got = by_situation(rows, min_tries=3)
    assert got[0].startswith("- hp up close when he jumps: 6 tries, net +9.0") and len(got) == 2


def test_parse_claims():
    ok = {"claims": [{"kind": "use_more", "move": "hp", "range": "close", "when": "jumping", "why": "lands"},
                     {"kind": "avoid", "move": "sweep", "range": None, "when": None, "why": "x"},
                     {"kind": "avoid", "move": "lp", "range": "far", "when": None}]}
    claims, problems = parse_claims(ok)
    assert len(claims) == MAX_CLAIMS == 2 and "only the first" in problems[0]
    assert claims[0] == {"kind": "use_more", "move": "hp", "range": "close", "when": "jumping", "why": "lands"}
    for bad in (None, [], "x", {"claims": "hp"}, {"claims": [1]}):
        assert parse_claims(bad)[0] == [] and parse_claims(bad)[1]
    claims, _ = parse_claims({"claims": [{"kind": "use_more", "move": "hp", "range": "", "when": "none"}]})
    assert claims[0]["range"] is None and claims[0]["when"] is None       # empty / "none" mean no condition
