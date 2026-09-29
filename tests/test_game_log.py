"""System 1's game log (sf2.system1.game_log): one entry per decision, with what happened after it."""
from ram_rows import row


def test_game_log_puts_the_opponents_reaction_on_the_action_that_caused_it():
    from sf2.system1.game_log import action_entry, clock
    def raw(p1=None, p2=None, timer=0x87):
        r = row(p1=p1, p2=p2, timer=timer)
        return r
    before = raw()
    rows = [raw(p1={"state": 0x0A}), raw(p1={"state": 0x0A}, p2={"state": 0x0A}),           # I whiff, he attacks
            raw(p1={"state": 0x0E, "react": 0x00, "life": 150}, p2={"state": 0x0A}),         # he hits me
            raw(p1={"state": 0x0E, "life": 150}), raw(p1={"life": 150})]
    d = {"action": "hp", "p_hit": 0.9, "predicted": "hit", "probs": {"hp": 0.9, "lp": 0.2}}
    e = action_entry(3, 120, "ryu", "ken", before, rows, d, "whiff")
    assert (e["game"], e["clock"], e["me"], e["opp"], e["side"]) == (3, 87, "ryu", "ken", "left")
    assert e["actual"] == "whiff" and e["dealt"] == 0
    assert e["i_was_hit"] and e["taken"] == 26 and e["opp_attacked"] and not e["opp_blocked"]
    assert e["opp_reaction"] == ["stand", "attack", "stand"] and e["my_life_after"] == 150
    assert clock(0x99) == 99 and e["top3"][0] == ["hp", 0.9]
