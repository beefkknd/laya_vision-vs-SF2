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


def test_the_log_keeps_every_score_laya_vision_gave():
    """2026-09-30: only the top 3 were logged, so "rank by expected damage" (docs/component_boundaries.md) could not be
    replayed offline - the throw is outside the top 3 in 81-100% of close decisions."""
    from sf2.system1.game_log import action_entry
    before = row()
    rows = [row(), row()]
    probs = {"hp": 0.9, "lp": 0.2, "throw": 0.61, "sweep": 0.3, "block_low": 0.12}
    e = action_entry(0, 0, "chunli", "ken", before, rows, {"action": "hp", "p_hit": 0.9, "predicted": "hit",
                                                           "probs": probs}, "hit")
    assert e["scores"] == probs and list(e)[-1] == "scores"               # appended: the old keys keep their order
    assert e["top3"] == [["hp", 0.9], ["throw", 0.61], ["sweep", 0.3]]
