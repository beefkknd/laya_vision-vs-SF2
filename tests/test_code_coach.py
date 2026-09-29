"""The no-LLM coach: net hit points per (move, range) -> use-more / avoid lines text laya can read."""
from sf2.advice import parse
from sf2.code_coach import lines, memory, table

MOVES = ["lp", "sweep", "c.mk", "spinning_bird_kick", "forward"]


def atk(move, rng, dealt, taken, opp="ryu"):
    return {"action": move, "range": rng, "dealt": dealt, "taken": taken, "actual": "hit" if dealt else "whiff",
            "opp": opp, "kind": "attack", "me": "chunli"}


ROWS = ([atk("sweep", "mid", 10, 0)] * 30 + [atk("lp", "close", 2, 0)] * 30 + [atk("c.mk", "mid", 0, 8)] * 30 +
        [atk("spinning_bird_kick", "mid", 20, 0)] * 5 +                    # too few tries to judge
        [atk("sweep", "close", 0, 20)] * 30)                               # sweep is bad up close, good at mid


def test_best_used_worst_avoided_one_line_per_move():
    got = lines(table(ROWS))
    assert got == ["use more sweep at mid range", "use more lp up close", "avoid c.mk at mid range"]
    for line in got:                                                      # every line reads as text laya's rule
        les = parse(line, MOVES)
        assert les.move and les.where and les.polarity in ("soft", "neg")


def test_short_uses_this_opponent_playbook_the_others():
    rows = ROWS + [atk("lp", "far", 9, 0, opp="ken")] * 30
    assert memory("chunli", "ryu", "short", rows)["lessons"][0]["text"] == "use more sweep at mid range"
    assert [x["text"] for x in memory("chunli", "ryu", "playbook", rows)["lessons"]] == ["use more lp far away"]
