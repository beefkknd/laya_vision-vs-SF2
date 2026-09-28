"""The label rule text laya is trained on (sf2.advice): every polarity word, every condition, and the fallbacks."""
import pytest

from sf2.advice import FAILS, FORWARD, MAY, WORKS, answers, parse, situation_text

MOVES = ["lp", "mp", "c.mk", "mk", "sweep", "spinning_bird_kick", "block_high", "block_low", FORWARD]
OPTS = {"lp": MAY, "sweep": WORKS, "spinning_bird_kick": FAILS, FORWARD: None}


def pick(text, rng="close", doing="standing", opts=OPTS):
    return answers(rng, doing, opts, [parse(t, MOVES) for t in ([text] if isinstance(text, str) else text)])


@pytest.mark.parametrize("text", ["use more lp", "prefer lp", "go for lp", "lp works, do it more"])
def test_soft_positive_picks_the_move(text):
    assert pick(text) == (["lp"], "soft")


@pytest.mark.parametrize("text", ["avoid sweep", "never use sweep", "stop using sweep", "use less sweep",
                                  "fewer sweep", "don't sweep: it whiffs", "do not use sweep"])
def test_negative_rules_the_move_out(text):
    got, why = pick(text)
    assert "sweep" not in got and got == ["lp"] and why == "vision"


def test_negative_of_the_only_working_move_walks_in():
    assert pick("avoid sweep", opts={"sweep": WORKS, "lp": FAILS, FORWARD: None}) == ([FORWARD], "walk")


def test_hard_positive_beats_a_failing_rating():
    assert pick("always use spinning_bird_kick") == (["spinning_bird_kick"], "hard")


def test_soft_positive_yields_to_likely_fails():
    assert pick("use more spinning_bird_kick") == (["sweep"], "vision")


@pytest.mark.parametrize("text,rng,doing,want", [
    ("use more lp up close", "close", "standing", ["lp"]),
    ("use more lp up close", "mid", "standing", ["sweep"]),
    ("when he jumps, use lp", "close", "jumping", ["lp"]),
    ("when he jumps, use lp", "close", "standing", ["sweep"]),
    ("avoid sweep at mid range", "mid", "standing", ["lp"]),
    ("avoid sweep at mid range", "close", "standing", ["sweep"]),
    ("throw c.mk when he crouches", "close", "crouching", ["sweep"]),     # c.mk not offered: advice can't apply
])
def test_conditions(text, rng, doing, want):
    assert pick(text, rng, doing)[0] == want


def test_no_advice_takes_the_best_rating():
    assert answers("close", "standing", OPTS, []) == (["sweep"], "vision")


def test_ties_give_every_tied_move():
    opts = {"lp": WORKS, "sweep": WORKS, FORWARD: None}
    assert answers("close", "standing", opts, []) == (["lp", "sweep"], "vision")


def test_habit_changes_nothing():
    les = parse("he jumps a lot at far, be ready", MOVES)
    assert les.move is None and les.polarity == "none"
    assert pick("he jumps a lot at far, be ready", "far", "jumping") == (["sweep"], "vision")


def test_move_names_are_not_confused():
    assert parse("use more c.mk", MOVES).move == "c.mk"
    assert parse("use more mk", MOVES).move == "mk"


def test_stop_walking_forward():
    opts = {"lp": FAILS, "sweep": FAILS, FORWARD: None}
    assert pick("stop walking forward into him", opts=opts) == (["lp", "sweep"], "nothing_left")


def test_forward_must_be_offered():
    with pytest.raises(ValueError):
        answers("close", "standing", {"lp": WORKS}, [])


def test_bad_situation_rejected():
    with pytest.raises(ValueError):
        situation_text("near", "standing", "full", "full")


def test_conflicting_conditions_rejected():
    with pytest.raises(ValueError):
        parse("use lp up close and at far", MOVES)


@pytest.mark.parametrize("text,move,pol,where,when", [      # real lessons Qwen wrote (logs/system2/memory)
    ("no sweep up close: it misses", "sweep", "neg", "close", None),
    ("throw c.mk when he jumps at mid", "c.mk", "soft", "mid", "jumping"),
    ("stop throwing up close: he dodges it", "throw", "neg", "close", None),
    ("use mp up close: it lands and he does not punish it", "mp", "soft", "close", None),
    ("he jumps a lot at far, be ready", None, "none", "far", None),
    ("use lp when he is right next to you", "lp", "soft", "close", None),
    ("ryu punishes my sweep at mid range (42%)", "sweep", "neg", "mid", None),
    ("avoid spinning_bird_kick at close: punished 92% of the time", "spinning_bird_kick", "neg", "close", None),
])
def test_real_lessons(text, move, pol, where, when):
    les = parse(text, MOVES + ["throw", "jump"])
    assert (les.move, les.polarity, les.where, les.when) == (move, pol, where, when)


@pytest.mark.parametrize("text,when,where", [     # found by review: conditions Qwen writes other ways
    ("when ryu attacks, lp hits him", "attacking", None),
    ("when dhalsim jumps, use sweep", "jumping", None),
    ("use c.mk at mid: it lands when he stands", "standing", "mid"),
])
def test_conditions_named_opponent_or_after_the_colon(text, when, where):
    les = parse(text, MOVES)
    assert (les.when, les.where) == (when, where)
    assert pick(text, "close", "crouching")[1] != "soft"      # the condition does not hold: lesson ignored
