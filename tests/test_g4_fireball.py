"""G4: the fireball condition in the grammar and in the screen words. A lesson can condition on an incoming fireball
("... when a fireball comes ..."); the reader puts the SAME token into the situation sentence when it sees a fireball
coming, so the lesson applies only then. The token is the word "fireball" (advice.FIREBALL_CLAUSE / advice.FIRE)."""
from dataclasses import replace

import pytest

from sf2.screen.facts import FighterFacts, HudFacts, ProjectileFacts, ScreenFacts
from sf2.system1 import screen_words as W
from sf2.system1.advice import FIRE, FIREBALL_CLAUSE, parse, read, situation_text, two_stage
from sf2.system1.action_menu import CATEGORIES, CATEGORY_ORDER, DEFAULT_MOVE

MOVES = [m for cat in CATEGORY_ORDER for m in CATEGORIES[cat]]
OLD_MOVES = ["lp", "mp", "c.mk", "sweep", "spinning_bird_kick", "block_low", "forward"]


# ---------------------------------------------------------------- the grammar
def test_a_fireball_lesson_parses_with_its_conditions():
    les = parse("block_low when a fireball comes at mid", MOVES)
    assert (les.move, les.polarity, les.where, les.when, les.fireball) == ("block_low", "soft", "mid", None, True)


@pytest.mark.parametrize("text", ["use block_low against a fireball", "when a fireball is coming, use block_high",
                                  "block_low when a fireball approaches"])
def test_fireball_phrasings_all_set_the_flag(text):
    assert parse(text, MOVES).fireball is True


def test_an_old_non_fireball_lesson_still_parses():
    les = parse("no sweep up close: it misses", OLD_MOVES)
    assert (les.move, les.polarity, les.where, les.fireball) == ("sweep", "neg", "close", False)
    # and the new-vocab plain lesson has no fireball condition
    assert parse("always use cl.hp", MOVES).fireball is False


def test_read_falls_back_without_a_fireball_flag():
    les = read("use lp up close and at far", OLD_MOVES)   # unreadable (two ranges) -> inert, fireball False
    assert les.move is None and les.fireball is False


def test_applies_requires_a_fireball_only_when_the_lesson_asks_for_one():
    fb = parse("use block_low when a fireball comes", MOVES)
    plain = parse("use block_low", MOVES)
    assert fb.applies("mid", "standing", fireball=True)
    assert not fb.applies("mid", "standing", fireball=False)
    assert plain.applies("mid", "standing", fireball=False) and plain.applies("mid", "standing", fireball=True)


def test_two_stage_follows_a_fireball_lesson_only_when_a_fireball_is_present():
    lessons = [parse("use block_low when a fireball comes", MOVES)]
    assert two_stage("mid", "standing", "standing", lessons, fireball=True) == (["block"], ["block_low"], "soft")
    cats, moves, rule = two_stage("mid", "standing", "standing", lessons, fireball=False)
    assert rule == "default" and moves == [DEFAULT_MOVE]


# ---------------------------------------------------------------- the situation sentence (screen words)
def _fighter(side, char, player, x, health, facing="right"):
    return FighterFacts(side, char, True, x, 190, False, facing, "x/y", "stand", 0.9, False, None, health, player)


def _facts(projectiles=()):
    me = _fighter("left", "chunli", 1, 100, 1.0)
    him = _fighter("right", "ryu", 2, 160, 1.0, facing="left")
    return ScreenFacts(me, him, tuple(projectiles), HudFacts((1.0, 1.0), 99, (False, False)), "fighting", 60)


def _proj(owner_side):
    return ProjectileFacts(130, 150, owner_side, "ryu/fireball", 0.9)


def test_says_fireball_only_for_an_incoming_shot():
    assert not W.says_fireball(_facts())                              # no projectile
    assert W.says_fireball(_facts([_proj("right")]))                 # the opponent (right) threw it
    assert W.says_fireball(_facts([_proj(None)]))                    # unattributed: treat as a threat
    assert not W.says_fireball(_facts([_proj("left")]))              # my own shot (I am player 1, left)


def test_sentence_says_the_fireball_only_when_one_is_present():
    plain = W.sentence(W.moment(_facts()))
    fiery = W.sentence(W.moment(_facts([_proj("right")])))
    assert FIREBALL_CLAUSE not in plain and "fireball" not in plain
    assert fiery.endswith(FIREBALL_CLAUSE) and fiery.startswith(plain)


def test_the_token_is_identical_in_both_places():
    # the clause the reader emits is the very token the grammar reads back
    assert FIRE.search(FIREBALL_CLAUSE)
    fiery = W.sentence(W.moment(_facts([_proj("right")])))
    assert FIRE.search(fiery)
    assert parse("block_low " + FIREBALL_CLAUSE.lower().rstrip("."), MOVES).fireball is True


def test_situation_text_fireball_flag_matches_the_constant():
    assert situation_text("mid", "standing", "full", "full", fireball=True).endswith(FIREBALL_CLAUSE)
    assert not situation_text("mid", "standing", "full", "full").endswith(FIREBALL_CLAUSE)
