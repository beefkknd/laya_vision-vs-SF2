"""G1: the two-stage menu. Text laya chooses from the UNRATED action menu (sf2.system1.action_menu) by following
Qwen's advice lines, not table ratings: round 1 a category, round 2 the move inside it, the stance having pruned
the options. The default when nothing applies is block (action_menu.DEFAULT_MOVE), never walk."""
import collections
import importlib.util
import os
import sys

import pytest

from sf2.system1.action_menu import CATEGORIES, CATEGORY_ORDER, DEFAULT_MOVE, category_of
from sf2.system1.advice import (available_moves, category_question, chosen_moves, move_question, moves_in_stance,
                                 parse, stance_of, two_stage)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOVES = [m for cat in CATEGORY_ORDER for m in CATEGORIES[cat]]


def les(*texts):
    return [parse(t, MOVES) for t in texts]


# ---------------------------------------------------------------- stance pruning (round-2 option universe)
@pytest.mark.parametrize("stance,prefixes", [
    ("standing", ("s.",)), ("close", ("cl.",)), ("crouch", ("c.",)), ("air", ("j.", "jf.")),
])
def test_normals_are_pruned_to_the_stance(stance, prefixes):
    for cat in ("punch", "kick"):
        got = moves_in_stance(cat, stance)
        assert got and all(m.split(".")[0] + "." in prefixes for m in got)
        # every normal of the category that this stance cannot do is gone
        assert set(got) == {m for m in CATEGORIES[cat] if m.split(".")[0] + "." in prefixes}


def test_throw_only_up_close_and_grounded_moves_only_grounded():
    assert moves_in_stance("throw", "close") == CATEGORIES["throw"]
    for s in ("standing", "crouch", "air"):
        assert moves_in_stance("throw", s) == []
    for s in ("standing", "close", "crouch"):
        assert moves_in_stance("block", s) == CATEGORIES["block"]
        assert moves_in_stance("special", s) == CATEGORIES["special"]
    assert moves_in_stance("block", "air") == [] and moves_in_stance("special", "air") == []


def test_combos_are_air_only():
    assert set(moves_in_stance("combo", "air")) == set(CATEGORIES["combo"])
    for s in ("standing", "close", "crouch"):
        assert moves_in_stance("combo", s) == []


def test_stance_of_mapping():
    assert stance_of("air", "close") == "air" and stance_of("crouch", "mid") == "crouch"
    assert stance_of("stand", "close") == "close"
    assert stance_of("stand", "mid") == "standing" and stance_of("stand", "far") == "standing"
    with pytest.raises(ValueError):
        stance_of("flying", "mid")


# ---------------------------------------------------------------- the two-stage label rule
def test_soft_line_picks_its_category_and_move():
    cats, moves, rule = two_stage("far", "standing", "standing", les("use more s.mk"))
    assert (cats, moves, rule) == (["kick"], ["s.mk"], "soft")


def test_hard_line_beats_everything():
    cats, moves, rule = two_stage("close", "standing", "close", les("always use cl.hp"))
    assert (cats, moves, rule) == (["punch"], ["cl.hp"], "hard")


def test_default_is_block_not_walk():
    # no line applies -> the hardcoded default move, category "block"
    cats, moves, rule = two_stage("mid", "standing", "standing", les("he jumps a lot at far"))
    assert rule == "default" and moves == [DEFAULT_MOVE] and cats == [category_of(DEFAULT_MOVE)] == ["block"]
    assert two_stage("mid", "standing", "standing", [])[1] == [DEFAULT_MOVE]


def test_negative_rules_a_move_out_back_to_default():
    cats, moves, rule = two_stage("far", "standing", "standing", les("use more s.mk", "avoid s.mk"))
    assert rule == "default" and moves == [DEFAULT_MOVE]


def test_line_whose_move_this_stance_cannot_do_is_ignored():
    # c.mk is a crouch kick; while standing it is not on the menu, so the advice cannot apply -> default
    assert two_stage("far", "standing", "standing", les("use more c.mk"))[2] == "default"
    # but crouching it is available and followed
    assert two_stage("far", "standing", "crouch", les("use more c.mk")) == (["kick"], ["c.mk"], "soft")


def test_condition_that_does_not_hold_is_ignored():
    assert two_stage("mid", "standing", "standing", les("use more s.mk up close"))[2] == "default"
    assert two_stage("close", "standing", "close", les("use more cl.mk up close")) == (["kick"], ["cl.mk"], "soft")


def test_chosen_moves_matches_two_stage():
    assert chosen_moves("far", "standing", "standing", les("always use s.lp")) == (["s.lp"], "hard")


# ---------------------------------------------------------------- the question shapes (unrated menus)
def test_category_question_offers_the_seven_unrated_categories():
    q = category_question()
    assert q["type"] == "choice"
    assert list(q["criteria"]) == CATEGORY_ORDER and len(q["criteria"]) == 7
    assert all(k == v for k, v in q["criteria"].items())          # unrated: the option text IS the name


def test_move_question_is_the_pruned_unrated_moves():
    q = move_question(moves_in_stance("kick", "crouch"))
    assert set(q["criteria"]) == set(moves_in_stance("kick", "crouch"))
    assert all(k == v for k, v in q["criteria"].items())
    assert list(move_question([]) ["criteria"]) == [DEFAULT_MOVE]  # empty prune -> the default move is offered


# ---------------------------------------------------------------- the data builder's labels
def _build():
    if os.path.join(HERE, "scripts") not in sys.path:
        sys.path.insert(0, os.path.join(HERE, "scripts"))      # so the script's `import _path` resolves
    spec = importlib.util.spec_from_file_location("build_advice_data", os.path.join(HERE, "scripts",
                                                                                    "build_advice_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_built_rows_have_correct_two_stage_labels_and_valid_options():
    mod = _build()
    splits = mod.build(per_char=80, seed=1)
    rows = splits["train"] + splits["test"]
    assert rows
    seen_default_block = False
    for r in rows:
        opts = list(r["question"]["criteria"])
        assert r["answers"], r["id"]
        assert set(r["answers"]) <= set(opts), (r["id"], r["answers"], opts)   # encode() needs the answer offered
        if r["round"] == "cat":
            assert opts == mod.CATEGORY_ORDER
            assert all(a in mod.CATEGORY_ORDER for a in r["answers"])
        else:
            assert all(mod.category_of(a) == mod.category_of(r["answers"][0]) for a in r["answers"])
            if r["rule"] == "default":
                assert r["answers"] == [mod.DEFAULT_MOVE]
                seen_default_block = True
    assert seen_default_block          # the default-to-block case is exercised


def test_built_rows_come_in_cat_move_pairs_and_both_rounds_appear():
    mod = _build()
    splits = mod.build(per_char=80, seed=2)
    rounds = {r["round"] for r in splits["train"]}
    assert rounds == {"cat", "move"}
    n = collections.Counter(r["round"] for r in splits["train"])
    assert n["cat"] == n["move"]                       # one of each per decision
