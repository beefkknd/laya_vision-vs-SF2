"""T1+T2: the ONE shared two-stage training dataset for text laya, covering ALL 8 characters.

What these pin (every one was seen RED before T1/T2 landed — see the module's git history / the T1T2 report):
  * all-8 coverage: every fighter's own menu is used; every category is reachable as a round-1 answer;
  * CONDITION-OFF teaches the default: a positive line whose condition is off -> category=block, move=block_high,
    for EVERY character;
  * the THROW ALIAS: a generic "throw"/"grab" in a lesson now resolves to the throw category and the concrete
    move throw_F+hp (it used to be inert -> no move);
  * default rows -> block_high; block is always an offered option in BOTH rounds;
  * the two shortcut checks (a metadata-only predictor <= majority + 0.05) for the category AND the move question;
  * old Chun-Li two-stage labels are unchanged where conditions are on (categories_for("chunli") == action_menu).

These import the generator by file (it lives under scripts/, imports _path), same as tests/test_g1_two_stage.py.
"""
import collections
import importlib.util
import os
import sys

import pytest

from sf2.system1.action_menu import CATEGORIES, CATEGORY_ORDER, DEFAULT_MOVE, category_of
from sf2.system1.advice import available_moves, moves_in_stance, parse, two_stage
from sf2.vocab import FIGHTERS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _build():
    if os.path.join(HERE, "scripts") not in sys.path:
        sys.path.insert(0, os.path.join(HERE, "scripts"))
    spec = importlib.util.spec_from_file_location("build_advice_data", os.path.join(HERE, "scripts",
                                                                                    "build_advice_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _build()
SPLITS = MOD.build(per_char=120, seed=7)
ROWS = [r for rs in SPLITS.values() for r in rs]
CAT_ROWS = [r for r in ROWS if r["round"] == "cat"]
MOVE_ROWS = [r for r in ROWS if r["round"] == "move"]


# ---------------------------------------------------------------- the per-character menu
def test_chunli_categories_equal_action_menu_exactly():
    # backward compat: the module default (Chun-Li) is byte-identical to the hand-written action_menu
    assert MOD.char_categories("chunli") == CATEGORIES


@pytest.mark.parametrize("char", FIGHTERS)
def test_every_character_has_all_seven_categories_and_block(char):
    cats = MOD.char_categories(char)
    assert set(cats) == set(CATEGORY_ORDER)
    assert cats["block"] == ["block_high", "block_low"]
    assert cats["special"], "%s has no specials" % char
    # move / punch / kick are the shared, character-agnostic normals
    for shared in ("move", "punch", "kick"):
        assert cats[shared] == CATEGORIES[shared]


def test_specials_are_each_characters_own():
    # a spot-check that the move NAMES are the real per-character ones, not Chun-Li's for everyone
    assert "sonic_boom" in MOD.char_categories("guile")["special"]
    assert "yoga_fire" in MOD.char_categories("dhalsim")["special"]
    assert "hadoken_hp" in MOD.char_categories("ryu")["special"]
    assert MOD.char_categories("guile")["special"] != MOD.char_categories("chunli")["special"]


# ---------------------------------------------------------------- all-8 coverage in the data
def test_all_eight_characters_appear_in_every_split():
    for split, rows in SPLITS.items():
        assert {r["me"] for r in rows} == set(FIGHTERS), split


def test_every_category_is_reachable_as_a_round1_answer():
    seen = {c for r in CAT_ROWS for c in r["answers"]}
    assert seen == set(CATEGORY_ORDER), sorted(set(CATEGORY_ORDER) - seen)


def test_each_characters_own_special_moves_show_up_as_round2_options():
    by_char = collections.defaultdict(set)
    for r in MOVE_ROWS:
        by_char[r["me"]].update(r["question"]["criteria"])
    for char in FIGHTERS:
        specials = set(MOD.char_categories(char)["special"])
        assert specials & by_char[char], (char, specials)


# ---------------------------------------------------------------- condition-off / default -> block, per character
def test_condition_off_rows_label_to_block_for_every_character():
    seen = set()
    for r in ROWS:
        if r["case"] == "condition_off":
            assert r["rule"] == "default", r["id"]
            if r["round"] == "cat":
                assert r["answers"] == ["block"], r["id"]
            else:
                assert r["answers"] == [DEFAULT_MOVE], r["id"]
            seen.add(r["me"])
    assert seen == set(FIGHTERS), sorted(set(FIGHTERS) - seen)


def test_default_rows_target_block_high():
    move_defaults = [r for r in MOVE_ROWS if r["rule"] == "default"]
    assert move_defaults
    for r in move_defaults:
        assert r["answers"] == [DEFAULT_MOVE] == ["block_high"], r["id"]


def test_block_is_always_offered_in_both_rounds():
    for r in CAT_ROWS:
        assert "block" in r["question"]["criteria"], r["id"]
    for r in MOVE_ROWS:
        assert "block_high" in r["question"]["criteria"], r["id"]


def test_condition_off_is_heavy():
    # the lever: >= 40% of rows teach the default (block) via an off / absent / negated / empty rule
    block_target = sum(r["rule"] == "default" for r in ROWS) / len(ROWS)
    assert block_target >= 0.40, block_target
    cond_off = sum(r["case"] == "condition_off" for r in ROWS) / len(ROWS)
    assert cond_off >= 0.20, cond_off


# ---------------------------------------------------------------- the throw alias
def test_generic_throw_now_resolves_to_a_concrete_throw_move():
    moves = [m for cat in CATEGORY_ORDER for m in CATEGORIES[cat]]
    for text in ("always throw him up close", "grab him", "go for a throw up close", "throw"):
        les = parse(text, moves)
        assert les.move == "throw_F+hp", (text, les.move)
    # polarity still read from the words
    assert parse("avoid throwing him", moves).polarity == "neg"
    assert parse("always throw him", moves).polarity == "hard"


def test_generic_throw_was_inert_without_the_alias_is_now_live_in_two_stage():
    moves = [m for cat in CATEGORY_ORDER for m in CATEGORIES[cat]]
    les = [parse("always throw him up close", moves)]
    cats, mv, rule = two_stage("close", "standing", "close", les)
    assert (cats, mv, rule) == (["throw"], ["throw_F+hp"], "hard")


def test_dataset_contains_a_followed_generic_throw_lesson():
    hits = [r for r in MOVE_ROWS if r["answers"] == ["throw_F+hp"]
            and any(MOD.HAS_GENERIC_THROW(line) for line in r["lessons"])]
    assert hits, "no generic-throw lesson resolved to throw_F+hp in the data"


# ---------------------------------------------------------------- the shortcut (no-leakage) checks
def test_metadata_only_predictor_is_near_chance_for_both_rounds():
    cat_chance, cat_meta, move_chance, move_meta = MOD.shortcut_scores(SPLITS)
    assert cat_meta <= cat_chance + 0.05, ("cat", cat_chance, cat_meta)
    assert move_meta <= move_chance + 0.05, ("move", move_chance, move_meta)


# ---------------------------------------------------------------- backward compatibility (conditions ON)
def test_old_chunli_labels_unchanged_where_the_condition_holds():
    ch = MOD.char_categories("chunli")
    cases = [
        ("far", "standing", "standing", ["use more s.mk"], (["kick"], ["s.mk"], "soft")),
        ("close", "standing", "close", ["always use cl.hp"], (["punch"], ["cl.hp"], "hard")),
        ("mid", "standing", "standing", ["he jumps a lot at far"], (["block"], [DEFAULT_MOVE], "default")),
    ]
    for rng, doing, stance, texts, want in cases:
        les = [parse(t, [m for c in CATEGORY_ORDER for m in CATEGORIES[c]]) for t in texts]
        # default categories (no arg) and explicit chunli categories must agree, and match the pre-existing answer
        assert two_stage(rng, doing, stance, les) == want
        assert two_stage(rng, doing, stance, les, categories=ch) == want


def test_held_out_wordings_only_in_test_split():
    for split in ("train", "val"):
        assert all(r["words"] == "trained" for r in SPLITS[split]), split
    assert any(r["words"] == "held_out" for r in SPLITS["test"])
