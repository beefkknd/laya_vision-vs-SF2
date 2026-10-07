"""Blanka becomes a PLAYABLE character: a RAM-free move menu in sf2.moves_free (owner 2026-10-06,
"get all of them unlocked"). Blanka: Electric Thunder (mash P, same shape as honda's Hundred Hand
Slap) + Rolling Attack (charge B->F+P), with Up-Ball (charge D->U+K) tried and kept only if it fires
on this World-Warrior ROM. This pins the menu STRUCTURE; firing is settled on screen (scratchpad
blanka_probe). Completes the eight-World-Warrior playable roster.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from sf2.moves_free import menu                                           # noqa: E402
from sf2.system1.advice import available_moves, char_categories, moves_in_stance  # noqa: E402


def test_blanka_menu_loads_and_has_the_shared_move_families():
    names = [m.name for m in menu("blanka")]
    assert "walk_forward" in names and "block_high" in names
    assert any(n.startswith("s.") for n in names) and any(n.startswith("c.") for n in names)
    assert any(n.startswith("throw_F+") for n in names)


def test_blanka_signature_specials_exist():
    cats = char_categories("blanka")
    assert set(cats) >= {"move", "punch", "kick", "block", "throw", "special", "combo"}
    assert "electricity" in cats["special"]            # mash-P signature
    assert "rolling_attack" in cats["special"]         # charge B->F+P


def test_stance_law_is_character_general_for_blanka():
    cats = char_categories("blanka")
    for stance in ("standing", "close", "crouch", "air"):
        assert isinstance(moves_in_stance("punch", stance, cats), list)
    assert available_moves("standing", cats)


def test_blanka_is_a_supported_player():
    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "play_loop_screen.py")
    src = open(src_path).read()
    assert '"blanka"' in src.split("SUPPORTED_ME")[1].split(")")[0]
