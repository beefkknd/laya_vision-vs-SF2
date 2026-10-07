"""Guile becomes a PLAYABLE character: a RAM-free move menu in sf2.moves_free (owner 2026-10-06,
"get all of them unlocked"). Guile is a CHARGE character: Sonic Boom (charge B->F+P projectile) and
Flash Kick (charge D->U+K anti-air), reusing the charge timings proven on this World-Warrior ROM by
honda. This pins the menu STRUCTURE; which specials fire is settled on screen (scratchpad guile_probe).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from sf2.moves_free import menu                                           # noqa: E402
from sf2.system1.advice import available_moves, char_categories, moves_in_stance  # noqa: E402


def test_guile_menu_loads_and_has_the_shared_move_families():
    names = [m.name for m in menu("guile")]
    assert "walk_forward" in names and "block_high" in names
    assert any(n.startswith("s.") for n in names) and any(n.startswith("c.") for n in names)
    assert any(n.startswith("throw_F+") for n in names)


def test_guile_charge_specials_exist():
    cats = char_categories("guile")
    assert set(cats) >= {"move", "punch", "kick", "block", "throw", "special", "combo"}
    assert "sonic_boom" in cats["special"] and "flash_kick" in cats["special"]


def test_stance_law_is_character_general_for_guile():
    cats = char_categories("guile")
    for stance in ("standing", "close", "crouch", "air"):
        assert isinstance(moves_in_stance("punch", stance, cats), list)
    assert available_moves("standing", cats)


def test_guile_is_a_supported_player():
    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "play_loop_screen.py")
    src = open(src_path).read()
    assert '"guile"' in src.split("SUPPORTED_ME")[1].split(")")[0]
