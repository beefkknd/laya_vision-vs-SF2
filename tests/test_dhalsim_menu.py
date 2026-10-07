"""Dhalsim becomes a PLAYABLE character: a RAM-free move menu in sf2.moves_free (owner 2026-10-06,
"the long arm/long leg guy, unique moves"). Dhalsim is the first ZONER player; his signature long
reach is his NORMALS (the stretch s./cl. punches & kicks the shared normals() presses), and his
special is Yoga Fire (QCF+P projectile). This pins the menu STRUCTURE (loads, 7 categories, stance
law, supported --me); which specials fire on this World-Warrior ROM is settled empirically on screen
(scratchpad/study/dhalsim_probe*). Yoga Flame (HCB+P) was DROPPED -- a clean half-circle-back never
registered as the special via scripted tokens on this ROM, so there is no honest descriptor to ship.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from sf2.moves_free import menu                                           # noqa: E402
from sf2.system1.advice import available_moves, char_categories, moves_in_stance  # noqa: E402


def test_dhalsim_menu_loads_and_has_the_shared_move_families():
    names = [m.name for m in menu("dhalsim")]
    assert "walk_forward" in names and "block_high" in names
    assert any(n.startswith("s.") for n in names) and any(n.startswith("c.") for n in names)   # the stretch normals
    assert any(n.startswith("throw_F+") for n in names)


def test_dhalsim_yoga_fire_exists_and_flame_is_not_shipped():
    cats = char_categories("dhalsim")
    assert set(cats) >= {"move", "punch", "kick", "block", "throw", "special", "combo"}
    assert "yoga_fire" in cats["special"]            # QCF+P projectile, verified on screen
    assert "yoga_flame" not in cats["special"]       # HCB+P never registered on this ROM -> dropped, not faked


def test_stance_law_is_character_general_for_dhalsim():
    cats = char_categories("dhalsim")
    for stance in ("standing", "close", "crouch", "air"):
        assert isinstance(moves_in_stance("punch", stance, cats), list)
    assert available_moves("standing", cats)


def test_dhalsim_is_a_supported_player():
    src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "play_loop_screen.py")
    src = open(src_path).read()
    assert '"dhalsim"' in src.split("SUPPORTED_ME")[1].split(")")[0]
