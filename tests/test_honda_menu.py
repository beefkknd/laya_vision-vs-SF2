"""E.Honda becomes a PLAYABLE character: a RAM-free move menu in sf2.moves_free (owner 2026-10-05, "train a new, unique
character"). Honda is the first non-shoto, non-chunli player, so this pins the menu's STRUCTURE (the invariants that
hold whatever the ROM supports) -- that it loads, the categories are right, and the character-general stance law
(fixed in 9ecca86/21ce807) does not crash on honda's moves. WHICH charge specials actually fire on this World-Warrior
ROM is settled empirically (a separate on-screen verification), so this test only pins honda's signature move
(hundred_hand_slap) + the shared normals/blocks/throws/movement, not the verify-dependent charge moves."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from sf2.moves_free import menu                                           # noqa: E402
from sf2.system1.advice import available_moves, char_categories, moves_in_stance  # noqa: E402


def test_honda_menu_loads_and_has_the_shared_move_families():
    names = [m.name for m in menu("honda")]
    assert "walk_forward" in names and "block_high" in names             # movement + blocks
    assert any(n.startswith("s.") for n in names) and any(n.startswith("c.") for n in names)   # normals
    assert any(n.startswith("throw_F+") for n in names)                   # throws


def test_honda_signature_move_is_a_special():
    cats = char_categories("honda")
    assert set(cats) >= {"move", "punch", "kick", "block", "throw", "special", "combo"}  # the 7 categories
    assert "hundred_hand_slap" in cats["special"]                         # the Hundred Hand Slap exists


def test_stance_law_is_character_general_for_honda():
    cats = char_categories("honda")
    # no crash on honda's moves (the Chun-Li-hardcoded category_of bug would raise here)
    for stance in ("standing", "close", "crouch", "air"):
        got = moves_in_stance("punch", stance, cats)
        assert isinstance(got, list)
    assert available_moves("standing", cats)                              # non-empty followable set


def test_honda_is_a_supported_player():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pls", os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "play_loop_screen.py"))
    # SUPPORTED_ME is a module constant; read it without executing argparse/main
    src = open(spec.origin).read()
    assert '"honda"' in src.split("SUPPORTED_ME")[1].split(")")[0]        # honda listed in SUPPORTED_ME
