"""Rule-line -> registry entry building (what text-laya plays via the --carry file), test-first.

A Playbook's rules are raw advice-grammar lines; to run them they become verified registry entries
(the shape play_loop_screen's --carry loads). This mirrors the entry shape measure_rule already uses.
Unfollowable lines (no move / bare movement token) must be rejected loudly, not silently played.
"""
import pytest

from sf2.system1.advice import char_menu_moves
from sf2.system2.rule_entry import candidate_rules, carry_entries, claim_of, entry


def test_claim_of_parses_a_situational_line():
    moves = char_menu_moves("chunli")
    c = claim_of("use more throw up close", moves)
    assert c["kind"] == "use_more"
    assert c["move"] and c["range"] == "close"


def test_entry_is_a_verified_record_that_renders_to_its_line():
    moves = char_menu_moves("chunli")
    e = entry(claim_of("use more throw up close", moves))
    assert e["state"] == "verified"
    assert isinstance(e["line"], str) and "throw" in e["line"]
    assert set(("claim", "line", "state", "since", "evidence")) <= set(e)


def test_carry_entries_builds_a_small_playbook():
    moves = char_menu_moves("chunli")
    es = carry_entries(["use more throw up close", "avoid walk_forward at mid range"], moves)
    assert len(es) == 2
    assert all(e["state"] == "verified" for e in es)


def test_unfollowable_line_is_rejected():
    moves = char_menu_moves("chunli")
    with pytest.raises(Exception):
        claim_of("use more forward at mid range", moves)  # bare 'forward' is not a followable move token


def test_candidate_rules_applies_coach_claims_onto_incumbent():
    moves = char_menu_moves("chunli")
    incumbent = ("use more throw up close",)
    # a real-shaped Coach claim (answer): s.mk at mid when he attacks
    claim = claim_of("use more s.mk at mid range when he attacks", moves)
    cand = candidate_rules(incumbent, [claim], moves)
    assert cand[0] == "use more throw up close"
    assert any("s.mk" in r and "mid" in r for r in cand)
    assert len(cand) == 2


def test_candidate_rules_dedupes_an_already_present_rule():
    moves = char_menu_moves("chunli")
    incumbent = ("use more throw up close",)
    claim = claim_of("use more throw up close", moves)
    assert candidate_rules(incumbent, [claim], moves) == incumbent  # no duplicate added
