"""The memory System 2 writes and System 1 consults (docs/TWO_SYSTEM_PLAN.md §2.2)."""
import pytest

from sf2 import contract as C
from sf2 import memory as M
from sf2.actions import ACTIONS

NOTE = ("me=chunli stand hp=80 opp=ryu jump hp=45 dist=mid facing=right corner=none time=early last=hk "
        "fireball=none")


def _probs(**kw):
    p = {a: 0.0 for a in ACTIONS}
    p.update(kw)
    return p


def _mem(tmp_path, **files):
    for name, text in files.items():
        (tmp_path / ("%s.txt" % name)).write_text(text)
    return M.load(str(tmp_path))


def test_empty_memory_never_changes_the_move(tmp_path):
    mem = M.load(str(tmp_path))  # no files at all
    move, info = mem.apply(NOTE, _probs(hk=0.6, hp=0.4), "hk")
    assert move == "hk" and info == {"rule": None, "author": None, "fired": False, "changed": False}


def test_a_rule_nudges_only_within_tau(tmp_path):
    mem = _mem(tmp_path, claude="opp_state=jump dist=mid -> hp\n")
    move, info = mem.apply(NOTE, _probs(hk=0.5, hp=0.3), "hk", tau=0.3)  # gap 0.2 <= 0.3
    assert move == "hp" and info["fired"] and info["changed"] and info["author"] == "claude"
    move, info = mem.apply(NOTE, _probs(hk=0.8, hp=0.1), "hk", tau=0.3)  # gap 0.7 > 0.3: laya keeps its move
    assert move == "hk" and info["fired"] and not info["changed"]


def test_a_rule_that_agrees_with_laya_fires_without_changing(tmp_path):
    mem = _mem(tmp_path, claude="opp_state=jump -> hp\n")
    move, info = mem.apply(NOTE, _probs(hp=0.9), "hp")
    assert move == "hp" and info["fired"] and not info["changed"]


def test_no_match_means_no_fire(tmp_path):
    mem = _mem(tmp_path, claude="opp_state=attack -> block\n")
    move, info = mem.apply(NOTE, _probs(hk=0.5, block=0.4), "hk")
    assert move == "hk" and not info["fired"]


def test_most_specific_rule_wins_then_author_then_weight(tmp_path):
    mem = _mem(tmp_path,
               owner="opp_state=jump -> block\n",
               claude="opp_state=jump dist=mid -> hp\n",
               qwen="opp_state=jump dist=mid -> lk weight=0.9\nopp=ryu opp_state=jump dist=mid -> hk\n")
    assert mem.pick(C.parse_note(NOTE)).move == "hk"  # three conditions beat two
    mem = _mem(tmp_path, owner="opp_state=jump dist=mid -> block\n", claude="opp_state=jump dist=mid -> hp\n",
               qwen="opp_state=jump dist=mid -> lk weight=0.9\nopp=dhalsim opp_state=jump dist=mid -> hk\n")
    assert mem.pick(C.parse_note(NOTE)).move == "block"  # a tie in specificity: the owner outranks Claude and Qwen


def test_author_comes_from_the_file_not_the_line(tmp_path):
    mem = _mem(tmp_path, qwen="opp_state=jump -> hp author=owner\n")
    assert mem.pick(C.parse_note(NOTE)).author == "qwen"


def test_files_other_than_the_three_authors_are_ignored(tmp_path):
    mem = _mem(tmp_path, notes="garbage that is not a rule\n", claude="opp_state=jump -> hp\n")
    assert len(mem.rules) == 1


def test_a_bad_line_names_its_file_and_line(tmp_path):
    with pytest.raises(ValueError, match=r"claude\.txt.*line 2"):
        _mem(tmp_path, claude="opp_state=jump -> hp\ndist=near -> hp\n")


def test_loop_guard_rejects_a_rule_that_repeats_its_own_last_move():
    with pytest.raises(ValueError, match="repeat"):
        C.parse_rule("last=lp dist=close -> lp")
    C.parse_rule("last=lp dist=close -> hp")  # a follow-up to a different move is fine


def test_each_rule_pushes_with_its_own_weight_by_default(tmp_path):
    mem = _mem(tmp_path, owner="opp_state=jump dist=mid -> hp weight=1\n",
               claude="opp_state=jump dist=close -> hp weight=0.2\n")
    move, info = mem.apply(NOTE, _probs(jump_forward=0.9, hp=0.02), "jump_forward")  # gap 0.88 <= weight 1
    assert move == "hp" and info["changed"]
    close = NOTE.replace("dist=mid", "dist=close")
    move, info = mem.apply(close, _probs(jump_forward=0.5, hp=0.2), "jump_forward")  # gap 0.3 > weight 0.2
    assert move == "jump_forward" and info["fired"] and not info["changed"]


def test_an_explicit_tau_overrides_every_rule_weight(tmp_path):
    mem = _mem(tmp_path, claude="opp_state=jump dist=mid -> hp weight=1\n")
    move, _ = mem.apply(NOTE, _probs(jump_forward=0.9, hp=0.02), "jump_forward", tau=0.3)
    assert move == "jump_forward"
