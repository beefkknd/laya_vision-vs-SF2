"""System 2's writer: the prompt Qwen gets and how its reply becomes validated rules (docs/TWO_SYSTEM_PLAN.md)."""
from sf2 import contract as C
from sf2 import memory as M
from sf2 import system2 as S
from sf2.actions import ACTIONS

NOTE = ("me=guile stand hp=60 opp=ryu jump hp=90 dist=mid facing=right corner=none time=early last=forward "
        "fireball=none")


def _moment(i, why="surprised", taken=18):
    return C.moment_record(match=0, round=0, frame=100 + 4 * i, why=why, notes_before=[NOTE] * 3, note=NOTE,
                           probs={"jump_forward": 0.5, "hk": 0.2}, played="jump_forward", dealt=0, taken=taken,
                           frames=["a.png", "b.png"])


def test_the_prompt_describes_the_game_but_gives_no_tactics():
    msgs = S.build_messages([_moment(0)], rules=[], feedback=None)
    text = " ".join(m["content"] for m in msgs)
    for field in C.FIELDS:
        assert field in text
    for move in ACTIONS:
        assert move in text
    assert "->" in text and "(empty)" in text  # the rule format, and the memory starts empty
    for tactic in ("anti-air", "apex", "throw range", "crouch-guard"):
        assert tactic not in text.lower()


def test_the_prompt_carries_each_moment_and_the_last_batch_feedback():
    msgs = S.build_messages([_moment(0), _moment(1, why="audit", taken=0)], rules=[C.parse_rule("dist=far -> forward")],
                            feedback="batch 1: your rules changed 3% of moves; net damage per round -4.0 vs no memory")
    user = msgs[-1]["content"]
    assert "m0r0f100" in user and "m0r0f104" in user and "surprised" in user and "audit" in user
    assert "dist=far -> forward" in user and "net damage per round -4.0" in user


def test_parse_keeps_valid_rules_and_explains_the_rest():
    reply = """<think>Guile keeps jumping into Ryu's attacks... maybe dist=close -> sweep? no.</think>
Here are my rules:
1. `opp_state=jump dist=mid -> block`  # he jumps in; guard
- opp_state=attack dist=close -> back
opp_state=jump dist=near -> hp
dist=far -> fireball
some prose without an arrow
"""
    rules, rejected = S.parse_reply(reply)
    assert [str(r).split(" weight")[0] for r in rules] == ["opp_state=jump dist=mid -> block",
                                                          "opp_state=attack dist=close -> back"]
    assert rules[0].reason == "he jumps in; guard"
    assert len(rejected) == 2 and any("near" in why for _, why in rejected)
    assert all("->" in line for line, _ in rejected)  # prose without an arrow is not a rejected rule


def test_merge_adds_only_new_rules_to_qwen_txt(tmp_path):
    new = [C.parse_rule("opp_state=jump dist=mid -> block # guard"), C.parse_rule("dist=far -> forward")]
    assert S.merge(str(tmp_path), new) == 2
    again = [C.parse_rule("opp_state=jump dist=mid -> block # said differently"), C.parse_rule("dist=far -> jump")]
    assert S.merge(str(tmp_path), again) == 1  # the same situation and move is not added twice
    mem = M.load(str(tmp_path))
    assert len(mem.rules) == 3 and {r.author for r in mem.rules} == {"qwen"}


def test_a_later_rule_for_the_same_situation_replaces_the_earlier_one(tmp_path):
    S.merge(str(tmp_path), [C.parse_rule("dist=far -> forward")])
    S.merge(str(tmp_path), [C.parse_rule("dist=far -> jump_forward # revised")], replace=True)
    mem = M.load(str(tmp_path))
    assert [r.move for r in mem.rules] == ["jump_forward"]
