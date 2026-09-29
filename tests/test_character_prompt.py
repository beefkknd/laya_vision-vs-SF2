"""The character prompt (sf2.system2.character_prompt): the lesson loop narrowed to one opponent and one moment -
what he does that hurts her ("his threats"), and "if you see X": her answers in each of his situations, what works and
what does not. Owner, 2026-09-29: "what opponent specific move always win against me, or, if I see X, what works, what
doesn't work". The logs cannot name his move (a CPU special is logged as a plain attack), text laya is not retrained:
Qwen says it in the grammar's words (docs/qwen_learning.md, the gap)."""
import json

from sf2.system1.advice import read
from sf2.system2 import character_prompt as C
from sf2.system2 import lessons as L

MOVES = ["sweep", "c.mk", "hp", "block_high", "block_low", "forward"]


def act(move, rng, taken=0, dealt=0, air=False, state="stand", reaction=("stand",), kind="attack"):
    return {"action": move, "range": rng, "kind": kind, "dealt": dealt, "taken": taken, "opp_air": air,
            "opp_state": state, "opp_reaction": list(reaction), "actual": "hit" if dealt else "whiff",
            "opp_attacked": "attack" in reaction}


def world():
    """Up close when he jumps in: sweep loses 12, c.mk 2; at mid he attacks and hp loses 8; far nothing happens."""
    return ([act("sweep", "close", taken=12, air=True, reaction=("jump", "attack")) for _ in range(20)]
            + [act("c.mk", "close", taken=2, air=True, reaction=("jump", "stand")) for _ in range(20)]
            + [act("hp", "mid", taken=8, state="attack", reaction=("attack", "stand")) for _ in range(10)]
            + [act("sweep", "mid", dealt=4, state="attack", reaction=("hit_stun",)) for _ in range(10)]
            + [act("forward", "far", kind="movement") for _ in range(10)])


def test_his_threats_rank_his_moves_by_the_damage_they_did():
    lines = C.threats(world())
    assert lines[0].startswith("- he jumps in up close") and "280 damage (78%)" in lines[0]
    assert lines[1].startswith("- he attacks on the ground at mid range") and "80 damage (22%)" in lines[1]
    assert not any("far away" in x for x in lines)                         # he never hurt her from there


def test_a_jump_counts_as_the_threat_even_when_he_lands_an_attack():
    assert C.threat(act("sweep", "close", taken=5, reaction=("jump", "attack"))) == "jumps in"
    assert C.threat(act("sweep", "close", taken=5, air=True, reaction=("attack",))) == "jumps in"
    assert C.threat(act("sweep", "close", taken=5, reaction=("stand", "attack"))) == "attacks on the ground"
    assert C.threat(act("back", "close", taken=5, reaction=("stand",))) == "hits her from afar"      # a fireball
    assert C.threat(act("sweep", "close", taken=0, reaction=("attack",))) is None


def test_if_you_see_ranks_her_answers_in_each_of_his_situations():
    close = C.if_you_see(world(), MOVES)[0]
    assert close.startswith("Up close, when he jumps")
    assert close.index("c.mk") < close.index("sweep")                    # best answer first
    assert "never tried there: hp, block_high, block_low, forward" in close          # moves System 1 can pick, untried there


def test_every_situation_heading_uses_the_grammar_words():
    for block in C.if_you_see(world(), MOVES):
        head = block.splitlines()[0]
        assert any(w in head for w in L.WHEN_WORDS.values()), head


def test_the_answer_template_is_valid_json_in_both_modes():
    for stable in (None, "losing"):
        system = C.messages("chunli", "ken", [], world(), world(), [], [], MOVES, stable=stable)[0]["content"]
        body = system[system.index("{"):]
        keys = set(json.loads(body.replace("|", "_")))
        assert keys == ({"answer", "stop", "what_if"} if stable else {"answer", "stop"})


def test_the_prompt_names_the_gap_and_how_to_say_his_moves():
    system = C.messages("chunli", "ken", [], world(), world(), [], [], MOVES)[0]["content"]
    assert "fireball" in system and "when he attacks" in system and "far away" in system


def test_a_character_lesson_must_name_what_he_is_doing():
    claims, problems = C.parse_claims({"answer": {"kind": "always", "move": "block_high", "range": "close",
                                                  "when": "jumping"},
                                       "stop": {"kind": "avoid", "move": "sweep", "range": "close", "when": None}})
    assert [c["view"] for c in claims] == ["answer"] and "stop" in problems[0] and "what he" in problems[0]
    claims, problems = C.parse_claims({"attack": {"kind": "avoid", "move": "sweep"}})
    assert claims == [] and problems


def test_every_accepted_claim_reads_back_in_text_laya():
    for kind in L.KINDS:
        for when in L.WHEN_WORDS:
            claims, problems = C.parse_claims({"answer": {"kind": kind, "move": "c.mk", "range": "mid", "when": when}})
            les = read(L.render(claims[0]), MOVES)
            assert (les.move, les.when) == ("c.mk", when) and not problems


def qwen_lessons():
    from tests.test_lock import qwen_lessons as load
    return load()


def test_the_loop_asks_with_the_prompt_it_was_given(monkeypatch):
    q = qwen_lessons()
    seen = {}

    def chat(msgs, tag):
        seen["system"] = msgs[0]["content"]
        return '{"answer": {"kind": "always", "move": "block_high", "range": "close", "when": "jumping"}, "stop": null}'
    monkeypatch.setattr(q, "chat", chat)
    claims, problems, _ = q.ask("ken", [], world(), world(), [], [], [], prompt="character")
    assert "only against ken" in seen["system"] and claims[0]["view"] == "answer" and not problems
    claims, _, _ = q.ask("ken", [], world(), world(), [], [], [], prompt="views")
    assert "ATTACK" in seen["system"] and claims == []                        # the two-view prompt reads no "answer"


def test_the_chance_baseline_names_what_he_is_doing_like_a_character_lesson():
    q = qwen_lessons()
    assert all(c["when"] for c in q.random_claims(world(), 200, need_when=True))
    assert any(c["when"] is None for c in q.random_claims(world(), 200))
