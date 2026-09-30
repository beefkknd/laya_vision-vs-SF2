"""The character prompt (sf2.system2.character_prompt): the lesson loop narrowed to one opponent and one moment -
what he does that hurts her ("his threats"), and "if you see X": her answers in each of his situations, what works and
what does not. Owner, 2026-09-29: "what opponent specific move always win against me, or, if I see X, what works, what
doesn't work". The logs cannot name his move (a CPU special is logged as a plain attack), text laya is not retrained:
Qwen says it in the grammar's words (docs/qwen_learning.md, the gap)."""
import json
import os

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


def test_two_runs_against_one_opponent_keep_separate_console_logs():
    """2026-09-29: six runs at once, two per opponent, wrote into the same logs/qwen_lessons/ken_loop.log."""
    q = qwen_lessons()
    a, b = (q.log_dir(os.path.join("rollouts", "x", s)) for s in ("20260929-170523_ken", "20260929-170526_ken"))
    assert a != b and a.startswith(os.path.join("logs", "qwen_lessons"))


def test_the_fgc_variant_names_each_situation_the_way_players_do():
    """Owner 2026-09-29: phrase Qwen the way the fighting-game community describes SF2 (docs/qwen_learning.md 0g)."""
    assert C.term("jumping", "mid") == "anti-air" and C.term("attacking", "far") == "his zoning (fireball)"
    assert C.term("standing", "mid") == "footsies"
    blocks = C.if_you_see(world(), MOVES, terms=True)
    assert blocks[0].startswith("Up close, when he jumps - anti-air")
    assert all(not b.startswith("Up close, when he jumps -") for b in C.if_you_see(world(), MOVES))   # plain stays


def test_the_fgc_primer_carries_only_this_opponents_notes():
    ken = C.messages_fgc("chunli", "ken", [], world(), world(), [], [], MOVES)[0]["content"]
    honda = C.messages_fgc("chunli", "honda", [], world(), world(), [], [], MOVES)[0]["content"]
    assert "Shoryuken" in ken and "Headbutt" not in ken
    assert "Headbutt" in honda and "Shoryuken" not in honda
    assert "code checks" in ken.lower() and "arcade" in ken                    # hypotheses, not facts
    body = ken[ken.index('{"answer"'):]
    assert set(json.loads(body.replace("|", "_"))) == {"answer", "stop"}


def test_the_loop_knows_the_fgc_prompt():
    q = qwen_lessons()
    assert q.PROMPTS["character_fgc"].messages is C.messages_fgc and q.PROMPTS["character_fgc"].VIEWS == C.VIEWS


# His moves (sf2/system1/opp_moves.py): new runs log opp_move per decision; the threats name it when known and keep the
# old wording when not. The lesson grammar does not change.
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "system2", "character_prompt_golden.json")
GOLDEN_MOVES = ["c.mk", "sweep", "c.hp", "c.lk", "forward", "back", "hp", "mp", "block_high", "block_low",
                "spinning_bird_kick", "throw"]


def moved(a, move):
    return dict(a, opp_move=move, opp_shot=0)


def test_his_threat_names_his_move_when_it_is_known():
    far = act("back", "far", taken=10, reaction=("stand",))
    assert C.threat(moved(far, "fireball")) == "hits with a fireball"
    assert C.threat(moved(act("sweep", "close", taken=9, reaction=("attack",)), "uppercut")) == "hits with an uppercut"
    assert C.threat(moved(act("sweep", "close", taken=9, reaction=("attack",)), "hurricane")) == \
        "hits with a hurricane kick"
    assert C.threat(moved(act("sweep", "mid", taken=9, reaction=("attack",)), "slap")) == \
        "hits with the hundred hand slap"
    assert C.threat(moved(act("sweep", "close", taken=9, reaction=("attack",)), "throw")) == "throws her"
    assert C.threat(moved(act("sweep", "close", taken=9, air=True), "jump_attack")) == "jumps in"
    assert C.threat(moved(far, "fireball")) != C.threat(far)
    assert C.threat(moved(act("sweep", "close", taken=0), "fireball")) is None             # took nothing


def test_a_normal_or_no_move_or_a_missing_field_keeps_the_old_wording():
    for a in (act("sweep", "close", taken=5, reaction=("stand", "attack")), act("back", "far", taken=5),
              act("sweep", "close", taken=5, air=True, reaction=("jump",))):
        for move in ("normal", "none", None, "", "bogus"):
            b = dict(a, opp_move=move) if move is not None else a
            assert C.threat(b) == C.threat(a)


def test_the_threats_list_says_his_moves_at_their_range():
    rows = ([moved(act("back", "far", taken=15), "fireball") for _ in range(4)]
            + [moved(act("sweep", "close", taken=20, reaction=("attack",)), "uppercut") for _ in range(2)]
            + [act("hp", "mid", taken=8, state="attack", reaction=("attack",))])
    lines = C.threats(rows)
    assert lines[0].startswith("- he hits with a fireball far away: 4 times, 60 damage")
    assert lines[1].startswith("- he hits with an uppercut up close: 2 times, 40 damage")
    assert lines[2].startswith("- he attacks on the ground at mid range: 1 times")


def test_the_prompt_says_how_to_put_a_named_move_into_a_lesson_only_when_moves_are_named():
    rows = [moved(act("back", "far", taken=15), "fireball") for _ in range(12)]
    user = C.messages("chunli", "ryu", [], rows, rows, [], [], MOVES)[1]["content"]
    assert "he hits with a fireball far away" in user and C.MOVE_NOTE in user
    plain = [act("back", "far", taken=15) for _ in range(12)]
    assert C.MOVE_NOTE not in C.messages("chunli", "ryu", [], plain, plain, [], [], MOVES)[1]["content"]
    assert "when he attacks" in C.MOVE_NOTE                          # the grammar words, not a new word


def test_golden_the_plain_character_prompt_is_byte_identical_without_opp_move():
    """Pinned at 235d470 on 400 of the lock lesson_loop_v1's decisions per opponent (fields the prompt reads). Since
    2026-09-30 forward is not offered for lessons (the default); the old prompt is forward_lessons=True, and
    tests/test_forward_prompts.py pins that only the move lists differ."""
    with open(GOLDEN) as f:
        g = json.load(f)
    for opp, rows in g["rows"].items():
        assert "opp_move" not in rows[0]
        user = C.messages("chunli", opp, [], rows, rows[-60:], [], [], GOLDEN_MOVES, forward_lessons=True)[1]["content"]
        assert user == g["text"][opp], opp


def test_golden_on_the_whole_locked_play_data(monkeypatch):
    """Both plain and fgc, on every locked decision per opponent (sha256 pinned at 235d470); skipped without the lock."""
    import hashlib
    import pytest
    from sf2.eval import lock as lk
    data = os.environ.get("SF2_DATA", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if not os.path.isdir(os.path.join(data, "locks", "lesson_loop_v1", "artifacts")):
        pytest.skip("no lock lesson_loop_v1 under %s" % data)
    monkeypatch.setattr(lk, "ROOT", os.path.join(data, "locks"))
    with open(GOLDEN) as f:
        full = json.load(f)["full"]
    for opp in ("ken", "ryu", "honda"):
        rows = lk.play_rows("lesson_loop_v1", "chunli", opp)
        for fgc in (False, True):
            m = C.messages("chunli", opp, [], rows, rows[-300:], [], [], GOLDEN_MOVES, fgc=fgc, forward_lessons=True)
            assert [len(rows), hashlib.sha256(json.dumps(m).encode()).hexdigest()] == full["%s_%s" % (opp, fgc)]
