"""System 1 with a value checkpoint (docs/plan_laya_vision_value.md): the note version and the value questions come from
the checkpoint's config; the move is the best expected net over every choice, forward included; a checkpoint without
them (runs/all8) plays exactly as before."""
import numpy as np
import pytest

from sf2.system1.system1 import System1, _close, choices

IMG = np.zeros((224, 256, 3), np.uint8)
LEVELS = {"throw": 4, "forward": 2}      # throw: big gain, forward: even, everything else: loss


class FakeAgent:
    def __init__(self, cfg):
        self.cfg, self.seen = cfg, []

    def predict(self, state, questions):
        self.seen.append((state, dict(questions)))
        ans = {}
        for k, q in questions.items():
            if q["type"] == "score":
                lvl = LEVELS.get(k.split(":", 1)[1], 1)
                ans[k] = {"probabilities": {str(i): 1.0 if i == lvl else 0.0 for i in range(5)}}
            else:
                h = 0.9 if k == "lk" else 0.1   # P(hit) prefers lk: the old policy would play lk
                ans[k] = {"probabilities": {"hit": h, "whiff": 1 - h, "blocked": 0.0, "none": 0.0, "got_hit": 0.0}}
        return {"answers": ans}


def s1(cfg, explore=0.0):
    s = System1(None, "chunli", explore=explore)
    s._use(FakeAgent(cfg))
    return s


def test_old_checkpoint_plays_as_before():
    s = s1({})
    d = s.decide(IMG, IMG, "note")
    assert d["action"] == "lk" and "values" not in d and s.note_version == 1
    assert not any(k.startswith("value:") for k in s.agent.seen[0][1])


def test_value_checkpoint_picks_best_expected_net_incl_forward():
    s = s1({"note_version": 2, "value_questions": True})
    d = s.decide(IMG, IMG, "note")
    assert s.note_version == 2
    assert d["action"] == "throw" and d["values"]["throw"] > 40
    assert set(d["values"]) == set(choices("chunli"))
    assert d["probs"]["lk"] == 0.9            # P(hit)/P(blocked) still logged as scores
    assert d["p_hit"] == 0.1


def test_value_checkpoint_forward_when_everything_loses():
    s = s1({"note_version": 2, "value_questions": True})
    LEVELS_BAK = dict(LEVELS)
    try:
        LEVELS.clear()
        LEVELS["forward"] = 2
        assert s.decide(IMG, IMG, "note")["action"] == "forward"
    finally:
        LEVELS.clear()
        LEVELS.update(LEVELS_BAK)


def test_value_checkpoint_explore_still_logs_values():
    s = s1({"note_version": 2, "value_questions": True}, explore=1.0)
    d = s.decide(IMG, IMG, "note")
    assert d["explored"] is True and "throw" in d["values"]


def test_values_logged_in_entry():
    s = s1({"note_version": 2, "value_questions": True})
    r = {"p1_x": 100, "p2_x": 160, "p1_y": 192, "p2_y": 192, "p1_state": 0, "p2_state": 0, "p1_life": 176,
         "p2_life": 176, "p1_react": 0, "p2_react": 0, "timer": 0x99}
    d = dict(s.decide(IMG, IMG, "n"), prompt="n")
    assert _close(0, "chunli", "ryu", (r, [r], d, "hit", 0, None))["values"]["throw"] == d["values"]["throw"]


def test_bad_note_version_refused():
    with pytest.raises(ValueError):
        s1({"note_version": 7})


def test_value_questions_need_note_v2():
    with pytest.raises(ValueError):
        s1({"value_questions": True})


def test_value_checkpoint_refuses_a_short_memory_in_the_note():
    # review 2026-09-30: play_system1 defaults to --memory memory; the memory line would go into a note v2 the value
    # checkpoint never saw in training - refused, not silently off-distribution
    from sf2.system1.system1 import _decide
    s = s1({"note_version": 2, "value_questions": True})
    s.short = {"lessons": [{"text": "use more throw up close"}]}
    r = {"p1_x": 100, "p2_x": 160, "p1_y": 192, "p2_y": 192, "p1_state": 0, "p2_state": 0, "p1_life": 176,
         "p2_life": 176, "p1_react": 0, "p2_react": 0, "timer": 0x99}
    with pytest.raises(ValueError):
        _decide(s, "ryu", r, IMG, IMG)
    s.short = {"lessons": []}
    assert _decide(s, "ryu", r, IMG, IMG)[1].endswith("opp_attacking=0")
