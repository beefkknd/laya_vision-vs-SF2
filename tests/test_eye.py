"""The U decision path (sf2.system1.eye; docs/prereg_u_perception.md): a perception checkpoint answers every question in
one predict per decision from the two frames (HUD visible) and "me=<char>" only - no RAM reaches it. Question 8 ->
per-move rank score P(likely works) + 0.5 P(may work), rating word = its argmax; without an advisor the best rank score
plays, or walking in when no move's word is better than "likely fails"; with one, text laya's shortlist is the top 3
by rank score (avoided moves skipped) + named moves + forward, rated with question 8's own words, and its situation is
laya-vision's own answers in text laya's words."""
import inspect
import json

import numpy as np
import pytest

from sf2.data import u_data as U
from sf2.data.perception import Q8_ANSWERS, QUESTIONS
from sf2.emu.vs import NAMES
from sf2.system1 import eye as E
from sf2.system1.advice import FAILS, FORWARD, MAY, OPP_STATES, WORKS, rating
from sf2.system1.advisor import shortlist as advisor_shortlist
from sf2.system1.system1 import System1, _close, _decide, choices
from sf2.vocab import BARS, RANGES

RAW = np.full((224, 256, 3), 7, np.uint8)
ME = "chunli"
MOVES = [m for m in choices(ME) if m != FORWARD]


def one_hot(order, pick, p=1.0):
    rest = (1.0 - p) / (len(order) - 1)
    return {a: (p if a == pick else rest) for a in order}


def answers(q8=None, **over):
    """A full answer set: perception answers (defaults: mid, neutral, grounded ...) and q8 words per move."""
    base = {"range": "mid", "trend": "steady", "phase": "neutral", "air": "grounded", "projectile": "none",
            "me_can_act": "free", "him_can_act": "free", "corner": "neither", "my_bar": "full", "his_bar": "full"}
    base.update(over)
    out = {k: one_hot(QUESTIONS[k], v, 0.9) for k, v in base.items()}
    for m in MOVES:
        out[U.Q8_KEY + m] = (q8 or {}).get(m, {WORKS: 0.0, MAY: 0.1, FAILS: 0.9})
    return out


class FakeAgent:
    def __init__(self, ans=None, cfg=None):
        self.cfg = {"perception": True, "note_version": 3} if cfg is None else cfg
        self.ans, self.seen = ans or answers(), []

    def predict(self, state, questions):
        self.seen.append((state, dict(questions)))
        return {"answers": {k: {"probabilities": dict(self.ans[k])} for k in questions}}


class FakeAdvisor:
    def __init__(self):
        self.asked = []

    def ask(self, text, q):
        self.asked.append((text, q))
        first = next(iter(q["criteria"]))
        return {m: 1.0 if m == first else 0.0 for m in q["criteria"]}


class PoisonRow(dict):
    """A RAM row that fails the test on ANY read: the U path must never touch RAM."""

    def __getitem__(self, k):
        raise AssertionError("RAM field %r read on the U path" % (k,))

    def get(self, k, default=None):
        raise AssertionError("RAM field %r read on the U path" % (k,))


def s1(ans=None, advisor=None):
    s = System1(None, ME, advisor=advisor)
    s._use(FakeAgent(ans))
    return s


# ---- rank score and words ------------------------------------------------------------------------------------------

def test_rank_score_is_works_plus_half_may_and_the_word_is_the_argmax():
    ans = answers(q8={"lk": {WORKS: 0.5, MAY: 0.1, FAILS: 0.4}, "throw": {WORKS: 0.2, MAY: 0.7, FAILS: 0.1}})
    rank, words = E.rank_scores(ans, MOVES), E.q8_words(ans, MOVES)
    assert rank["lk"] == pytest.approx(0.55) and rank["throw"] == pytest.approx(0.55)
    assert words["lk"] == WORKS and words["throw"] == MAY and words["mp"] == FAILS
    assert set(rank) == set(words) == set(MOVES) and FORWARD not in rank


def test_word_ties_go_to_the_first_answer():
    ans = answers(q8={"lk": {WORKS: 0.4, MAY: 0.4, FAILS: 0.2}})
    assert E.q8_words(ans, MOVES)["lk"] == WORKS


def test_without_advisor_the_best_rank_score_plays():
    ans = answers(q8={"lk": {WORKS: 0.2, MAY: 0.7, FAILS: 0.1}, "hp": {WORKS: 0.6, MAY: 0.0, FAILS: 0.4}})
    d = s1(ans).decide(RAW, RAW, "me=chunli")
    assert d["action"] == "hp" and d["best"] == "hp"            # 0.6 > 0.55


def test_without_advisor_walk_in_when_no_word_beats_likely_fails():
    ans = answers(q8={"lk": {WORKS: 0.45, MAY: 0.0, FAILS: 0.55}})    # best rank score, but its word is "fails"
    d = s1(ans).decide(RAW, RAW, "me=chunli")
    assert d["best"] == "lk" and d["action"] == FORWARD


def test_one_word_better_than_fails_is_enough_to_play_the_best_rank_score():
    ans = answers(q8={"lk": {WORKS: 0.45, MAY: 0.0, FAILS: 0.55}, "mp": {WORKS: 0.0, MAY: 0.6, FAILS: 0.4}})
    d = s1(ans).decide(RAW, RAW, "me=chunli")
    assert d["action"] == "lk"           # rank 0.45 > mp's 0.30; mp's "may work" keeps play on


# ---- one predict per decision, frames v3, note v3, everything logged ---------------------------------------------

def test_one_predict_with_every_question_two_hud_frames_and_the_v3_note():
    agent = FakeAgent()
    s = System1(None, ME)
    s._use(agent)
    d = s.decide(RAW, RAW, "me=chunli")
    assert len(agent.seen) == 1
    state, qs = agent.seen[0]
    assert qs == U.questions(ME) and state["context"] == "me=chunli"
    imgs = [np.asarray(i) for i in state["images"]]
    assert len(imgs) == 2 and all(i.shape == (256, 256, 3) for i in imgs)
    assert (imgs[1][:224] == 7).all() and (imgs[1][224:] == 0).all()     # HUD rows kept, padding black
    assert set(d["eye"]) == set(U.questions(ME)) and set(d["rank"]) == set(MOVES)
    assert d["probs"] == {} and d["p_hit"] is None


def test_every_answer_is_logged_in_the_action_entry():
    from tests.ram_rows import row
    s = s1()
    d = s.decide(RAW, RAW, "me=chunli")
    r = row()
    e = _close(0, ME, "ryu", (r, [r], dict(d, prompt="me=chunli"), "none", 0, None))
    assert e["eye"] == d["eye"] and e["rank"] == d["rank"] and e["eye_situation"] == d["eye_situation"]
    assert e["prompt"] == "me=chunli" and e["top3"] == []


def test_an_entry_without_the_eye_is_unchanged():
    from tests.ram_rows import row
    r = row()
    d = {"action": "lk", "best": "lk", "p_hit": 0.6, "predicted": "hit", "probs": {"lk": 0.6}, "prompt": "x"}
    e = _close(0, ME, "ryu", (r, [r], d, "hit", 0, None))
    assert not {"eye", "rank", "eye_situation"} & set(e)


# ---- no RAM on the U path --------------------------------------------------------------------------------------------

def test_no_ram_field_reaches_the_u_decision():
    s = s1(advisor=FakeAdvisor())
    d, text = _decide(s, "ryu", PoisonRow(), RAW, RAW)      # any read of the row fails the test
    assert text == "me=chunli" and d["action"]
    s = s1()
    d, text = _decide(s, "ryu", PoisonRow(), RAW, RAW)
    assert text == "me=chunli"


def code_tokens(mod):
    """Every identifier, attribute, imported name and string constant in a module's code (docstrings excluded)."""
    import ast
    tree = ast.parse(inspect.getsource(mod))
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {a.name for a in n.names} | {getattr(n, "module", None) or ""}
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
            out.add(n.value)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
    return out


def imported(mod):
    import ast
    out = set()
    for n in ast.walk(ast.parse(inspect.getsource(mod))):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {a.name for a in n.names} | {getattr(n, "module", None) or ""}
    return out


def test_the_eye_module_names_no_ram_field():
    toks = code_tokens(E)
    assert toks & (set(NAMES) | {"GROUND_Y", "rows", "ram", "ram_log", "p1", "p2"}) == set()
    imp = imported(E)
    assert imp & {"situation", "view", "note", "range_of", "opp_doing", "state_name", "System1"} == set()
    assert not any(t.endswith(("emu", "vs", "vs_sweep", "game_log", "system1", "value_oracle")) for t in imp)


def test_the_eye_refuses_a_checkpoint_without_the_perception_tag():
    with pytest.raises(ValueError):
        E.Eye(FakeAgent(cfg={"note_version": 2}), ME)
    with pytest.raises(ValueError):
        E.Eye(FakeAgent(cfg={"perception": True, "note_version": 2}), ME)


def test_the_eye_refuses_exploration_and_a_short_memory_without_advisor():
    s = System1(None, ME, explore=0.5)
    with pytest.raises(ValueError):
        s._use(FakeAgent())
    s = s1()
    s.short = {"lessons": [{"text": "use more lk"}]}
    with pytest.raises(ValueError):
        _decide(s, "ryu", PoisonRow(), RAW, RAW)


def test_a_non_perception_checkpoint_plays_as_before():
    from tests.test_system1_value import FakeAgent as OldAgent
    s = System1(None, ME)
    s._use(OldAgent({}))
    assert s.eye is None and s.note_version == 1


# ---- text laya's situation from laya-vision's own answers ------------------------------------------------------------

@pytest.mark.parametrize("band,want", [("throw", "close"), ("poke", "mid"), ("mid", "mid"), ("far", "far")])
def test_range_mapping(band, want):
    assert E.range_word(band) == want and want in RANGES


@pytest.mark.parametrize("phase,air,him,want", [
    ("neutral", "grounded", "free", "standing"),
    ("attacking", "grounded", "free", "attacking"),
    ("recovering after a miss", "grounded", "free", "attacking"),     # still in his attack state (0x0A / 0x0C)
    ("blocking", "grounded", "free", "standing"),                     # not a punish window
    ("being hit", "grounded", "stunned", "stunned"),
    ("being hit", "grounded", "free", "stunned"),
    ("neutral", "grounded", "knocked down", "stunned"),
    ("neutral", "grounded", "dizzy", "stunned"),
    ("attacking", "jumping at me", "free", "jumping"),                # air first, as opp_doing
    ("neutral", "landing", "free", "jumping"),
    ("neutral", "jumping away or straight up", "free", "jumping"),
    ("attacking", "grounded", "knocked down", "attacking"),            # attacking before stunned, as opp_doing
])
def test_doing_mapping(phase, air, him, want):
    assert E.doing_of(phase, air, him) == want and want in OPP_STATES


def test_every_answer_maps_into_text_layas_vocabulary():
    for band in QUESTIONS["range"]:
        assert E.range_word(band) in RANGES
    for ph in QUESTIONS["phase"]:
        for a in QUESTIONS["air"]:
            for h in QUESTIONS["him_can_act"]:
                assert E.doing_of(ph, a, h) in OPP_STATES
    assert set(QUESTIONS["my_bar"]) == set(BARS) == set(QUESTIONS["his_bar"])


def test_situation_of_reads_q1_q2_q3_q5_q7():
    ans = answers(range="throw", phase="attacking", my_bar="half", his_bar="low")
    assert E.situation_of(ans) == ("close", "attacking", "half", "low")


# ---- with an advisor --------------------------------------------------------------------------------------------------

def test_shortlist_equals_the_advisors_when_words_are_its_ratings():
    scores = {"lk": 0.7, "mp": 0.35, "hp": 0.1, "throw": 0.55, "block_high": 0.15}
    words = {m: rating(v, m) for m, v in scores.items()}
    lessons = ["avoid throw up close", "use more hp"]
    for sit in (("close", "standing"), ("mid", "jumping")):
        assert E.shortlist(scores, words, lessons, list(scores), sit) == advisor_shortlist(scores, lessons,
                                                                                          list(scores), sit)


def test_with_advisor_shortlist_words_come_straight_from_q8_and_the_situation_from_the_answers():
    adv = FakeAdvisor()
    ans = answers(range="throw", phase="attacking", my_bar="high", his_bar="half",
                  q8={"lk": {WORKS: 0.6, MAY: 0.3, FAILS: 0.1}, "mp": {WORKS: 0.1, MAY: 0.5, FAILS: 0.4},
                      "throw": {WORKS: 0.3, MAY: 0.3, FAILS: 0.4}, "hp": {WORKS: 0.0, MAY: 0.2, FAILS: 0.8}})
    s = s1(ans, advisor=adv)
    s.short = {"lessons": [{"text": "use more hp"}]}
    d = s.decide(RAW, RAW, "me=chunli")
    assert d["shortlist"] == {"lk": WORKS, "throw": FAILS, "mp": MAY, "hp": FAILS, FORWARD: None}
    text, q = adv.asked[0]
    assert text.startswith("He is up close and attacking. My bar is high, his bar is half.")
    assert "use more hp" in text
    assert d["action"] == "lk" and d["eye_situation"] == ["close", "attacking", "high", "half"]


def test_avoided_moves_are_skipped_from_the_top3():
    adv = FakeAdvisor()
    ans = answers(q8={"lk": {WORKS: 0.9, MAY: 0.0, FAILS: 0.1}, "mp": {WORKS: 0.5, MAY: 0.0, FAILS: 0.5},
                      "hp": {WORKS: 0.4, MAY: 0.0, FAILS: 0.6}, "throw": {WORKS: 0.3, MAY: 0.0, FAILS: 0.7}})
    s = s1(ans, advisor=adv)
    s.short = {"lessons": [{"text": "avoid lk"}]}
    d = s.decide(RAW, RAW, "me=chunli")
    assert list(d["shortlist"]) == ["mp", "hp", "throw", FORWARD]


def test_the_no_advice_arm_tells_text_laya_advice_none():
    adv = FakeAdvisor()
    s = s1(advisor=adv)
    s.short = {"lessons": [{"text": "use more hp"}]}
    s.advice_on = False
    s.decide(RAW, RAW, "me=chunli")
    assert adv.asked[0][0].endswith("Advice: none.")


def test_forward_is_never_asked():
    assert not any(k == U.Q8_KEY + FORWARD for k in U.questions(ME))
    assert json.dumps(U.questions(ME)).count("walking in?") == len(MOVES)
