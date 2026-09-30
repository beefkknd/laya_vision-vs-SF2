"""sf2/eval/boundary.py: trace one advice line through laya-vision, the label rule, text laya, the game, Qwen and the
verifier, and name the first stage that breaks. Synthetic rows, one fixture per stage."""
from sf2.eval import boundary as b

MOVES = ["lp", "mk", "c.mk", "sweep", "throw", "spinning_bird_kick", "block_high", "block_low", "forward"]
THROW = "use more throw up close"


def row(rng="close", state="stand", air=False, top3=(("sweep", 0.6), ("lp", 0.4), ("mk", 0.2)), short=None,
        action="sweep", dealt=0, taken=0, follows=True, rnd=0):
    short = dict(short) if short is not None else {m: b.rating(s, m) for m, s in top3}
    short.setdefault("forward", None)
    return {"range": rng, "opp_state": state, "opp_air": air, "top3": [list(x) for x in top3], "shortlist": short,
            "action": action, "dealt": dealt, "taken": taken, "follows_rule": follows, "round": rnd}


# ---- V: laya-vision --------------------------------------------------------------------------------------------

def test_vision_counts_only_where_the_line_applies():
    rows = [row(top3=(("throw", 0.55), ("lp", 0.4), ("mk", 0.1))),       # close: throw offered, "likely works"
            row(top3=(("throw", 0.2), ("lp", 0.4), ("mk", 0.1))),        # close: throw "likely fails"
            row(), row(),                                                # close: no throw
            row(rng="far", top3=(("throw", 0.9), ("lp", 0.4), ("mk", 0.1)))]   # far: the line does not apply
    v = b.stage_vision(rows, b.lesson_of(THROW, MOVES))
    assert v["decisions"] == 4
    assert v["top3"] == 2 and v["top3_share"] == 0.5
    assert v["ratings"] == {"likely works": 1, "likely fails": 1}


def test_vision_rates_a_block_on_its_own_scale_and_reads_what_he_does():
    les = b.lesson_of("use more block_low far away when he attacks", MOVES)
    rows = [row(rng="far", state="attack", top3=(("block_low", 0.15), ("lp", 0.4), ("mk", 0.1))),
            row(rng="far", state="attack", air=True, top3=(("block_low", 0.25), ("lp", 0.4), ("mk", 0.1)))]  # jumping
    v = b.stage_vision(rows, les)
    assert v["decisions"] == 1
    assert v["ratings"] == {"may work": 1}          # 0.15 on the block scale; the attack scale would say "likely fails"


# ---- R: the label rule -----------------------------------------------------------------------------------------

def test_rule_says_the_move_and_names_why_not():
    lines = [THROW, "avoid sweep up close"]
    rows = [row(short={"throw": "may work", "lp": "likely works"}),              # soft, offered, not failing: says it
            row(short={"throw": "likely fails", "lp": "likely works"}),          # soft vs likely fails: blocked
            row(short={"lp": "likely works"}),                                   # not offered at all
            row(rng="far", short={"throw": "likely works"})]                     # does not apply
    r = b.stage_rule(rows, b.lesson_of(THROW, MOVES), lines, MOVES)
    assert r["decisions"] == 3
    assert r["says"] == 1 and abs(r["says_share"] - 1 / 3) < 1e-9
    assert r["why_not"] == {"likely fails": 1, "not offered": 1}
    assert r["ratings"] == {"may work": 1, "likely fails": 1}


def test_rule_on_an_avoid_line_counts_it_ruled_out_and_what_the_rule_did_instead():
    les = b.lesson_of("avoid sweep up close", MOVES)
    rows = [row(short={"sweep": "likely works", "lp": "may work"}),
            row(short={"sweep": "likely works", "lp": "likely fails"}),              # all else fails: walk in
            row(short={"lp": "may work"})]                   # the shortlist already dropped it (advisor.shortlist)
    r = b.stage_rule(rows, les, ["avoid sweep up close"], MOVES)
    assert r["decisions"] == 3 and r["offered"] == 2
    assert r["says"] == 3 and r["says_share"] == 1.0
    assert r["rules"] == {"vision": 2, "walk": 1}
    hard = b.stage_rule(rows[:1], les, ["avoid sweep up close", "always sweep"], MOVES)
    assert hard["says"] == 1                                # avoid beats always in the rule: still ruled out


def test_rule_ruled_out_by_another_line():
    rows = [row(short={"throw": "likely works", "lp": "may work"})]
    r = b.stage_rule(rows, b.lesson_of(THROW, MOVES), [THROW, "avoid throw"], MOVES)
    assert r["says"] == 0 and r["why_not"] == {"ruled out": 1}


# ---- T: text laya ----------------------------------------------------------------------------------------------

def test_text_follows_and_picks_the_move_when_the_rule_says_it():
    s = {"throw": "may work", "lp": "likely works"}
    rows = [row(short=s, action="throw"), row(short=s, action="throw"), row(short=s, action="lp", follows=False),
            row(short={"throw": "likely fails", "lp": "likely works"}, action="lp")]
    t = b.stage_text(rows, b.lesson_of(THROW, MOVES), [THROW], MOVES)
    assert t["decisions"] == 4 and t["follows"] == 3
    assert t["rule_says"] == 3 and t["picked"] == 2 and abs(t["pick_share"] - 2 / 3) < 1e-9


# ---- G: the game -----------------------------------------------------------------------------------------------

def test_game_per_try_vs_her_other_decisions_and_use_shift():
    arm = [row(action="throw", dealt=20 + i % 2) for i in range(25)] + [row(action="lp", taken=10 + i % 2)
                                                                         for i in range(25)]
    none = [row(action="lp", taken=10) for _ in range(12)]
    g = b.stage_game(arm, none, b.lesson_of(THROW, MOVES))
    assert g["arm"]["tries"] == 25 and g["arm"]["cls"] == "better"
    assert g["use_arm"] == 0.5 and g["use_none"] == 0.0


def test_round_ab_is_run_level():
    arm = [[{"dealt": 50, "taken": 0}] * 3 for _ in range(4)]
    none = [[{"dealt": 0, "taken": 20}] * 3 for _ in range(4)]
    ab = b.round_ab(arm, none)
    assert ab["mean"] == 70 and ab["verdict"] == "HELPS"


# ---- Q: Qwen and H: the verifier -------------------------------------------------------------------------------

def _out(kind, move, rng, when, state, why):
    return {"claim": {"kind": kind, "move": move, "range": rng, "when": when}, "state": state, "why": why}


LEDGERS = [
    [{"game": -1, "outcome": [_out("use_more", "throw", "close", None, "rejected", "clearly worse: -3.0 (9 tries)"),
                              _out("avoid", "throw", None, None, "rejected", "too few tries to judge (4)")]},
     {"game": 2, "outcome": [_out("use_more", "throw", "close", None, "refused", "already rejected: x")]}],
    [{"game": -1, "outcome": [_out("avoid", "throw", "far", None, "registered", "clearly worse: y")]}],
    [{"game": -1, "outcome": [_out("use_more", "lp", "close", None, "registered", "clearly better: z"),
                              _out("use_more", "throw", "far", None, "testing", "to be tried in play")]}],
]


def test_qwen_counts_exact_proposals_and_claims_naming_the_move():
    hist = [row(action="throw", dealt=5)] * 3 + [row(action="lp")] * 20
    q = b.stage_qwen(LEDGERS, hist, b.lesson_of(THROW, MOVES), MOVES)
    assert q["ledgers"] == 3
    assert q["exact"] == 2 and q["exact_runs"] == 1
    assert q["exact_states"] == {"rejected": 1, "refused": 1}
    assert q["exact_whys"] == {"clearly worse": 1, "already rejected": 1}
    assert q["naming"] == {"use_more": 3, "avoid": 2}            # "use more throw far away" names it, is not L
    assert q["history"]["tries"] == 3 and q["history"]["cls"] == "few"


def test_verifier_disagrees_when_it_rejects_a_line_the_ab_shows_helps():
    q = b.stage_qwen(LEDGERS, [], b.lesson_of(THROW, MOVES), MOVES)
    h = b.stage_verifier(q, {"verdict": "HELPS"})
    assert h["expected"] == "registered" and h["agree"] is False
    assert b.stage_verifier(q, {"verdict": "NOT SHOWN"})["agree"] is None
    never = b.stage_qwen([], [], b.lesson_of(THROW, MOVES), MOVES)
    assert b.stage_verifier(never, {"verdict": "HELPS"})["agree"] is None


# ---- the diagnosis ---------------------------------------------------------------------------------------------

def _stages(top3_share=0.5, says_share=0.9, follows=0.95, pick=0.9, cls="better", verdict="HELPS", exact=1,
            hist_tries=30, agree=True):
    return {"V": {"decisions": 100, "top3_share": top3_share, "offered_share": top3_share},
            "R": {"decisions": 100, "says_share": says_share, "why_not": {"likely fails": 7}},
            "T": {"decisions": 100, "follows_share": follows, "rule_says": 50, "pick_share": pick},
            "G": {"arm": {"tries": 40, "cls": cls}, "ab": {"verdict": verdict, "mean": 10.0}},
            "Q": {"exact": exact, "naming": {}, "history": {"tries": hist_tries, "cls": "unclear"}},
            "H": {"agree": agree, "expected": "registered", "states": {}}}


def test_diagnosis_names_the_first_break_in_chain_order():
    les = b.lesson_of(THROW, MOVES)
    d = b.diagnose(les, _stages(top3_share=0.02, exact=0, hist_tries=6))
    assert d["breaks"][0].startswith("vision") and d["diagnosis"].startswith("vision")
    assert any(x.startswith("Qwen") and "6 tries" in x for x in d["breaks"])
    d = b.diagnose(les, _stages(says_share=0.3))
    assert d["breaks"][0].startswith("rule") and "likely fails" in d["breaks"][0]
    d = b.diagnose(les, _stages(follows=0.5))
    assert d["breaks"][0].startswith("text")
    assert b.diagnose(les, _stages(follows=0.0))["breaks"][0].startswith("text")     # 0% is a break, not "unknown"
    d = b.diagnose(les, _stages(agree=False))
    assert d["breaks"] == [d["breaks"][0]] and d["breaks"][0].startswith("verifier")
    assert b.diagnose(les, _stages())["breaks"] == []


def test_diagnosis_does_not_blame_qwen_for_a_line_not_shown_to_help():
    for verdict in ("NOT SHOWN", None):
        d = b.diagnose(b.lesson_of(THROW, MOVES), _stages(verdict=verdict, exact=0))
        assert not any(x.startswith("Qwen") for x in d["breaks"])


def test_diagnosis_flags_an_avoid_line_that_leaves_only_walking():
    les = b.lesson_of("avoid spinning_bird_kick", MOVES)
    s = _stages()
    s["R"] = {"decisions": 100, "says_share": 1.0, "why_not": {}, "rules": {"walk": 70, "vision": 30}}
    d = b.diagnose(les, s)
    assert d["breaks"][0].startswith("rule") and "walks in 70%" in d["breaks"][0]
    s["R"]["rules"] = {"walk": 20, "vision": 80}
    assert b.diagnose(les, s)["breaks"] == []


# ---- the whole trace -------------------------------------------------------------------------------------------

def test_trace_picks_the_single_line_arm_and_rechecks_the_logged_rule():
    s = {"throw": "may work", "lp": "likely works"}
    arms = {"none": {"lines": [], "rows": [row()] * 4, "rounds": [[{"dealt": 0, "taken": 10}]] * 3},
            "throw": {"lines": [THROW], "rows": [dict(row(short=s, action="throw"), rule_answers=["throw"])] * 4,
                      "rounds": [[{"dealt": 30, "taken": 0}]] * 3},
            "expert": {"lines": [THROW, "avoid lp"], "rows": [row(short=s, action="throw")] * 4,
                       "rounds": [[{"dealt": 20, "taken": 0}]] * 3}}
    t = b.trace(THROW, MOVES, arms, [], [])
    assert t["arm"] == "throw" and t["arms_with_line"] == ["expert", "throw"]
    assert t["R"]["logged_mismatch"] == 0
    assert t["G"]["ab"]["verdict"] == "HELPS"
    bad = dict(arms, throw=dict(arms["throw"], rows=[dict(row(short=s, action="throw"), rule_answers=["lp"])] * 4))
    assert b.trace(THROW, MOVES, bad, [], [])["R"]["logged_mismatch"] == 4


# ---- the CLI's loading ---------------------------------------------------------------------------------------

def _cli():
    import importlib.util
    import os
    import sys
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(here, "scripts"))
    spec = importlib.util.spec_from_file_location("boundary_cli", os.path.join(here, "scripts", "boundary.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_cli_loads_a_batch_relative_to_base(tmp_path):
    import json
    cli = _cli()
    for s in range(2):
        root = tmp_path / "rollouts" / ("run%d" % s)
        for arm, lines in (("none", None), ("throw", [THROW])):
            d = root / ("ryu_" + arm)
            d.mkdir(parents=True)
            (d / "actions.jsonl").write_text(json.dumps(row(action="throw")) + "\n" + json.dumps({"x": 1}) + "\n")
            (d / "rounds.jsonl").write_text(json.dumps({"dealt": 10, "taken": 0}) + "\n")
            if lines:
                (root / ("memory_ryu_%s.json" % arm)).write_text(json.dumps({"lessons": [{"text": t} for t in lines]}))
    logs = tmp_path / "logs"
    logs.mkdir()
    for s in range(2):
        (logs / ("seed_%d.log" % s)).write_text("x\nsaved rollouts/run%d\n" % s)
    arms = cli.load_arms(cli.roots(str(logs), str(tmp_path)), "ryu")
    assert sorted(arms) == ["none", "throw"] and arms["throw"]["lines"] == [THROW]
    assert len(arms["none"]["rows"]) == 2 and len(arms["throw"]["rounds"]) == 2       # the non-decision row skipped
    t = b.trace(THROW, MOVES, arms, cli.ledgers(str(tmp_path), "ryu"), [])
    assert t["arm"] == "throw" and t["V"]["top3_share"] == 0.0
