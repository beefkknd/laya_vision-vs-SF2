"""scripts/eval_u.py (docs/prereg_u_perception.md "Offline gates"): per question accuracy (masked unknowns: no rows),
per character; Q2's confusion; gate 2 = the table's best move (RAM's cell) in the eye's top 3 by rank score over the
decisions where that best is not walking in (forward-best decisions reported apart), >= 0.9 on test_real and on held-out
Guile; question 8's softness (confident answers where the target is spread); gate 1 floors from a file (report-only
without one). No model is loaded: the answers are given."""
import importlib.util
import json
import os
import sys

import pytest

from sf2.data import u_data as U
from sf2.data.perception import Q8_ANSWERS, QUESTIONS
from sf2.system1.advice import FAILS, MAY, WORKS
from tests.test_eye import MOVES, answers
from tests.test_u_data import TH, entry, ram_rec, targets

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELL = ("chunli", "mid", 0, 0)
TABLE = {CELL: {"lk": 9.0, "throw": 5.0, "mp": 1.0, "forward": 2.0}, ("chunli", "close", 0, 0): {"forward": 5.0}}


def ev():
    if os.path.join(HERE, "scripts") not in sys.path:
        sys.path.insert(0, os.path.join(HERE, "scripts"))
    spec = importlib.util.spec_from_file_location("eval_u", os.path.join(HERE, "scripts", "eval_u.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def decision(game, rng="mid", opp="ryu", n=60):
    rows, _ = U.decision_rows("chunli", opp, entry(game, 64, opp=opp, rng=rng), ram_rec(game, 64, n=n), TH, targets())
    return rows


def good_q8(top):
    """q8 answers ranking ``top`` (a list) first, everything else likely fails."""
    q8 = {m: {WORKS: 0.0, MAY: 0.0, FAILS: 1.0} for m in MOVES}
    for i, m in enumerate(top):
        q8[m] = {WORKS: 0.9 - 0.1 * i, MAY: 0.0, FAILS: 0.1 + 0.1 * i}
    return q8


def test_accuracy_per_question_and_character_skips_masked_rows():
    rows = decision(2) + decision(5, n=3)                 # the second: trend unknown, no row
    ans = [answers(q8=good_q8(["lk"])), answers(trend="closing", q8=good_q8(["lk"]))]
    m = ev().metrics(list(zip(ev().by_decision(rows).values(), ans)), TABLE)
    acc = m["accuracy"]["chunli"]
    assert acc["range"] == {"n": 2, "acc": 1.0} and acc["trend"] == {"n": 1, "acc": 1.0}
    assert m["accuracy"]["all"]["range"]["n"] == 2


def test_wrong_answers_count_against_accuracy_and_fill_the_q2_confusion():
    rows = decision(2) + decision(5)
    ans = [answers(phase="attacking"), answers()]
    m = ev().metrics(list(zip(ev().by_decision(rows).values(), ans)), TABLE)
    assert m["accuracy"]["chunli"]["phase"] == {"n": 2, "acc": 0.5}
    assert m["q2_confusion"]["neutral"] == {"attacking": 1, "neutral": 1}


def test_gate2_table_best_in_top3_and_forward_best_apart():
    d_hit, d_miss, d_fwd = decision(2), decision(5), decision(8, rng="close")
    ans = [answers(q8=good_q8(["mp", "throw", "lk"])), answers(q8=good_q8(["mp", "throw", "hp"])), answers()]
    m = ev().metrics(list(zip([d_hit, d_miss, d_fwd], ans)), TABLE)
    g = m["gate2"]["all"]
    assert (g["n"], g["in_top3"], g["share"]) == (2, 1, 0.5)
    assert m["gate2"]["forward_best"] == {"n": 1, "eye_walks_in": 1}


def test_q8_softness_on_spread_targets():
    rows = decision(2)                     # lk target confident (0.7? no: spread), throw spread
    confident = {m: {WORKS: 0.95, MAY: 0.03, FAILS: 0.02} for m in MOVES}
    m = ev().metrics([(rows, answers(q8=confident))], TABLE)
    soft = m["q8"]
    assert soft["spread"]["n"] == 2 and soft["spread"]["confident_answers"] == 1.0
    assert soft["word_acc"]["n"] == 2


def test_gates_pass_fail_and_floors():
    e = ev()
    m_ok = {"gate2": {"all": {"n": 10, "in_top3": 9, "share": 0.9}},
            "accuracy": {"chunli": {"range": {"n": 10, "acc": 0.8}}, "all": {}}}
    m_bad = {"gate2": {"all": {"n": 10, "in_top3": 8, "share": 0.8}},
             "accuracy": {"chunli": {"range": {"n": 10, "acc": 0.6}}, "all": {}}}
    splits = {"test_real": m_ok, "test_heldout_guile": m_ok, "val": m_bad}
    assert e.gate_failures(splits, None) == []                     # val is reference only, no floors: report-only
    assert e.gate_failures(dict(splits, test_heldout_guile=m_bad), None) == [
        "gate 2 test_heldout_guile: 0.800 < 0.9 (8 of 10)"]
    assert e.gate_failures(splits, {"range": 0.7}) == []
    assert e.gate_failures(dict(splits, test_real=m_bad), {"range": 0.7}) == [
        "gate 2 test_real: 0.800 < 0.9 (8 of 10)", "gate 1 test_real chunli range: 0.600 < 0.7 (n 10)"]


def test_no_decisions_for_gate2_is_a_failure_not_a_pass():
    e = ev()
    m = {"gate2": {"all": {"n": 0, "in_top3": 0, "share": None}}, "accuracy": {"all": {}}}
    assert e.gate_failures({"test_real": m, "test_heldout_guile": m}, None)


def test_floors_file_is_read_and_checked(tmp_path):
    e = ev()
    p = tmp_path / "floors.json"
    p.write_text(json.dumps({"floors": {"range": 0.7}}))
    assert e.read_floors(str(p)) == {"range": 0.7}
    p.write_text(json.dumps({"floors": {"nonsense": 0.7}}))
    with pytest.raises(SystemExit):
        e.read_floors(str(p))
    assert e.read_floors(None) is None
