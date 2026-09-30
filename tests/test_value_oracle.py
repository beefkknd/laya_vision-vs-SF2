"""The lookup-table value ranking (Dr Fable 2026-09-30, docs/prereg_value_oracle.md): mean net per (character, range,
opp_attacking, opp_airborne, move) from explored training decisions, shrunk toward 0; System 1 plays its best move."""
import numpy as np
import pytest

from sf2.data.value_oracle import SHRINK, build, cell, rank
from sf2.system1.system1 import System1, choices

NOTE = "me=chunli dist=close side=left dx=+20 my_bar=full opp_bar=full opp_airborne=0 opp_crouch=0 opp_attacking=0"


def e(move, net, rng="close", att="stand", air=False, explored=True, game=0, opp="ryu", me="chunli"):
    return {"me": me, "opp": opp, "range": rng, "opp_state": att, "opp_air": air, "action": move, "dealt": max(net, 0),
            "taken": -min(net, 0), "explored": explored, "game": game}


def test_cell_from_note():
    assert cell(NOTE) == ("chunli", "close", 0, 0)
    assert cell(NOTE.replace("opp_attacking=0", "opp_attacking=1")) == ("chunli", "close", 1, 0)
    with pytest.raises(ValueError):
        cell(NOTE.rsplit(" opp_attacking", 1)[0])       # a v1 note has no opp_attacking


def test_build_uses_explored_training_games_and_shrinks():
    rows = [e("throw", 20)] * 10 + [e("throw", -50, explored=False)] * 10 + [e("throw", -50, game=2)] * 10
    t = build(rows)
    assert t[("chunli", "close", 0, 0)]["throw"] == pytest.approx(20 * 10 / (10 + SHRINK))


def test_build_excludes_the_guile_holdout():
    t = build([e("throw", 20, opp="guile")] * 10)
    assert t == {}


def test_attacking_and_airborne_from_the_log():
    t = build([e("throw", 10, att="special")] * 5 + [e("lk", 10, air=True)] * 5)
    assert ("chunli", "close", 1, 0) in t and ("chunli", "close", 0, 1) in t


def test_rank_unknown_move_is_zero():
    t = {("chunli", "close", 0, 0): {"throw": 5.0}}
    r = rank(t, NOTE, choices("chunli"))
    assert r["throw"] == 5.0 and r["lk"] == 0.0 and set(r) == set(choices("chunli"))


def test_system1_plays_the_table():
    t = {("chunli", "close", 0, 0): {"throw": 5.0, "forward": 1.0}}
    s = System1(None, "chunli", oracle=t)
    img = np.zeros((224, 256, 3), np.uint8)
    d = s.decide(img, img, NOTE)
    assert d["action"] == "throw" and d["values"]["throw"] == 5.0 and s.note_version == 2
