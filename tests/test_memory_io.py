"""Memory files: written whole or not at all; a change from outside the loop (the brain panel) is picked up only while
System 2 is idle, and a bad file never stops the game."""
import json
import os
import time

import pytest

from sf2.memory import OutsideWatch, save, try_load

MOVES = ["sweep", "lp"]
GOOD = {"me": "chunli", "opp": "ryu", "lessons": []}


def test_save_leaves_no_temp_and_replaces_whole(tmp_path):
    p = str(tmp_path / "short" / "chunli_vs_ryu.json")
    save(p, GOOD)
    save(p, dict(GOOD, lessons=[]))
    assert json.load(open(p)) == GOOD and os.listdir(tmp_path / "short") == ["chunli_vs_ryu.json"]


def test_a_failed_save_keeps_the_old_file(tmp_path):
    p = str(tmp_path / "m.json")
    save(p, GOOD)
    with pytest.raises(TypeError):
        save(p, {"me": object()})                              # not JSON: fails half way through
    assert json.load(open(p)) == GOOD and os.listdir(tmp_path) == ["m.json"]


def test_try_load_reports_instead_of_crashing(tmp_path):
    p = tmp_path / "m.json"
    assert try_load(str(p), MOVES) == (None, None)
    p.write_text('{"me": "chunli", "less')                     # half written
    mem, why = try_load(str(p), MOVES)
    assert mem is None and "not JSON" in why
    p.write_text(json.dumps({"lessons": [{"text": "x", "kind": "nope"}]}))
    mem, why = try_load(str(p), MOVES)
    assert mem is None and "kind" in why
    p.write_text("[1, 2]")
    assert "not a memory" in try_load(str(p), MOVES)[1]


def bump(p, mem):
    save(str(p), mem)
    t = time.time() + 5
    os.utime(p, (t, t))


def test_outside_change_only_while_system2_is_idle(tmp_path):
    p = tmp_path / "s.json"
    bump(p, GOOD)
    w = OutsideWatch()
    w.seen("ryu", str(p))                                      # what the loop loaded
    assert w.check("ryu", str(p), busy=False, actions=MOVES) is None
    new = dict(GOOD, lessons=[{"text": "use more sweep", "kind": "use_more", "action": "sweep",
                               "evidence": {"tries": 1, "count": 1, "refs": ["g0f1"]}}])
    bump(p, new)
    assert w.check("ryu", str(p), busy=True, actions=MOVES) is None      # maybe System 2's own write: wait
    mem, msg = w.check("ryu", str(p), busy=False, actions=MOVES)
    assert mem == new and "1 lessons" in msg
    assert w.check("ryu", str(p), busy=False, actions=MOVES) is None     # seen once, not again


def test_a_bad_outside_file_keeps_the_memory_in_play(tmp_path):
    p = tmp_path / "s.json"
    bump(p, GOOD)
    w = OutsideWatch()
    w.seen("ryu", str(p))
    p.write_text("{half")
    t = time.time() + 9
    os.utime(p, (t, t))
    keep, msg = w.check("ryu", str(p), busy=False, actions=MOVES)
    assert keep is OutsideWatch.KEEP and "kept the memory in play" in msg
    assert w.check("ryu", str(p), busy=False, actions=MOVES) is None


def test_an_opponent_never_loaded_is_not_an_outside_change(tmp_path):
    p = tmp_path / "s.json"
    bump(p, GOOD)
    assert OutsideWatch().check("ken", str(p), busy=False, actions=MOVES) is None


@pytest.mark.parametrize("bad", [{"lessons": [1]}, {"lessons": [{"text": "x", "kind": "avoid", "evidence": [1]}]},
                                 {"lessons": [{"text": "x", "kind": "avoid", "evidence": {"tries": 3, "count": "2"}}]},
                                 {"lessons": "none"}, {"lessons": [{"text": "x", "kind": ["avoid"]}]}])
def test_try_load_never_raises_on_odd_shapes(tmp_path, bad):              # found by the blind review, 2026-09-29
    p = tmp_path / "m.json"
    p.write_text(json.dumps(bad))
    mem, why = try_load(str(p), MOVES)
    assert mem is None and why


def test_a_panel_edit_after_system2_saved_is_still_picked_up(tmp_path):
    p = tmp_path / "s.json"
    bump(p, GOOD)
    w = OutsideWatch()
    w.seen("ryu", str(p))
    saved_at = os.path.getmtime(p)                 # System 2 saved here ...
    t = time.time() + 20                           # ... then the panel wrote before the loop polled System 2
    save(str(p), dict(GOOD, lessons=[]))
    os.utime(p, (t, t))
    w.seen("ryu", str(p), stamp=saved_at)          # the loop takes System 2's version, stamped as System 2 left it
    got = w.check("ryu", str(p), busy=False, actions=MOVES)
    assert got is not None and "changed outside" in got[1]
