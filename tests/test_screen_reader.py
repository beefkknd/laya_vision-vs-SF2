"""The screen reader (sf2.screen) on frozen frames (tests/fixtures/screen, written by scripts/collect_reader_gate.py
--fixtures from held-out games; the expected facts are the referee's: RAM / OAM at collection time), and seeded faults
that these checks must catch (each fault patched in, the checks must turn red).

Needs the reader's files (out/sprite_catalog, out/screen_reader/anchors.json, hud_digits.npz): skipped without them.
"""
import json
import os
from typing import Dict, List

import numpy as np
import pytest

from sf2.screen import assets as A
from sf2.screen import hud as HUD
from sf2.screen import reader as R

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "screen")
NEEDED = [os.path.join(A.CATALOG, "catalog.json"), A.ANCHORS, HUD.DIGITS, os.path.join(FIX, "expected.json")]
pytestmark = pytest.mark.skipif(not all(os.path.exists(p) for p in NEEDED),
                                reason="reader files missing (scripts/collect_reader_gate.py --anchors / --calib-hud)")
X_TOL, HEALTH_TOL = 4, 0.03


def _fixtures():
    with open(os.path.join(FIX, "expected.json")) as f:
        exp = json.load(f)
    z = np.load(os.path.join(FIX, "frames.npz"))
    return exp, z["frames"], z["prev"]


def check(banks=None) -> List[str]:
    """Every mismatch between the reader and the expected facts over the fixtures (empty = all good)."""
    exp, frames, prevs = _fixtures()
    banks = banks or A.load_banks()
    bad = []
    locks: Dict[str, R.RoundLock] = {}
    for e, fr in zip(exp, frames):
        if e["what"] == "start":
            (left, right), _ = R.identify(fr, banks)
            if [left, right] != e["chars"]:
                bad.append("%s start: identified %s, expected %s" % (e["game"], [left, right], e["chars"]))
            locks[e["game"]] = R.lock_round(fr, banks)
    for e, fr, pv in zip(exp, frames, prevs):
        tag = "%s %s k=%d" % (e["game"], e["what"], e["k"])
        facts = R.read_screen(fr, locks[e["game"]], pv, banks)
        by_player = {f.player: f for f in (facts.left, facts.right)}
        xs = {p: e["p"][str(p)]["x"] for p in (1, 2)}
        for p in (1, 2):
            want, got = e["p"][str(p)], by_player.get(p)
            if got is None or got.character != e["chars"][p - 1]:
                bad.append("%s: player %d is %s" % (tag, p, got and got.character))
                continue
            if want["key"] is None:
                continue
            if got.x is None or abs(got.x - want["x"]) > X_TOL:
                bad.append("%s: p%d x %s, expected %d" % (tag, p, got.x, want["x"]))
            if got.facing != want["facing"]:
                bad.append("%s: p%d facing %s, expected %s" % (tag, p, got.facing, want["facing"]))
            if got.action != want["action"]:
                bad.append("%s: p%d action %s, expected %s" % (tag, p, got.action, want["action"]))
            if got.in_air != (want["air_label"] == "air"):
                bad.append("%s: p%d in_air %s, expected label %s" % (tag, p, got.in_air, want["air_label"]))
            side = "left" if xs[p] < xs[3 - p] else "right"
            if abs(xs[1] - xs[2]) > 8 and got.side != side:
                bad.append("%s: p%d on the %s, expected %s" % (tag, p, got.side, side))
            if got.health is None or abs(got.health - e["health"][p - 1]) > HEALTH_TOL:
                bad.append("%s: p%d health %s, expected %.3f" % (tag, p, got.health, e["health"][p - 1]))
        if facts.hud.timer != e["clock"]:
            bad.append("%s: clock %s, expected %d" % (tag, facts.hud.timer, e["clock"]))
        if e["what"] == "shot" and not facts.projectiles:
            bad.append("%s: no projectile" % tag)
        if e["what"] == "clock00" and facts.round_state != "over":
            bad.append("%s: round %s, expected over" % (tag, facts.round_state))
        # gate amendment 2026-10-02: a bar that looks empty is not round over (the next round starting is)
        if e["what"] in ("start", "apart", "bar_empty") and facts.round_state != "fighting":
            bad.append("%s: round %s, expected fighting" % (tag, facts.round_state))
    return bad


def test_fixtures_read_right():
    assert check() == []


def test_facts_are_immutable():
    exp, frames, _ = _fixtures()
    lock = R.lock_round(frames[0])
    f = R.read_screen(frames[0], lock)
    with pytest.raises(Exception):
        f.left.x = 0


def test_read_is_pure():
    exp, frames, prevs = _fixtures()
    lock = R.lock_round(frames[0])
    i = next(j for j, e in enumerate(exp) if e["game"] == exp[0]["game"] and e["what"] != "start")
    assert R.read_screen(frames[i], lock, prevs[i]) == R.read_screen(frames[i].copy(), lock, prevs[i])


def test_rejects_a_wrong_frame():
    exp, frames, _ = _fixtures()
    lock = R.lock_round(frames[0])
    with pytest.raises(ValueError):
        R.read_screen(frames[0][:, :200], lock)
    with pytest.raises(ValueError):
        R.read_screen(frames[0].astype(np.float32), lock)


def test_round_reader_time_over_rule():
    """The clock at 00 ends the round after TIME_OVER_FRAMES frames, not on the first 00 frame."""
    exp, frames, _ = _fixtures()
    i0 = next(j for j, e in enumerate(exp) if e["what"] == "clock00")
    start = next(j for j, e in enumerate(exp) if e["what"] == "start" and e["game"] == exp[i0]["game"])
    rr = R.RoundReader()
    assert rr.feed(frames[start]).round_state == "fighting"
    states = [rr.feed(frames[i0]).round_state for _ in range(R.TIME_OVER_FRAMES)]
    assert states[:-1] == ["fighting"] * (R.TIME_OVER_FRAMES - 1) and states[-1] == "over"


def test_unknown_fighter_defaults_to_block_and_is_logged(tmp_path):
    """Owner 2026-10-02: no confident match -> action "block" (owner decision), unknown set, the crop / side / score logged."""
    import json as _json
    from sf2.screen.unknown_log import UnknownLog
    exp, frames, _ = _fixtures()
    i = next(j for j, e in enumerate(exp) if e["what"] == "apart")
    start = next(j for j, e in enumerate(exp) if e["what"] == "start" and e["game"] == exp[i]["game"])
    rr = R.RoundReader(log=UnknownLog(str(tmp_path)))
    assert not any(f.unknown for f in (rr.feed(frames[start]).left, rr.feed(frames[i]).right))
    assert rr.log.count == 0
    f = rr.feed(frames[i])
    hidden = frames[i].copy()
    x0, y0, x1, y1 = f.right.box
    hidden[max(0, y0):y1, max(0, x0):x1] = hidden[max(0, y0):y1, max(0, x0):x1][:, ::-1] // 2   # scrambled sprite
    g = rr.feed(hidden)
    assert g.right.unknown and g.right.action == R.DEFAULT_ACTION == "block" and not g.left.unknown
    lines = [_json.loads(x) for x in open(tmp_path / "unknown.jsonl")]
    assert len(lines) == 1 and lines[0]["side"] == "right" and lines[0]["best_score"] < R.UNKNOWN_SCORE
    assert (tmp_path / "crops" / lines[0]["crop"]).exists()


# ---------------------------------------------------------------------------------------------- round over
LAG = 1                         # frame k shows RAM row k - 1 (scripts/gate_screen_reader.py LAG)
NEXT_WINDOW = 300               # the gate: no earlier than RAM's result, within 300 frames after the next round's start
ROUND_SEQ = os.path.join(FIX, "round_seq")


def _round_seq():
    with open(ROUND_SEQ + ".json") as f:
        exp = json.load(f)
    z = np.load(ROUND_SEQ + ".npz")
    return exp, z["frames"], z["game"]


def round_mismatches(step) -> List[str]:
    """Feed each game's sparse sequence (round start .. KO .. black .. the next round's start .. after) through
    read_screen + ``step`` ((facts, track) -> (facts, track)); every frame whose round_state / new_round is not the
    amended rule's (empty = all good): "fighting" up to the next round's first frame, there "over" + new_round (and
    within the gate's window of RAM's result / next round start), "fighting" after."""
    exp, frames, game = _round_seq()
    banks = A.load_banks()
    bad = []
    for g in sorted(set(game.tolist())):
        idx = [i for i in range(len(exp)) if game[i] == g]
        lock = R.lock_round(frames[idx[0]], banks)
        track = None
        for i in idx:
            e = exp[i]
            facts, track = step(R.read_screen(frames[i], lock, None, banks), track)
            new = e["what"] == "next_start"
            want = ("over", True) if new else ("fighting", False)
            got = (facts.round_state, facts.new_round)
            if got != want:
                bad.append("%s %s k=%d: %s, expected %s" % (e["game"], e["what"], e["k"], got, want))
            if new and not e["result_k"] <= e["k"] - LAG <= e["next_k"] + NEXT_WINDOW:
                bad.append("%s: next round frame %d outside RAM's window" % (e["game"], e["k"]))
    return bad


@pytest.mark.skipif(not os.path.exists(ROUND_SEQ + ".json"), reason="no round_seq fixture")
def test_round_over_is_the_next_round_starting():
    assert round_mismatches(R.step_round) == []


def _old_bar_empty_rule(facts, st):
    """Known-bad twin: the rule before the amendment (a bar empty ends the round at once; over until both bars are
    full again with the clock not 00)."""
    zeros, over = st or (0, False)
    hud = facts.hud
    zeros = zeros + 1 if hud.timer == 0 else 0
    new = False
    if any(hud.bar_empty) or zeros >= R.TIME_OVER_FRAMES:
        over = True
    elif over and hud.health == (1.0, 1.0) and hud.timer not in (0, None):
        over, new, zeros = False, True, 0
    from dataclasses import replace
    return replace(facts, round_state="over" if over else "fighting", new_round=new), (zeros, over)


@pytest.mark.skipif(not os.path.exists(ROUND_SEQ + ".json"), reason="no round_seq fixture")
def test_fault_old_bar_empty_rule_fails_round_over():
    bad = round_mismatches(_old_bar_empty_rule)
    assert any("ko_empty" in b for b in bad), bad


def test_fault_old_bar_empty_single_frame_rule_fails_fixtures(monkeypatch):
    """read_screen with the old single-frame rule (a bar empty -> over) turns the bar_empty fixture red."""
    real = R.read_screen

    def old(frame, lock, prev=None, banks=None):
        f = real(frame, lock, prev, banks)
        from dataclasses import replace
        return replace(f, round_state="over") if any(f.hud.bar_empty) else f
    monkeypatch.setattr(R, "read_screen", old)
    assert any("bar_empty" in b and "expected fighting" in b for b in check())


@pytest.mark.skipif(not os.path.exists(ROUND_SEQ + ".json"), reason="no round_seq fixture")
def test_round_reader_relocks_on_the_new_round():
    exp, frames, game = _round_seq()
    idx = [i for i in range(len(exp)) if game[i] == game[0]]
    rr = R.RoundReader()
    events = []
    for i in idx:
        lock = rr.lock
        f = rr.feed(frames[i])
        events.append((exp[i]["what"], f.round_state, f.new_round, lock is not None and rr.lock is not lock))
    assert [e for e in events if e[1] == "over" or e[2] or e[3]] == [("next_start", "over", True, True)], events


def test_step_round_is_pure():
    exp, frames, _ = _fixtures()
    lock = R.lock_round(frames[0])
    facts = R.read_screen(frames[0], lock)
    t = R.RoundTrack(zeros=3, progressed=True)
    a = R.step_round(facts, t)
    assert a == R.step_round(facts, t) and t == R.RoundTrack(zeros=3, progressed=True)


# ---------------------------------------------------------------------------------------------- seeded faults
def _fresh_banks():
    return A.load_banks.__wrapped__(A.CATALOG, A.ANCHORS)


def test_fault_no_flip_is_caught(monkeypatch):
    monkeypatch.setattr(A, "FACES", (A.FACE_RIGHT,))
    assert any("facing" in b or " x " in b for b in check(_fresh_banks()))


def test_fault_unmasked_background_is_caught(monkeypatch):
    """Comparing the whole box (transparent pixels as black) instead of the sprite's own pixels."""
    real = A._load_png

    def opaque(path):
        a = real(path).copy()
        a[..., 3] = 255
        return a
    monkeypatch.setattr(A, "_load_png", opaque)
    assert check(_fresh_banks()) != []


def test_fault_no_anchor_table_is_caught(monkeypatch):
    banks = A.load_banks.__wrapped__(A.CATALOG, A.ANCHORS + ".missing")
    assert any(" x " in b for b in check(banks))


def test_fault_hud_bar_row_off_by_one_is_caught(monkeypatch):
    monkeypatch.setattr(HUD, "BAR_ROW", 34)
    assert any("health" in b for b in check())


def test_fault_hud_clock_rows_off_by_one_is_caught(monkeypatch):
    monkeypatch.setattr(HUD, "CLOCK_ROWS", slice(49, 61))
    assert any("clock" in b for b in check())


def test_fault_swapped_sides_is_caught(monkeypatch):
    from dataclasses import replace
    real = R.by_side

    def swapped(f1, f2):
        left, right = real(f1, f2)
        return replace(right, side="left"), replace(left, side="right")
    monkeypatch.setattr(R, "by_side", swapped)
    assert any("on the" in b for b in check())
