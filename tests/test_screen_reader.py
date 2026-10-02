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
        if e["what"] in ("clock00", "bar_empty") and facts.round_state != "over":
            bad.append("%s: round %s, expected over" % (tag, facts.round_state))
        if e["what"] in ("start", "apart") and facts.round_state != "fighting":
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
    assert len(check(_fresh_banks())) >= 5


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
