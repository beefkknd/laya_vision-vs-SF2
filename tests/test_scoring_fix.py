"""Two scoring fixes before the measured screen-loop run.

FIX 1 - the round result the screen decides (sf2.system2.screen_evidence): a completed round scored from the screen
(no RAM referee) used to come back result="unknown". It now reads win / loss / draw off the two drawn HEALTH BARS at
the last decision (his bar empty -> KO win, her bar empty -> KO loss, otherwise more life wins, equal is a draw).

FIX 2 - the in-loop replay scorer hang (sf2.eval.runner.open_fight): the screen loop's in-play scorer
(scripts/play_loop_screen._real_score) hands open_fight the savestate BYTES it already read, but open_fight treated
``state`` as a PATH: os.path.exists() on 172 KB of bytes was False, so it raised "no savestate <raw bytes>" - the
symptom's "dumped raw savestate bytes" - and aborted the loop after round 0, before any replay Mesen was launched.
open_fight now takes either the bytes or a path (sf2.eval.runner.read_state), so the loop scores every round headless.

Seen RED (how, before the fixes):
  - FIX 1: round_evidence(..., replay=None) returned summary["result"] == "unknown" for the crafted and the real
    smoke rounds - test_screen_result_* asserted win/loss/draw and failed.
  - FIX 2: read_state(..., <bytes>) raised SystemExit whose message was the 172 KB savestate - test_read_state_* and
    test_real_round_scores_via_bytes (open_fight with state=bytes) failed before the fix.
Both confirmed by `git stash`ing sf2/eval/runner.py + sf2/system2/screen_evidence.py and re-running this file.
"""
import concurrent.futures as cf
import json
import os
import shutil
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from sf2.config import DEFAULT_ROM, REPO                                   # noqa: E402
from sf2.eval.runner import read_state                                     # noqa: E402
from sf2.system2 import screen_evidence as E                               # noqa: E402
from sf2.vocab import FULL_LIFE                                            # noqa: E402

ME = "chunli"
# a real screen-only round the smoke aborted on (only g00_r0 was written before the loop died)
REAL_RUN = os.path.join(REPO, "rollouts", "loop_screen", "20261002_205940_honda")


# --------------------------------------------------------------------- FIX 1 (pure, fast)
def _dec(my_life, his_life, action="cl.hp", doing="attacking", his_air=False, dx=36):
    return {"i": 0, "k": 0, "action": action, "situation": ["close", doing, "full", "full"],
            "follows_rule": True, "rule": "soft", "advice_text": "Advice: none.", "lines": [],
            "moment": {"my_life": my_life, "his_life": his_life, "doing": doing, "his_air": his_air,
                       "side": "left", "dx": dx}}


def test_round_result_from_health_bars():
    assert E.round_result(64, 0) == "win"       # his bar empty -> KO win
    assert E.round_result(0, 80) == "loss"      # her bar empty -> KO loss
    assert E.round_result(100, 40) == "win"     # more life
    assert E.round_result(40, 100) == "loss"
    assert E.round_result(50, 50) == "draw"     # equal
    assert E.round_result(0, 0) == "draw"       # double KO


@pytest.mark.parametrize("my_last,his_last,want", [
    (54, 126, "loss"), (64, 0, "win"), (70, 176, "loss"), (0, 80, "loss"), (90, 90, "draw")])
def test_screen_round_never_unknown(my_last, his_last, want):
    # a two-decision round whose LAST drawn bars are (my_last, his_last); the screen path (replay=None) must decide
    decs = [_dec(176, 176), _dec(my_last, his_last)]
    _, summary = E.round_evidence(0, ME, "honda", decs, replay=None)
    assert summary["source"] == "screen"
    assert summary["result"] == want
    assert summary["result"] != "unknown"


def test_screen_round_matches_the_real_smoke_rounds():
    # the four rounds of the fully-played --no-score smoke: last drawn bars decide, and dealt/taken are kept as summed
    cases = {"20261002_221402_honda": {"g00_r0": "loss", "g00_r1": "win", "g01_r0": "loss", "g01_r1": "loss"}}
    for run, rounds in cases.items():
        base = os.path.join(REPO, "rollouts", "loop_screen", run)
        if not os.path.isdir(base):
            pytest.skip("missing smoke run %s" % run)
        for rd, want in rounds.items():
            decs = E.read_decisions(os.path.join(base, rd))
            _, summary = E.round_evidence(0, ME, "honda", decs, replay=None)
            assert summary["result"] == want, "%s: got %s, want %s" % (rd, summary["result"], want)


def test_replay_result_still_wins_over_the_screen():
    # when a replay score is given, its RAM-truth result is used (not the health-bar guess)
    decs = [_dec(176, 176), _dec(10, 150)]          # screen bars would say "loss"
    replay = {"result": "win", "dealt": 60, "taken": 26, "hp": 34, "my_life_end": 120, "opp_life_end": 100}
    _, summary = E.round_evidence(0, ME, "honda", decs, replay)
    assert summary["result"] == "win" and summary["source"] == "replay"


# --------------------------------------------------------------------- read_end_bars: the KO, not the refill
# BUG: a round ends, the KO is the empty bar (e.g. his_life 0), then the screen REFILLS both bars to full life for the
# round-over banner. read_end_bars scanned for the LAST row with both bars drawn and returned that refill (176/176),
# so round_result saw 176==176 and called every --no-score round a "draw" -- win/loss went structurally invisible.
def _write_reads(path, rows):
    with open(path, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def test_read_end_bars_returns_the_KO_not_the_refill(tmp_path):
    rd = tmp_path / "g00_r1"
    rd.mkdir()
    _write_reads(rd / "reads.jsonl", [
        {"k": 1, "my_life": FULL_LIFE, "his_life": FULL_LIFE, "round": "fighting"},   # round start, both full
        {"k": 2, "my_life": 58, "his_life": 0, "round": "fighting"},                   # THE KO: his bar empty
        {"k": 3, "my_life": None, "his_life": None, "round": "fighting"},              # bars briefly unread
        {"k": 4, "my_life": FULL_LIFE, "his_life": FULL_LIFE, "round": "over"},        # the round-over REFILL
    ])
    end = E.read_end_bars(str(rd))
    assert end == {"my_life": 58, "his_life": 0}                   # the KO frame, NOT the 176/176 refill
    assert E.round_result(end["my_life"], end["his_life"]) == "win"


def test_read_end_bars_win_at_full_own_life_is_not_the_refill(tmp_path):
    # she KOs him without taking a hit: (full, 0) must NOT be mistaken for the (full, full) refill
    rd = tmp_path / "r"
    rd.mkdir()
    _write_reads(rd / "reads.jsonl", [
        {"k": 1, "my_life": FULL_LIFE, "his_life": 0, "round": "fighting"},
        {"k": 2, "my_life": FULL_LIFE, "his_life": FULL_LIFE, "round": "over"},
    ])
    assert E.read_end_bars(str(rd)) == {"my_life": FULL_LIFE, "his_life": 0}


def test_read_end_bars_keeps_a_genuine_double_KO_draw(tmp_path):
    # both bars empty is a real draw, not a refill -- it must be kept
    rd = tmp_path / "r"
    rd.mkdir()
    _write_reads(rd / "reads.jsonl", [
        {"k": 1, "my_life": 0, "his_life": 0, "round": "fighting"},
        {"k": 2, "my_life": FULL_LIFE, "his_life": FULL_LIFE, "round": "over"},
    ])
    end = E.read_end_bars(str(rd))
    assert end == {"my_life": 0, "his_life": 0}
    assert E.round_result(end["my_life"], end["his_life"]) == "draw"


@pytest.mark.parametrize("rd,want", [("g00_r1", "loss"), ("g01_r1", "win")])
def test_read_end_bars_on_the_real_ab_hyb_rounds(rd, want):
    # the bug as caught in the wild: ab_hyb round_05_ryu scored these as "draw" off the 176/176 refill
    base = os.path.join(REPO, "playbooks", "ab_hyb", "round_05_ryu", rd)
    if not os.path.isdir(base):
        pytest.skip("missing ab_hyb session (gitignored)")
    end = E.read_end_bars(base)
    assert end is not None and (end["my_life"], end["his_life"]) != (FULL_LIFE, FULL_LIFE)
    assert E.round_result(end["my_life"], end["his_life"]) == want


# --------------------------------------------------------------------- FIX 2 state-reading (pure, fast)
def test_read_state_accepts_bytes():
    blob = b"MSS\x01not-a-path-just-the-savestate-content" * 100
    assert read_state(ME, "honda", blob) == blob     # bytes used as is, no os.path.exists() on them


def test_read_state_reads_a_path(tmp_path):
    p = tmp_path / "s.state"
    p.write_bytes(b"savestate-on-disk")
    assert read_state(ME, "honda", str(p)) == b"savestate-on-disk"


def test_read_state_missing_path_is_a_clean_error():
    with pytest.raises(SystemExit) as e:
        read_state(ME, "honda", "states/does_not_exist.state")
    assert "does_not_exist" in str(e.value) and len(str(e.value)) < 200   # names the path, never dumps bytes


# --------------------------------------------------------------------- FIX 2 end to end (needs Mesen + the ROM)
def _have_mesen() -> bool:
    from sf2.emu.headless import find_mesen
    try:
        find_mesen()
    except FileNotFoundError:
        return False
    rom = os.environ.get("SF2_ROM") or os.path.join(REPO, DEFAULT_ROM)
    return os.path.exists(rom) and os.path.isdir(os.path.join(REAL_RUN, "g00_r0"))


mesen = pytest.mark.skipif(not _have_mesen(), reason="needs Mesen, the ROM and the real smoke round dir")

REPLAY_PORT = 48428


def _score(round_dir, opp, state_bytes, port):
    # exactly the in-loop path: open_fight is handed the savestate BYTES (not a path), with a hard timeout so a
    # regression that waits for a manual Mesen fails the test instead of hanging the suite.
    from sf2.eval.runner import open_fight
    from sf2.system1.screen_replay import score_round

    def go():
        with open_fight(ME, opp, port, state=state_bytes) as (b, _):
            return score_round(b, state_bytes, round_dir, ME, opp)
    with cf.ThreadPoolExecutor(1) as ex:
        return ex.submit(go).result(timeout=120)


@mesen
def test_real_round_scores_via_bytes_no_hang():
    run = json.load(open(os.path.join(REAL_RUN, "run.json")))
    state = open(os.path.join(REPO, run["state"]["path"]), "rb").read()
    s = _score(os.path.join(REAL_RUN, "g00_r0"), run["opp"], state, REPLAY_PORT)
    assert s["replay_match"] is True
    assert s["result"] in ("win", "loss", "draw")
    assert {"hp", "dealt", "taken"} <= set(s)


@mesen
def test_tampered_replay_is_refused(tmp_path):
    from sf2.system1.screen_replay import ReplayMismatch

    run = json.load(open(os.path.join(REAL_RUN, "run.json")))
    state = open(os.path.join(REPO, run["state"]["path"]), "rb").read()
    rd = str(tmp_path / "g00_r0")
    shutil.copytree(os.path.join(REAL_RUN, "g00_r0"), rd)
    # drop the first pressed move: a real divergence the pixel-exact check must refuse
    with open(os.path.join(rd, "decisions.jsonl")) as f:
        d = next(dc for dc in map(json.loads, f) if any(p != "-" for p in dc["pressed"]))
    inp = json.load(open(os.path.join(rd, "inputs.json")))
    n = len(d["pressed"])
    inp["inputs"] = inp["inputs"][:d["k"]] + ["-"] * n + inp["inputs"][d["k"] + n:]
    json.dump(inp, open(os.path.join(rd, "inputs.json"), "w"))
    with pytest.raises(ReplayMismatch):
        _score(rd, run["opp"], state, REPLAY_PORT + 1)
