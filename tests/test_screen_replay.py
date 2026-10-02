"""Replay determinism of screen-only play (docs/laya_text_only_plan.md: "A replay whose frames do not match the saved
frames pixel for pixel is refused"). Real emulator: one round of screen-only play (the lookup table, no text laya, so
it runs in seconds) on the screen-only bridge, then the offline replay WITH RAM must reproduce every captured frame;
the seeded fault (one recorded press dropped) must be refused. Skipped without Mesen, the ROM or the savestate.
~40 s: the slow tier."""
import hashlib
import json
import os
import shutil

import pytest

from sf2.config import DEFAULT_ROM, REPO

STATE = os.path.join(REPO, "states", "p1_chunli_vs_ryu.state")
PLAY_PORT, REPLAY_PORT = 48913, 48431


def _have_mesen() -> bool:
    from sf2.emu.headless import find_mesen
    try:
        find_mesen()
    except FileNotFoundError:
        return False
    rom = os.environ.get("SF2_ROM") or os.path.join(REPO, DEFAULT_ROM)
    return os.path.exists(rom) and os.path.exists(STATE)


pytestmark = pytest.mark.skipif(not _have_mesen(), reason="needs Mesen, the ROM and states/p1_chunli_vs_ryu.state")


@pytest.fixture(scope="module")
def played(tmp_path_factory):
    from sf2.data import value_oracle
    from sf2.system1.screen_emu import open_screen
    from sf2.system1.screen_play import play_screen_round
    from sf2.system1.system1 import System1

    out = tmp_path_factory.mktemp("screen_run")
    with open(STATE, "rb") as f:
        state = f.read()
    sid = {"path": STATE, "sha256": hashlib.sha256(state).hexdigest()}
    s1 = System1(None, "chunli", oracle=value_oracle.load(os.path.join(REPO, "lessons", "value_oracle_v1.json")))
    with open_screen(PLAY_PORT) as emu:
        summary = play_screen_round(emu, s1, "chunli", "ryu", state, sid, 9, str(out / "g00_r0"))
    return out / "g00_r0", state, summary


def _score(rec_dir, state):
    from sf2.eval.runner import open_fight
    from sf2.system1.screen_replay import score_round

    with open_fight("chunli", "ryu", REPLAY_PORT, state=STATE) as (b, _):
        return score_round(b, state, str(rec_dir), "chunli", "ryu")


def test_a_screen_round_replays_pixel_identical(played):
    rec, state, summary = played
    assert summary["decisions"] > 0 and summary["end"] in ("new_round", "time_over")
    s = _score(rec, state)
    assert s["replay_match"] is True
    assert s["captures"] == len(json.load(open(rec / "captures.json"))) > 100
    assert s["result"] in ("win", "loss", "draw")                   # the round really ended inside the recording
    assert s["result_frame"] <= s["screen_end_frame"]
    assert s["agreement"]["side"] > 0.9


def test_one_dropped_press_is_refused(played, tmp_path):
    from sf2.system1.screen_replay import ReplayMismatch

    rec, state, _ = played
    bad = tmp_path / "g00_r0"
    shutil.copytree(rec, bad)
    inp = json.load(open(bad / "inputs.json"))
    first_press = next(i for i, b in enumerate(inp["inputs"]) if b != "-")
    inp["inputs"][first_press] = "-"
    json.dump(inp, open(bad / "inputs.json", "w"))
    with pytest.raises(ReplayMismatch):
        _score(bad, state)
