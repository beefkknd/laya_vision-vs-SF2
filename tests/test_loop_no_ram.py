"""M1 (d): the screen runner reads NO RAM in play. The real emulator handle (sf2.system1.screen_emu.ScreenEmu) raises
RamForbidden on any RAM access; here it wraps a fake bridge that sends only frames. A scripted mini-game runs a
decision through sf2.system1.loop_runner.play_round and must finish without RamForbidden.

Seen RED: the guard itself is proven to bite (test_ram_access_raises). If the play loop reached for RAM - e.g. by
asking the bridge for RAM rows, or if the bridge ever returned any - ScreenEmu would raise and play_round would fail;
confirmed by having FakeBridge return a non-empty ``rams`` list (ScreenEmu._images raises).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import pytest                                                        # noqa: E402
from looptools import FakeBridge, FakeReader, FollowerLaya, make_facts   # noqa: E402

from sf2.system1.loop_runner import play_round                      # noqa: E402
from sf2.system1.screen_emu import RamForbidden, ScreenEmu          # noqa: E402

ME, OPP = "chunli", "ryu"


def test_ram_access_raises():
    emu = ScreenEmu(FakeBridge())
    with pytest.raises(RamForbidden):
        _ = emu.rams                                                # the guard bites on any RAM-ish attribute


def test_play_round_makes_a_decision_with_no_ram(tmp_path):
    emu = ScreenEmu(FakeBridge())
    script = [make_facts(ME, OPP),                                  # round-start lock read
              make_facts(ME, OPP),                                  # the decision moment (I can act)
              make_facts(ME, OPP, over=True)]                       # the round ends right after the press
    reader = FakeReader(ME, OPP, script)
    summary = play_round(emu, FollowerLaya(), ME, OPP, state=b"x", state_id={"path": "p", "sha256": "0"},
                         delay=8, lines=[], out=str(tmp_path), reader=reader)
    assert summary["decisions"] >= 1
    assert summary["chars"] == [ME, OPP]
    assert os.path.exists(os.path.join(str(tmp_path), "decisions.jsonl"))
    # and the handle only ever gave frames + took buttons (no RAM rows were produced)
    assert emu.inputs and emu.captures


def test_play_round_raises_if_the_bridge_leaks_ram(tmp_path):
    class LeakyBridge(FakeBridge):
        def run(self, frames, caps=()):
            obs = super().run(frames, caps)
            obs.rams = [[0, 0, 0]]                                  # a bridge that leaks RAM must be rejected
            return obs
    emu = ScreenEmu(LeakyBridge())
    reader = FakeReader(ME, OPP, [make_facts(ME, OPP), make_facts(ME, OPP), make_facts(ME, OPP, over=True)])
    with pytest.raises(RamForbidden):
        play_round(emu, FollowerLaya(), ME, OPP, state=b"x", state_id={"path": "p", "sha256": "0"},
                   delay=8, lines=[], out=str(tmp_path), reader=reader)
