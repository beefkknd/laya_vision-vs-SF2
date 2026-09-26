"""scripts/record_trace.py's Recorder against a fake Mesen."""
import os
import sys

from fake_mesen import MAP, FakeMesen

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from record_trace import Recorder  # noqa: E402


class ShortRunMesen(FakeMesen):
    """Mesen stops a Lua call after 1 s: with every window byte in the report, ~1500 frames is too long."""
    LIMIT = 300

    def _row(self):
        d = {"my_hp": self.hp[0], "opp_hp": self.hp[1], "my_x": self.x[0], "opp_x": self.x[1],
             "my_y": self.y[0], "opp_y": self.y[1]}
        return [d.get(n, self.t % 256) for n in self.vars]          # window bytes: the frame counter

    def run(self, frames, caps=()):
        assert len(frames) <= self.LIMIT, "RUN of %d frames: Mesen's script timeout" % len(frames)
        return super().run(frames, caps)


def test_a_long_run_is_recorded_in_short_chunks_with_every_frame_and_screenshot():
    rec = Recorder(ShortRunMesen())
    rec.set_vars(MAP)
    rec.load_state(b"")
    frames = [["right"]] * 700 + [[]] * 800
    obs = rec.run(frames, {0, 1496, 1500})
    assert len(obs.rams) == 1501 and all(len(r) == len(MAP) for r in obs.rams)
    assert [r[2] for r in obs.rams[:3]] == [80, 82, 84] and obs.rams[700][2] == 80 + 2 * 700
    assert sorted(obs.images) == [0, 1496, 1500] and obs.images[1500][0, 0, 0] == 1500 % 256
    assert len(rec.rows) == 1501 and [r["in"] for r in rec.rows[1:]] == [sorted(f) for f in frames]
    assert rec.rows[-1]["mem"][0][:2] == "%02x" % (1500 % 256)          # window bytes of the last frame
