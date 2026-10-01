"""The opt-in RAM log (docs/prereg_u_perception.md): per decision, the full RAM rows from BACK frames before the decision
frame n to LOOKAHEAD frames after it, in a sidecar ram.jsonl keyed by (game, frame). Default output is unchanged.

The fake bridge (tests/fixtures/fake_bridge.py) puts the global frame index into every row (p2_special) and into every
image (pixel [0, 0]), so alignment is checked mechanically: the stored "n" row must be the frame of the "now" image and
the n-4 row the frame of the "prev" image."""
import importlib.util
import json
import os
import random
import sys
from pathlib import Path

import pytest

from fixtures.fake_bridge import END, FakeBridge, frame_of
from sf2.emu.vs import NAMES
from sf2.system1 import system1 as S
from sf2.system1.game_log import ram_entry
from sf2.data.perception import decode

ROOT = Path(__file__).parent.parent
GOLDEN = Path(__file__).parent / "fixtures" / "ram_log" / "golden_default.jsonl"


def games(ram_log=None, patch=None):
    """The golden's games: chunli and ryu vs a fake Ryu, one game each (lines: actions then the summary)."""
    out, ram = [], []
    for me in ("chunli", "ryu"):
        s1 = S.System1(None, me, seed=3)
        b, rng = FakeBridge(), random.Random(7)
        kw = {} if ram_log is None else {"ram_log": ram_log}
        rnd = S.play_round(b, s1, "ryu", b"state", rng, None, 0, **kw)
        out += [json.dumps(e) for e in rnd.log] + [json.dumps(rnd.summary)]
        ram.append((me, rnd))
    return "\n".join(out) + "\n", ram


def test_default_output_is_byte_identical_to_before_the_ram_log():
    text, _ = games()
    assert text == GOLDEN.read_text()


def test_the_ram_log_leaves_the_actions_and_summary_byte_identical():
    text, ram = games(ram_log=True)
    assert text == GOLDEN.read_text()
    assert all(rnd.ram for _, rnd in ram)


def test_without_the_flag_nothing_is_kept():
    _, ram = games()
    assert all(rnd.ram == [] for _, rnd in ram)


def test_one_record_per_action_keyed_by_game_and_frame():
    _, ram = games(ram_log=True)
    for _, rnd in ram:
        assert [(r["game"], r["frame"]) for r in rnd.ram] == [(e["game"], e["frame"]) for e in rnd.log]
        assert all(r["names"] == NAMES for r in rnd.ram)


def test_the_n_row_is_the_decision_row_and_the_frames_are_consecutive():
    _, ram = games(ram_log=True)
    for _, rnd in ram:
        by_frame = {e["frame"]: e for e in rnd.log}
        for rec in rnd.ram:
            rows, n = decode(rec)
            e = by_frame[rec["frame"]]
            assert abs(rows[n]["p2_x"] - rows[n]["p1_x"]) == e["gap"]
            assert rows[n]["p1_state"] in (0, 2) and rows[n]["result"] == 0       # System 1 could act there
            fs = [r["p2_special"] for r in rows]
            assert fs == list(range(fs[0], fs[0] + len(fs)))                      # the real continuation, no gaps
            assert n == min(S.BACK, fs[n]) and len(rows) - n - 1 <= S.LOOKAHEAD


def test_the_lookahead_is_full_until_the_round_ends():
    _, ram = games(ram_log=True)
    for _, rnd in ram:
        for rec in rnd.ram:
            rows, n = decode(rec)
            ahead = len(rows) - n - 1
            if rows[n]["p2_special"] + S.LOOKAHEAD <= END:
                assert ahead == S.LOOKAHEAD
            else:
                assert rows[-1]["result"] != 0 and ahead < S.LOOKAHEAD


def test_the_stored_n_and_n_minus_4_rows_are_the_frames_of_the_now_and_prev_images(monkeypatch):
    seen = {}

    def save(img_dir, game, frame, prev, cur):
        seen[(game, frame)] = (frame_of(prev), frame_of(cur))
        return ["p", "c"]

    monkeypatch.setattr(S, "_save_images", save)
    s1 = S.System1(None, "chunli", seed=3)
    b = FakeBridge()
    rnd = S.play_round(b, s1, "ryu", b"state", random.Random(7), "imgs", 0, ram_log=True)
    assert len(seen) == len(rnd.ram) > 20
    for rec in rnd.ram:
        rows, n = decode(rec)
        prev_f, cur_f = seen[(rec["game"], rec["frame"])]
        assert rows[n]["p2_special"] == cur_f
        assert rows[n - 4]["p2_special"] == prev_f


def test_ram_entry_is_compact_and_decodes_back():
    rows = [dict({k: i for k in NAMES}) for i in range(12)]
    rec = ram_entry(3, 120, rows, 5)
    assert (rec["game"], rec["frame"], rec["n"]) == (3, 120, 5)
    assert all(isinstance(r, list) and len(r) == len(NAMES) for r in rec["rows"])
    back, n = decode(json.loads(json.dumps(rec)))
    assert back == rows and n == 5
    with pytest.raises(ValueError):
        ram_entry(3, 120, rows, 12)


# ---- play_system1 --ram-log ----

def play_system1():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("play_system1", ROOT / "scripts" / "play_system1.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_one(tmp_path, monkeypatch, flag):
    import contextlib

    p = play_system1()
    monkeypatch.setattr(p, "open_fight", lambda *a, **k: contextlib.nullcontext((FakeBridge(), b"state")))
    argv = ["play_system1.py", "--model", "none", "--memory", "none", "--games", "1", "--out", str(tmp_path),
            "--one", "chunli", "1"] + (["--ram-log"] if flag else [])
    monkeypatch.setattr(sys, "argv", argv)
    assert p.main() == 0
    return tmp_path / "chunli"


def test_play_system1_writes_ram_jsonl_only_with_the_flag(tmp_path, monkeypatch):
    out = run_one(tmp_path / "on", monkeypatch, True)
    recs = [json.loads(x) for x in open(out / "ram.jsonl")]
    acts = [json.loads(x) for x in open(out / "actions.jsonl")]
    assert [(r["game"], r["frame"]) for r in recs] == [(a["game"], a["frame"]) for a in acts]
    off = run_one(tmp_path / "off", monkeypatch, False)
    assert not os.path.exists(off / "ram.jsonl")
    assert (off / "actions.jsonl").read_text() == (out / "actions.jsonl").read_text()


def test_play_system1_passes_the_flag_to_its_children(tmp_path, monkeypatch):
    p = play_system1()
    got = []
    monkeypatch.setattr(p, "fan_out", lambda cmds, *a, **k: got.extend(cmds) or [])
    for flag in (True, False):
        got.clear()
        argv = ["play_system1.py", "--model", "none", "--chars", "chunli,ryu", "--out", str(tmp_path)]
        monkeypatch.setattr(sys, "argv", argv + (["--ram-log"] if flag else []))
        p.main()
        assert all(("--ram-log" in c) == flag for _, c in got) and len(got) == 2
