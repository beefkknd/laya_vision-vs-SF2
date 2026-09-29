"""What the CPU opponent was doing, by move, on every action entry (sf2.system1.opp_moves).

Fixtures: tests/fixtures/opp_moves/<opp>.json are real frames around labelled episodes, cut from the his_moves probe
(57,780 recorded frames vs Ryu, Ken and Honda, labels checked by screenshots). The whole-recording invariant reads
the probe itself (SF2_OPP_PROBE, default: the session scratchpad) and is skipped when it is not on this machine."""
import json
import os
import random
from pathlib import Path

import numpy as np
import pytest

from sf2.emu.vs import NAMES
from sf2.system1.game_log import action_entry
from sf2.system1.opp_moves import MOVES, NONE, OppMoveTracker, classify

FIX = Path(__file__).parent / "fixtures" / "opp_moves"
OPPS = ("ryu", "ken", "honda")
PROBE = Path(os.environ.get("SF2_OPP_PROBE", "/private/tmp/claude-502/-Users-worker-work-hobby-laya-vision-vs-SF2/"
                                             "3404ec4c-a642-4f56-b6e8-6c692ee5e912/scratchpad/his_moves"))
DECISION = {"action": "c.mk", "p_hit": 0.4, "predicted": "whiff", "probs": {"c.mk": 0.4, "sweep": 0.3}}


def _full(r):
    """A fixture row -> a full RAM row (every sf2.emu.vs name; the ones the probe excerpt leaves out are 0)."""
    return dict({n: 0 for n in NAMES}, **r)


def excerpts(opp):
    data = json.loads((FIX / ("%s.json" % opp)).read_text())
    out = []
    for e in data["excerpts"]:
        rows = [_full(dict(zip(data["keys"], v))) for v in e["rows"]]
        out.append(dict(e, rows=rows, opp=opp))
    return out


def all_excerpts():
    return [e for o in OPPS for e in excerpts(o)]


def single(e):
    """The excerpt cut one row after its episode (the padding may hold the start of his next attack)."""
    return e["rows"][:e["start"] + e["dur"] + 1]


def episode(e):
    """(the episode's rows, shot2 on the frame before it)."""
    return e["rows"][e["start"]:e["start"] + e["dur"]], e["rows"][e["start"] - 1]["shot2"]


def windows(rows, size=16, first=None):
    """Decisions every ``size`` frames: (before, rows to the next decision); ``before`` is the previous window's last
    row (as in play_round)."""
    out, before = [], rows[0] if first is None else first
    for i in range(1, len(rows), size):
        w = rows[i:i + size]
        out.append((before, w))
        before = w[-1]
    return out


def entry(before, rows, frame=0):
    return action_entry(1, frame, "chunli", "ken", before, rows, DECISION, "whiff")


def run_tracker(wins):
    tr, out, held = OppMoveTracker(), [], []
    for k, (before, rows) in enumerate(wins):
        got = tr.add(entry(before, rows, k), before, rows)
        held.append(k + 1 - len(out) - len(got))
        out += got
    out += tr.flush()
    return out, held


# --- the rules, one by one, on real episodes ---------------------------------------------------------------------

@pytest.mark.parametrize("e", all_excerpts(), ids=lambda e: "%s-%s-%d" % (e["opp"], e["label"], e["t"]))
def test_each_rule_labels_the_real_episode_like_the_probe(e):
    rows, pre = episode(e)
    assert all(r["p2_state"] == 0x0A for r in rows)
    assert classify(rows, pre) == e["label"]


def test_fixtures_cover_every_move():
    assert {e["label"] for e in all_excerpts()} == set(MOVES)
    assert sum((FIX / ("%s.json" % o)).stat().st_size for o in OPPS) < 200_000


def _one(label, opp="ken"):
    return next(e for e in excerpts(opp) if e["label"] == label)


def test_a_projectile_already_on_screen_is_not_his_fireball():
    rows, _ = episode(_one("fireball"))
    assert classify([dict(r, shot2=1) for r in rows], 1) != "fireball"


def test_slap_needs_sub_6_or_8_and_honda_bear_hug_is_a_throw():
    rows, pre = episode(_one("normal", "honda"))
    assert classify(rows, pre) == "normal"
    assert classify(rows[:3] + [dict(rows[3], p2_sub=8)] + rows[4:], pre) == "slap"
    assert classify(rows[:3] + [dict(rows[3], p2_sub=6)] + rows[4:], pre) == "slap"
    assert classify(rows[:3] + [dict(rows[3], p2_sub=4)] + rows[4:], pre) == "normal"
    assert classify(rows[:3] + [dict(rows[3], p1_state=0, p1_y=150)] + rows[4:], pre) == "throw"   # held up high
    assert classify(rows[:3] + [dict(rows[3], p1_state=0x04, p1_y=150)] + rows[4:], pre) == "normal"  # own jump


def test_hurricane_needs_16_level_frames_12_to_17_px_up():
    rows, pre = episode(_one("normal"))
    base = rows[:2]
    level = [dict(rows[2], p2_y=192 - 14) for _ in range(16)]
    assert classify(base + level, pre) == "hurricane"
    assert classify(base + level[:15], pre) == "normal"
    assert classify(base + [dict(r, p2_y=192 - 20) for r in level], pre) == "normal"   # too high: not the spin


def test_empty_episode_is_normal_and_moves_are_in_precedence_order():
    assert MOVES == ("throw", "fireball", "uppercut", "hurricane", "slap", "jump_attack", "normal")
    assert NONE == "none"


PRECEDENCE_CASES = [  # (base move, frame changes that add a lower/higher move, expected)
    ("fireball", {"p1_state": 0x14}, "throw"),              # throw > fireball
    ("uppercut", {"shot2": 1}, "fireball"),                 # fireball > uppercut (0->1 inside)
    ("uppercut", {"p2_sub": 8}, "uppercut"),                # uppercut > slap
    ("hurricane", {"p2_sub": 8}, "hurricane"),              # hurricane > slap
    ("jump_attack", {"p2_sub": 8}, "slap"),                 # slap > jump_attack
]


@pytest.mark.parametrize("base,change,want", PRECEDENCE_CASES)
def test_precedence_when_several_rules_match(base, change, want):
    rows, pre = episode(_one(base))
    mid = len(rows) // 2
    rows = rows[:mid] + [dict(rows[mid], **change)] + rows[mid + 1:]
    assert classify(rows, 0 if "shot2" in change else pre) == want


# --- the tracker: an episode spans several decisions ----------------------------------------------------------------

def _overlaps(e_rows, ep_rows):
    ids = {id(r) for r in ep_rows}
    return any(id(r) in ids for r in e_rows)


@pytest.mark.parametrize("label,opp", [("fireball", "ken"), ("fireball", "ryu"), ("uppercut", "ken"),
                                       ("hurricane", "ryu"), ("slap", "honda")])
def test_every_decision_overlapping_the_episode_is_stamped_with_its_move(label, opp):
    e = _one(label, opp)
    ep, _ = episode(e)
    wins = windows(single(e))
    out, held = run_tracker(wins)
    assert len(out) == len(wins)
    over = [k for k, (_, w) in enumerate(wins) if _overlaps(w, ep)]
    assert len(over) >= 3                                          # the episode spans several decisions
    for k, (o, (before, w)) in enumerate(zip(out, wins)):
        assert o["frame"] == k                                     # emitted in order
        assert o["opp_move"] == (label if k in over else NONE)
        assert o["opp_shot"] is bool(before["shot2"])
    assert max(held) >= len(over) - 1                              # entries were held until the episode closed


def test_entries_are_released_as_soon_as_their_episode_closes():
    e = _one("fireball")
    wins = windows(single(e), size=4)
    tr, released = OppMoveTracker(), []
    for k, (before, rows) in enumerate(wins):
        released.append(len(tr.add(entry(before, rows, k), before, rows)))
    assert released[0] == 1                                        # before the episode: out at once
    assert 0 in released                                           # during it: held
    assert sum(released) == len(wins)                              # the episode closed inside the excerpt
    assert tr.flush() == []


def test_no_episode_means_none_and_opp_shot_is_the_slot_at_the_decision():
    rows = [r for e in all_excerpts() for r in e["rows"] if r["p2_state"] != 0x0A][:41]
    rows = [dict(r, p2_state=0, shot2=int(i >= 20)) for i, r in enumerate(rows)]
    out, _ = run_tracker(windows(rows, size=8))
    assert [o["opp_move"] for o in out] == [NONE] * len(out)
    assert [o["opp_shot"] for o in out] == [False, False, False, True, True]
    assert len(out) == 5


def test_round_end_flush_stamps_an_unfinished_episode_from_what_it_has():
    e = _one("fireball")
    ep, _ = episode(e)
    onset = next(i for i, r in enumerate(ep) if r["shot2"])
    cut = e["start"] + onset + 3                                   # the round ends mid-episode, after the shot
    wins = windows(e["rows"][:cut], size=8)
    tr, out = OppMoveTracker(), []
    for k, (before, rows) in enumerate(wins):
        out += tr.add(entry(before, rows, k), before, rows)
    assert len(out) < len(wins)                                    # still held when the round ends
    out += tr.flush()
    assert len(out) == len(wins)
    assert out[-1]["opp_move"] == "fireball"
    assert tr.flush() == []


def test_the_move_of_an_entry_overlapping_two_episodes_is_the_higher_one():
    n, f = _one("normal"), _one("fireball")
    nep, npre = episode(n)
    fep, fpre = episode(f)
    gap = [dict(nep[-1], p2_state=0)]
    rows = [dict(nep[0], p2_state=0)] + nep + gap + fep + gap
    out, _ = run_tracker([(rows[0], rows[1:])])
    assert out[0]["opp_move"] == "fireball"


def test_the_existing_keys_are_unchanged_and_the_entry_is_not_mutated():
    e = _one("uppercut")
    wins = windows(e["rows"])
    tr, out, olds = OppMoveTracker(), [], []
    for k, (before, rows) in enumerate(wins):
        old = entry(before, rows, k)
        snapshot = json.dumps(old)
        olds.append(old)
        out += tr.add(old, before, rows)
        assert json.dumps(old) == snapshot
    out += tr.flush()
    for old, new in zip(olds, out):
        assert list(new)[:-2] == list(old) and list(new)[-2:] == ["opp_move", "opp_shot"]
        assert json.dumps({k: v for k, v in new.items() if k not in ("opp_move", "opp_shot")}) == json.dumps(old)


def test_opp_move_is_set_exactly_when_he_attacked_in_the_window():
    for e in all_excerpts():
        out, _ = run_tracker(windows(e["rows"], size=12))
        assert all((o["opp_move"] != NONE) == o["opp_attacked"] for o in out), (e["opp"], e["label"])
        assert all(o["opp_move"] in MOVES + (NONE,) for o in out)


# --- play_round writes the two keys on every entry (a replayed recording, no emulator) ------------------------------

class ReplayBridge:
    """Plays back recorded RAM rows: run(frames) returns the current row plus one row per frame."""

    def __init__(self, rows):
        self.rows, self.pos = rows, 0

    def run(self, frames, caps=()):
        n = len(frames)
        rams = [self.rows[min(self.pos + i, len(self.rows) - 1)] for i in range(n + 1)]
        self.pos += n
        if self.pos >= len(self.rows) - 1:
            rams[-1] = dict(rams[-1], result=1)
        blank = np.zeros((4, 4, 3), np.uint8)
        return type("Obs", (), {"rams": [[r[k] for k in NAMES] for r in rams], "images": [blank] * (n + 1)})


def test_play_round_stamps_every_logged_action():
    from sf2.system1.system1 import System1, play_round
    e = _one("fireball")
    rows = [dict(r, result=0, p1_state=0, p1_y=192) for r in e["rows"]]   # Chun-Li can act at every decision
    rnd = play_round(ReplayBridge(rows), System1(None, "chunli", seed=3), "ken", None, random.Random(0), None, 1)
    assert rnd.log and all({"opp_move", "opp_shot"} <= set(a) for a in rnd.log)
    assert "fireball" in [a["opp_move"] for a in rnd.log]
    assert [a["frame"] for a in rnd.log] == sorted(a["frame"] for a in rnd.log)
    assert all((a["opp_move"] != NONE) == a["opp_attacked"] for a in rnd.log)


def test_play_round_ending_mid_attack_still_logs_every_decision():
    from sf2.system1.system1 import System1, play_round

    class Counting(System1):
        calls = 0

        def decide(self, *a, **k):
            Counting.calls += 1
            return super().decide(*a, **k)

    e = _one("fireball")
    ep, _ = episode(e)
    cut = e["start"] + next(i for i, r in enumerate(ep) if r["shot2"]) + 3        # the round ends mid-fireball
    rows = [dict(r, result=0, p1_state=0, p1_y=192) for r in e["rows"][:cut]]
    rnd = play_round(ReplayBridge(rows), Counting(None, "chunli", seed=3), "ken", None, random.Random(0), None, 1)
    assert len(rnd.log) == Counting.calls and rnd.summary["actions"] == Counting.calls
    assert rnd.log[-1]["opp_move"] == "fireball"


# --- invariant: the tracker agrees with the probe's labels over the whole recordings --------------------------------

def _probe_rows(opp):
    path = PROBE / opp / "rows.jsonl"
    if not path.exists():
        pytest.skip("the his_moves probe is not on this machine (%s)" % path)
    with open(path) as f:
        return [json.loads(line) for line in f]


@pytest.mark.parametrize("opp", OPPS)
def test_tracker_agrees_with_the_probe_labels_on_95pct_of_complete_episodes(opp):
    rows = _probe_rows(opp)
    labels = json.loads((PROBE / "labels.json").read_text())[opp]
    got = {}
    segs = {}
    for r in rows:
        segs.setdefault(r["seg"], []).append(r)
    for seg_rows in segs.values():                                 # one tracker per round (a recorded segment)
        tr = OppMoveTracker()
        for before, w in windows(seg_rows):
            tr.add({}, before, w)
        tr.flush()
        got.update({(ep["first"]["seg"], ep["first"]["t"]): ep["move"] for ep in tr.episodes})
    complete = [lab for lab in labels if not lab["truncated"]]
    agree = sum(got.get((lab["seg"], lab["t"])) == lab["label"] for lab in complete)
    rate = agree / len(complete)
    print("%s: %d/%d complete episodes agree (%.1f%%)" % (opp, agree, len(complete), 100 * rate))
    assert rate >= 0.95
