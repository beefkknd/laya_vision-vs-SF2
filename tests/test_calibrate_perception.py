"""scripts/calibrate_perception.py: the perception thresholds from logged data (docs/prereg_u_perception.md)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ram_rows import row
from sf2.data import perception as P
from sf2.emu.vs import NAMES
from sf2.system1.game_log import ram_entry

ROOT = Path(__file__).parent.parent


def cal():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("calibrate_perception", ROOT / "scripts" / "calibrate_perception.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def entry(me="chunli", action="throw", gap=30, thrown=False, actual="whiff", air=False, **kw):
    return dict({"me": me, "opp": "ryu", "action": action, "gap": gap, "opp_air": air, "actual": actual,
                 "opp_reaction": ["stand", "thrown"] if thrown else ["stand"], "game": 0, "frame": 0}, **kw)


# ---- the cut: a decision stump, no free threshold ----

def test_stump_is_the_cut_that_classifies_most_entries_right():
    c = cal()
    pos = list(range(10, 41)) + [70]            # one stray long connect
    neg = [12, 20] + list(range(45, 120))
    assert c.stump(pos, neg) == 40


def test_stump_takes_the_smallest_of_equally_good_cuts():
    assert cal().stump([10, 20], [50, 60]) == 20
    assert cal().stump([10, 20, 40], [30, 50]) == 20          # 20 and 40 both classify 4 of 5 right


def test_stump_without_positives_is_none():
    assert cal().stump([], [10, 20]) is None


def test_throw_band_per_character_with_a_pooled_fallback():
    c = cal()
    es = ([entry(gap=g, thrown=True) for g in range(10, 48)] + [entry(gap=g) for g in range(48, 120)] +
          [entry(me="ryu", gap=g, thrown=True) for g in (20, 25)] + [entry(me="ryu", gap=g) for g in (60, 70)] +
          [entry(gap=20, thrown=True, air=True)] * 50)                # opp in the air at the decision: left out
    band = c.throw_band(es)
    assert band["chunli"] == 47 and "ryu" not in band            # 2 throws < MIN_POSITIVES: Ryu uses "all"
    assert band["all"] == 47
    assert band["n"]["chunli"] == (38, 72)


def test_poke_band_counts_ground_normals_that_hit_or_were_blocked():
    c = cal()
    es = ([entry(action="c.mk", gap=g, actual="hit") for g in range(20, 70)] +
          [entry(action="sweep", gap=g, actual="blocked") for g in range(20, 80, 2)] +
          [entry(action="mk", gap=g) for g in range(80, 160)] +
          [entry(action="spinning_bird_kick", gap=150, actual="hit")] * 80 +        # a special: not a normal
          [entry(action="hp", gap=150, actual="hit", air=True)] * 80)               # air: left out
    assert c.poke_band(es)["chunli"] == 78


# ---- walls and the corner from per-frame rows ----

def frames(p1_xs, p2_xs, p1_char=5, p2_char=0):
    return [row({"x": a, "char": p1_char}, {"x": b, "char": p2_char}) for a, b in zip(p1_xs, p2_xs)]


def test_walls_are_the_modes_at_the_stage_edges_and_d_the_widest_character_offset():
    c = cal()
    rows = (frames([44] * 4 + [53] * 300 + [55] * 40 + [200] * 296, [300] * 640) +    # 44: a transient (probe)
            frames([200] * 300, [459] * 200 + [400] * 100) +
            frames([200] * 300, [448] * 150 + [300] * 150, p2_char=1))      # Honda pinned 11 px short of 459
    w = c.walls(rows)
    assert w["walls"] == [53, 459]
    assert w["corner_d"] == 11
    assert w["by_char"]["honda"]["hi"] == 448 and w["x_seen"] == [44, 459]


# ---- k: my fastest connecting attack's frames from the decision to contact ----

def ram_rec(game, frame, contact_after, n_rows=40, back=8):
    def r(i):
        hit = contact_after is not None and i >= back + contact_after
        return row({"x": 100, "char": 5}, {"x": 140, "state": 0x0E if hit else 0, "react": 0x02 if hit else 0})
    return ram_entry(game, frame, [r(i) for i in range(n_rows)], back)


def test_k_is_the_median_frames_to_contact_of_my_fastest_move():
    c = cal()
    recs, acts = [], []
    for i, (move, s) in enumerate([("lp", 5)] * 6 + [("lp", 7)] * 5 + [("sweep", 9)] * 8 + [("hp", 3)] * 2):
        recs.append(ram_rec(0, i * 10, s))
        acts.append(entry(action=move, actual="hit", frame=i * 10))
    k = c.k_frames(acts, recs)
    assert k["k"] == 5 and k["by_move"]["chunli"]["lp"] == [5.0, 11]      # hp: 2 contacts < MIN_CONTACTS
    assert k["by_move"]["chunli"]["sweep"] == [9.0, 8]


def test_k_skips_whiffs_and_contact_already_there():
    c = cal()
    recs = [ram_rec(0, 0, 0)] * 6 + [ram_rec(0, 10, None)] * 6
    acts = [entry(action="lp", actual="hit", frame=0)] * 6 + [entry(action="lp", actual="whiff", frame=10)] * 6
    assert c.k_frames(acts, recs)["k"] is None


# ---- the output is what the labels read ----

def test_main_writes_thresholds_the_labels_accept(tmp_path, monkeypatch):
    c = cal()
    logs = tmp_path / "logs" / "a" / "chunli"
    logs.mkdir(parents=True)
    es = ([entry(gap=g, thrown=True) for g in range(10, 48)] + [entry(gap=g) for g in range(48, 120)] +
          [entry(action="c.mk", gap=g, actual="hit") for g in range(20, 70)] +
          [entry(action="mk", gap=g) for g in range(80, 160)])
    (logs / "actions.jsonl").write_text("".join(json.dumps(e) + "\n" for e in es))
    rows_file = tmp_path / "rows.jsonl"
    rows_file.write_text("".join(json.dumps(r) + "\n" for r in
                                 frames([53] * 300 + [200] * 300, [459] * 300 + [300] * 300)))
    out = tmp_path / "th.json"
    monkeypatch.setattr(sys, "argv", ["calibrate_perception.py", "--logs", str(tmp_path / "logs" / "*" / "*" /
                                      "actions.jsonl"), "--rows", str(rows_file), "--out", str(out), "--no-q8"])
    assert c.main() == 0
    th = json.loads(out.read_text())
    for key in ("lag", "throw_max", "poke_max", "mid_max", "trend_eps", "k", "walls", "corner_d", "near_px"):
        assert key in th, key
    assert th["throw_max"]["chunli"] == 47 and th["walls"] == [53, 459]
    assert th["undecided"]                       # what the data could not decide is listed, not hidden
    rs = frames([100] * 40, [140] * 40)
    lab = P.labels(rs, 9, th)
    assert lab["range"] == "throw" and P.UNKNOWN not in lab.values()


def test_k_ignores_his_guard_stance_before_the_contact():
    """The CPU raises its guard (state 0x08) as my attack starts, before any contact (seen in the smoke game): only
    block / hit stun, a throw or a life drop is contact."""
    c = cal()
    def rec(i):
        rows = []
        for j in range(40):
            st = 0x0E if j >= 8 + 6 else 0x08 if j >= 8 + 1 else 0
            rows.append(row({"x": 100, "char": 5}, {"x": 140, "state": st, "react": 0x06 if st == 0x0E else 0}))
        return ram_entry(0, i, rows, 8)
    recs = [rec(i) for i in range(6)]
    acts = [entry(action="c.hp", actual="blocked", frame=i) for i in range(6)]
    assert c.k_frames(acts, recs)["k"] == 6


def test_k_from_too_few_contacts_is_only_provisional():
    c = cal()
    recs = [ram_rec(0, i, s) for i, s in enumerate((4, 9))]
    acts = [entry(action=m, actual="hit", frame=i) for i, m in enumerate(("lp", "sweep"))]
    k = c.k_frames(acts, recs)
    assert k["k"] is None and k["provisional"] == 4 and k["contacts"] == 2
