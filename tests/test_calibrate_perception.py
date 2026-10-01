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


# ---- the collection (--ram-root) and reproducible sources ----

def write_dir(d, recs, acts):
    d.mkdir(parents=True)
    (d / "ram.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    (d / "actions.jsonl").write_text("".join(json.dumps(a) + "\n" for a in acts))


def test_ram_root_joins_actions_per_directory_not_across_them(tmp_path):
    """rollouts/u_perception/<opp>/<me>/: (game, frame) repeats across directories; each joins its own actions."""
    c = cal()
    write_dir(tmp_path / "ryu" / "chunli", [ram_rec(0, i, 5) for i in range(6)],
              [entry(action="lp", actual="hit", frame=i) for i in range(6)])
    write_dir(tmp_path / "ken" / "chunli", [ram_rec(0, i, 9) for i in range(6)],
              [entry(action="sweep", actual="hit", frame=i) for i in range(6)])
    k = c.k_from_root(str(tmp_path))
    assert k["by_move"]["chunli"] == {"lp": [5.0, 6], "sweep": [9.0, 6]} and k["k"] == 5


def test_ram_root_rows_count_each_frame_once():
    """Decision windows overlap (60 back, 60 ahead): a frame seen by several records counts once for the walls."""
    c = cal()
    rows = [row({"x": 53 + (i % 3), "char": 5}, {"x": 459, "char": 0}) for i in range(100)]
    recs = [ram_entry(0, f, rows[max(0, f - 30):f + 31], min(f, 30)) for f in range(0, 100, 10)]
    got = list(c.frames_of(recs))
    assert len(got) == 100 and [r["p1_x"] for r in got] == [r["p1_x"] for r in rows]


def test_sources_carry_sha256_not_just_paths(tmp_path):
    c = cal()
    f = tmp_path / "rows.jsonl"
    f.write_text("abc\n")
    src = c.source([str(f)])
    assert src == [{"path": str(f), "sha256": "edeaaff3f1774ad2888673770c6d64097e391bc362d7d6fb34982ddf0efd18cb",
                    "bytes": 4}]
    d = tmp_path / "probe"
    (d / "img").mkdir(parents=True)
    (d / "img" / "a.png").write_bytes(b"x")
    (d / "rows.jsonl").write_text("abc\n")
    one = c.source([str(d)])[0]
    (d / "img" / "a.png").write_bytes(b"y")
    assert c.source([str(d)])[0]["sha256"] != one["sha256"] and one["files"] == 2


def test_main_reads_a_collection_root_for_k_and_walls(tmp_path, monkeypatch):
    c = cal()
    logs = tmp_path / "logs" / "a" / "chunli"
    logs.mkdir(parents=True)
    es = [entry(gap=g, thrown=True) for g in range(10, 48)] + [entry(gap=g) for g in range(48, 120)]
    (logs / "actions.jsonl").write_text("".join(json.dumps(e) + "\n" for e in es))
    root = tmp_path / "u_perception"
    write_dir(root / "ryu" / "chunli", [ram_rec(0, i, 6) for i in range(6)],
              [entry(action="lp", actual="hit", frame=i) for i in range(6)])
    out = tmp_path / "v2.json"
    monkeypatch.setattr(sys, "argv", ["calibrate_perception.py", "--logs", str(tmp_path / "logs" / "*" / "*" /
                                      "actions.jsonl"), "--ram-root", str(root), "--out", str(out), "--no-q8"])
    assert c.main() == 0
    th = json.loads(out.read_text())
    assert th["k"] == 6 and "k" not in th["undecided"]
    assert th["walls"] == [100, 140] and th["sources"]["ram_root"]["path"] == str(root)


def test_walls_ignore_impossible_x_and_lone_transients():
    """v2 bug: one frame of Blanka at x 65369 (-167 wrapped) made the right wall 65369."""
    c = cal()
    rows = (frames([53] * 300 + [200] * 300, [459] * 300 + [300] * 300) +
            frames([65369] + [200] * 50, [300] * 51, p1_char=2) +              # Blanka: the wrapped frame
            frames([468] * 2 + [200] * 50, [300] * 52, p1_char=2))             # and a 2-frame transient at 468
    w = c.walls(rows)
    assert w["walls"] == [53, 459] and w["corner_d"] == 0
    assert "hi" not in w["by_char"]["blanka"]


def test_walls_refuse_an_impossible_wall():
    c = cal()
    with pytest.raises(SystemExit):
        c.walls(frames([300] * 10, [310] * 10))          # no pile-up anywhere: no wall
