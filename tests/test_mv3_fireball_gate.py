"""The fireball dataset's gates (sf2.data.mv3_fireball_gate): labels re-derived (0 problems), the projectile drawn in
the frame its label says (hadoken blue test both ways), the slot ownership report, contact sheets."""
import json
import os

import numpy as np
import pytest
from PIL import Image

from sf2.data import mv3_fireball as F
from sf2.data import mv3_fireball_gate as G
from test_mv3_fireball import src  # noqa: F401  (the module fixture)


def _png(path, blue):
    a = np.zeros((256, 256, 3), np.uint8)
    if blue:
        a[120:140, 100:130] = (60, 120, 255)
    a[10:30, 10:40] = (60, 120, 255)                      # blue in the HUD band: never counted
    Image.fromarray(a).save(path)
    return path


def test_blue_pixels_below_the_hud_only(tmp_path):
    assert G.blue_pixels(_png(str(tmp_path / "a.png"), True)) == 600
    assert G.blue_pixels(_png(str(tmp_path / "b.png"), False)) == 0


@pytest.fixture(scope="module")
def built(src, tmp_path_factory):  # noqa: F811
    s, root = src
    out = str(tmp_path_factory.mktemp("fg") / "fire")
    F.build_fireball(s, root, out)
    return out


def test_the_gates_pass_on_a_clean_build(built):
    rep = G.run_gates(built, min_disc=0)
    for g in ("labels", "drawn", "disk"):
        assert rep["gates"][g]["pass"], json.dumps(rep["gates"][g])[:800]
    d = rep["gates"]["drawn"]
    assert d["hadoken_rows"] > 0 and d["hadoken_drawn"] == d["hadoken_rows"] and d["none_clear"] == d["none_rows"] > 0
    own = rep["ownership"]
    # 4 shot games x 2 pairs x 5 flights; the shoryuken flight: only the other player pressed a projectile
    assert own == {"flights": 40, "own": 32, "other_only": 8}


def test_drawn_fails_when_the_labelled_frame_has_no_projectile(built):
    rows = G.rows_of(built)
    had = [r for r in rows if r["answer"] != "none"]
    bad = [dict(r, images=[r["images"][0], r["images"][0]]) for r in had]   # an earlier frame (t - 4)
    blank = sum(G.blue_pixels(os.path.join(built, r["_dir"], r["images"][1])) < G.BLUE_MIN for r in bad)
    assert blank > 0
    res = G.drawn_check(built, bad + [r for r in rows if r["answer"] == "none"], need=1.0)
    assert not res["pass"]


def test_drawn_fails_when_a_none_row_shows_a_projectile(built):
    rows = G.rows_of(built)
    none = [r for r in rows if r["answer"] == "none"]
    had = [r for r in rows if r["answer"] != "none"]
    fake = [dict(none[0], images=had[0]["images"], _dir=had[0]["_dir"])]
    assert not G.drawn_check(built, had + none[1:] + fake, need=1.0)["pass"]


def test_contact_sheets_one_per_thrower_and_none(built, tmp_path):
    paths = G.contact_sheets(built, str(tmp_path / "c"))
    assert sorted(os.path.basename(p) for p in paths) == ["fireball_%s.png" % c for c in
                                                         ("dhalsim", "guile", "ken", "none", "ryu")]


def test_the_labels_gate_fails_on_a_wrong_row(src, tmp_path):  # noqa: F811
    s, root = src
    out = str(tmp_path / "fire")
    F.build_fireball(s, root, out)
    path = os.path.join(out, "left", "train.jsonl")
    rows = [json.loads(x) for x in open(path)]
    rows[0] = dict(rows[0], label=2)
    with open(path, "w") as f:
        f.writelines(json.dumps(x) + "\n" for x in rows)
    rep = G.run_gates(out, min_disc=0)
    assert not rep["gates"]["labels"]["pass"] and not rep["pass"]


def test_ownership_reads_each_slot_by_its_own_character(tmp_path):
    root = tmp_path / "col"
    (root / "guile_vs_blanka").mkdir(parents=True)
    flights = [{"slot": 1, "start": 5, "end": 9, "spawn_words": {"1": "sonic_boom", "2": None}},
               {"slot": 2, "start": 15, "end": 19, "spawn_words": {"1": "sonic_boom", "2": "electricity"}},
               {"slot": 2, "start": 25, "end": 29, "spawn_words": {"1": "lp", "2": "rolling_attack"}}]
    (root / "guile_vs_blanka" / "games.jsonl").write_text(json.dumps({"game": 0, "flights": flights}) + "\n")
    assert G.ownership(str(root)) == {"flights": 3, "own": 1, "other_only": 1, "neither": 1}


def test_alignment_leaves_out_pairs_with_the_clock_at_or_below_20(built):
    """The fixture's clock is the row index mod 256: rows showing a clock <= 0x20 are left out, the rest kept."""
    rows = G.rows_of(built)
    root = json.load(open(os.path.join(built, "build.json")))["root"]
    want = sum(min((r["k_prev"] - 1) % 256, (r["k_now"] - 1) % 256) <= 0x20 for r in rows)
    res = G.alignment_check(built, root, rows, min_disc=0)
    assert 0 < want < len(rows) and res["left_out_clock_blink"] == want
