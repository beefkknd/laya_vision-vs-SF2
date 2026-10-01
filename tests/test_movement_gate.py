"""The dataset gate (sf2.data.movement_gate, scripts/gate_movement_data.py; docs/prereg_movement_data.md gates 1-3,
the disk budget and the contact sheet): counts per (opponent, answer) in train and test, stage coverage, labels
re-derived from the stored RAM by an independent implementation, RAM-to-image alignment by the HUD clock, the image
bytes; the exit code decides."""
import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image
from ram_rows import row
from test_movement_collect_io import FOUR, SMALL, SPEC

from sf2.data import movement as M
from sf2.data import movement_collect_io as IO
from sf2.data import movement_data2 as D
from sf2.data import movement_gate as G
from sf2.emu.vs import GROUND_Y

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPPS = ("ken", "ryu")
LOOSE = {"min_train": 1, "min_test": 1, "min_disc": 1}


# ---- the independent label ----------------------------------------------------------------------------------------

def random_rows(rng, n=12):
    states = [0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E, 0x14, 0x10]
    out = []
    for _ in range(n):
        st = rng.choice(states)
        out.append(row({"x": rng.choice([0, 100, 200, 300, 512, 600, 65369])},
                       {"x": rng.choice([100, 199, 200, 201, 202, 300, 512, 513, 65369]), "state": st,
                        "react": rng.choice([0x00, 0x06, 0x08, 0x0E]),
                        "y": rng.choice([GROUND_Y, GROUND_Y, GROUND_Y - 1, 100])}))
    return out


def test_the_independent_label_agrees_with_movement_on_random_rows():
    rng = random.Random(0)
    seen = set()
    for _ in range(4000):
        rows = random_rows(rng)
        for t in range(len(rows)):
            a = G.independent_answer(rows, t)
            assert a == M.movement(rows, t), (t, rows[t], rows[t - 4] if t >= 4 else None)
            seen.add(a)
    assert seen == set(M.ANSWERS) | {"unknown"}


def test_the_independent_label_does_not_import_the_label_module():
    src = open(os.path.join(ROOT, "sf2", "data", "movement_gate.py")).read()
    body = src[src.index("def independent_answer"):src.index("# ---- end independent")]
    assert "M." not in body and "movement(" not in body and "perception" not in body


# ---- a synthetic collection with HUD clock pixels ---------------------------------------------------------------

def clock_image(timer):
    """A screen whose clock region shows ``timer`` as a pattern of the clock's blue outline."""
    im = np.zeros((224, 256, 3), np.uint8)
    y0, x0 = G.CLOCK[0].start, G.CLOCK[1].start
    for b in range(8):
        if timer >> b & 1:
            im[y0 + 2 * b, x0:x0 + 4] = (0, 0, 255)
    return im


def timed_stream(spec, tick=7):
    rows = []
    for ans, n in spec:
        over = {"standing": {}, "crouching": {"state": 0x02}, "jumping": {"state": 0x04, "y": GROUND_Y - 30},
                "attacking": {"state": 0x0A}}[ans]
        rows += [row({"x": 200}, dict({"x": 300}, **over)) for _ in range(n)]
    return [dict(r, timer=0x99 - k // tick) for k, r in enumerate(rows)]


def clock_play(lag=1):
    def play(game, sampler):
        rows = timed_stream(SPEC * 3)
        for k, r in enumerate(rows):
            shown = rows[max(0, k - lag)]["timer"]
            sampler.feed(r, None if k == 0 else clock_image(shown))
        return {"result": "win"}
    return play


def dataset(tmp_path, lag=1, targets=SMALL):
    root = str(tmp_path / "coll")
    for o in OPPS:
        IO.collect_opponent(os.path.join(root, o), clock_play(lag), opp=o, cap=100, seed=0, targets=targets,
                            answers=FOUR, rss_gb=lambda: 1.0, log=lambda *_: None)
    with open(os.path.join(root, "run.json"), "w") as f:
        json.dump({"quota": targets}, f)
    out = str(tmp_path / "mv2")
    assert D.build(root, out)["problems"] == []
    return out


def gates(out, **kw):
    return G.run_gates(out, answers=FOUR, **dict(LOOSE, **kw))


def test_a_clean_dataset_passes_every_gate(tmp_path):
    rep = gates(dataset(tmp_path))
    assert {k: v["pass"] for k, v in rep["gates"].items()} == {k: True for k in G.GATES}
    assert rep["pass"] is True


def test_gate1_lists_shortfalls_by_name(tmp_path):
    rep = gates(dataset(tmp_path), min_train=7, min_test=2)
    g = rep["gates"]["counts"]
    assert not g["pass"] and not rep["pass"]
    assert "ken standing train 6 < 7" in g["shortfalls"]


def test_gate1_reads_train_and_test_not_val(tmp_path):
    out = dataset(tmp_path, targets={"train": 6, "val": 50, "test": 1})
    rep = gates(out, min_test=2)
    assert any("test 1 < 2" in s for s in rep["gates"]["counts"]["shortfalls"])


def test_gate2_fails_when_a_stage_is_thin(tmp_path):
    out = dataset(tmp_path)
    path = os.path.join(out, "ken", "train.jsonl")
    rows = IO.read_jsonl(path)
    rows = [dict(r, stage_bin="start", pos=0) if r["answer"] == "crouching" else r for r in rows]
    with open(path, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    g = gates(out)["gates"]["stages"]
    assert not g["pass"] and any("ken crouching" in p for p in g["problems"])


def test_gate2_ignores_cells_without_six_frame_episodes():
    rows = [{"opp": "ken", "answer": "jumping", "length": 4, "pos": 0, "stage_bin": "start"}] * 5
    assert G.stage_problems(rows) == []


def test_gate3_catches_a_label_that_disagrees_with_the_ram(tmp_path):
    out = dataset(tmp_path)
    path = os.path.join(out, "ryu", "val.jsonl")
    rows = IO.read_jsonl(path)
    i = next(i for i, r in enumerate(rows) if r["answer"] == "standing")
    rows[i] = dict(rows[i], answer="crouching", label=M.ANSWERS.index("crouching"))
    with open(path, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))
    g = gates(out)["gates"]["labels"]
    assert not g["pass"] and g["mismatches"] == 1 and g["checked"] > 10


def test_gate3_catches_images_not_at_t_minus_4_and_t(tmp_path):
    out = dataset(tmp_path)
    path = os.path.join(out, "ryu", "train.jsonl")
    rows = IO.read_jsonl(path)
    r = rows[0]
    rows[0] = dict(r, images=["frames/g%04d_k%05d.png" % (r["game"], k) for k in (r["t"] - 4, r["t"])])
    with open(path, "w") as f:
        f.write("".join(json.dumps(x) + "\n" for x in rows))
    assert not gates(out)["gates"]["labels"]["pass"]


def test_alignment_passes_at_lag_one_and_fails_at_lag_zero(tmp_path):
    good = gates(dataset(tmp_path / "a", lag=1))["gates"]["alignment"]
    assert good["pass"] and good["agreement"]["1"] == 1.0 and good["discriminating"] >= 1
    bad = gates(dataset(tmp_path / "b", lag=0))["gates"]["alignment"]
    assert not bad["pass"] and bad["best_lag"] == 0


@pytest.mark.parametrize("ad, n_disc, missing, ok", [
    ({"0": 0.5, "1": 0.97, "2": 0.5}, 30, 0, True),
    ({"0": 0.98, "1": 0.97, "2": 0.5}, 30, 0, False),       # lag 0 explains the discriminating pairs better
    ({"0": 0.5, "1": 0.97, "2": 0.97}, 30, 0, False),       # a tie is not lag 1
    ({"0": 0.5, "1": 0.85, "2": 0.5}, 30, 0, False),        # best, but below min_agree_disc
    ({"0": 0.5, "1": 0.97, "2": 0.5}, 5, 0, False),         # too few discriminating pairs
    ({"0": 0.5, "1": 0.97, "2": 0.5}, 30, 1, False),        # an image missing
])
def test_alignment_verdict(ad, n_disc, missing, ok):
    ag = {"0": 0.9, "1": 0.99, "2": 0.9}
    assert G.alignment_verdict(ag, ad, 400, n_disc, missing, 20, 0.95, 0.9) is ok
    assert G.alignment_verdict(dict(ag, **{"1": 0.94}), ad, 400, n_disc, missing, 20, 0.95, 0.9) is False


def test_alignment_without_evidence_fails(tmp_path):
    g = gates(dataset(tmp_path), min_disc=10 ** 6)["gates"]["alignment"]
    assert not g["pass"]


def test_disk_gate(tmp_path):
    out = dataset(tmp_path)
    assert gates(out)["gates"]["disk"]["pass"]
    g = gates(out, max_gb=1e-9)["gates"]["disk"]
    assert not g["pass"] and g["gb"] > 0


def test_contact_sheet_has_a_row_per_answer(tmp_path):
    out = dataset(tmp_path)
    paths = G.contact_sheets(out, str(tmp_path / "sheets"), per_answer=2, answers=FOUR)
    assert sorted(os.path.basename(p) for p in paths) == ["contact_ken.png", "contact_ryu.png"]
    im = Image.open(paths[0])
    assert im.size[1] >= len(FOUR) * G.THUMB and im.size[0] >= 2 * 2 * G.THUMB


def test_the_script_exit_code_decides(tmp_path):
    out = dataset(tmp_path)
    script = os.path.join(ROOT, "scripts", "gate_movement_data.py")
    ok = subprocess.run([sys.executable, script, "--data", out, "--answers", ",".join(FOUR), "--min-train", "1",
                         "--min-test", "1", "--min-disc", "1", "--sheets", str(tmp_path / "s")],
                        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert json.load(open(os.path.join(out, "gate.json")))["pass"] is True
    bad = subprocess.run([sys.executable, script, "--data", out, "--answers", ",".join(FOUR)],
                         capture_output=True, text=True)
    assert bad.returncode == 1 and "FAIL counts" in bad.stdout


def test_a_missing_ram_file_fails_the_label_gate(tmp_path):
    out = dataset(tmp_path)
    meta = json.load(open(os.path.join(out, "build.json")))
    os.remove(os.path.join(meta["root"], "ken", "ram", "g0001.json.gz"))
    g = gates(out)["gates"]["labels"]
    assert not g["pass"] and g["missing_ram"]


@pytest.mark.parametrize("bad", [{"min_train": -1}, {"min_share": 1.5}])
def test_bad_thresholds_are_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        G.run_gates(str(tmp_path), **bad)
