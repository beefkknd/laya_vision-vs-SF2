"""docs/eye_questions_v1.md, "Next: more fireball data": the per-round q1 fill line (sf2.data.eye_q1_fill) must count
exactly what the q1 build (sf2.data.eye_data, v1.1 rules and matching) would select, per answer and split; name the
cells not full (shot-sample cells under the per-game cap in the new games) and the train strata whose "yes" rows lack a
matching "no"; and scripts/collect_pairs.py must pass a raised --shot-per-game to the projectile sampler."""
import collections
import importlib.util
import json
import os
import subprocess
import sys

from sf2.data import eye_data as E
from sf2.data import eye_q1_fill as F
from sf2.data import pairs_shots as S
from sf2.data import pairs_train as T
from test_eye_data import OLD, pool, root  # noqa: F401  (module fixtures: ryu/ken games with shots)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_fill_counts_what_the_q1_build_selects(root, pool, tmp_path):  # noqa: F811
    rep = F.fill(pool, root)
    out = str(tmp_path / "q1")
    E.build(pool, "q1", out, root)
    got = collections.Counter((r["answer"], f) for a, fs in E.read_dataset(out).items() for f, rs in fs.items()
                              for r in rs)
    assert rep["counts"] == {a: {f: got[(a, f)] for f in T.FILES} for a in ("yes", "no")}
    assert rep["counts"]["yes"] == rep["counts"]["no"] and sum(rep["counts"]["yes"].values()) > 0


def test_strata_short_of_a_no_and_the_yes_rows_lost(root, pool):  # noqa: F811
    rep = F.fill(pool, root)
    cands, _ = E.cands_q1(pool, set())
    per = collections.defaultdict(collections.Counter)
    for c in cands:
        if c["split"] == "train":
            per[c["stratum"]][c["answer"]] += 1
    short = {k: v for k, v in per.items() if v["yes"] > v["no"]}
    assert rep["short_strata"] == len(short)
    assert rep["yes_lost"] == sum(v["yes"] - v["no"] for v in short.values())
    assert rep["yes_candidates"]["train"] == sum(v["yes"] for v in per.values())


def test_shot_cells_not_full_count_the_new_games_only(root):  # noqa: F811
    from sf2.data import pairs_collect_io as IO
    full = F.shot_cells(root, from_game=0, cap=1)
    none = F.shot_cells(root, from_game=10 ** 6, cap=1)
    assert none == {"cells": 0, "not_full": 0}
    games = sum(len(IO.committed(os.path.join(root, n))) for n in os.listdir(root))
    # every game of the two thrower pairs: 2 slots (both throwers) x 3 stages
    assert full["cells"] == games * 2 * 3
    taken = collections.Counter()
    for n in os.listdir(root):
        for s in S.committed_shots(os.path.join(root, n))[0]:
            taken[(n, s["game"], s["slot"], s["flight_stage"])] += 1
    assert full["not_full"] == full["cells"] - sum(1 for v in taken.values() if v >= 1)
    big = F.shot_cells(root, from_game=0, cap=10 ** 3)
    assert big["not_full"] == big["cells"]


def test_met_and_the_line():
    rep = {"counts": {"yes": {"train": 2000, "val": 1, "test": 1}, "no": {"train": 2000, "val": 1, "test": 1}},
           "short_strata": 3, "yes_lost": 9, "yes_candidates": {"train": 2009, "val": 1, "test": 1},
           "shots": {"cells": 10, "not_full": 4}, "seconds": 1.5}
    assert F.met(rep, 2000) and not F.met(rep, 2001)
    rep["counts"]["no"]["train"] = 1999
    assert not F.met(rep, 2000)
    line = F.line(rep, 2000)
    assert "yes 2000" in line and "no 1999" in line and "not full" in line and "4/10" in line and "not met" in line


def test_fill_cli_appends_the_report_and_exits_2_when_not_met(root, tmp_path):  # noqa: F811
    log = str(tmp_path / "fill.jsonl")
    r = subprocess.run([sys.executable, os.path.join(HERE, "scripts", "fill_eye_q1.py"), "--root", root, "--log", log,
                        "--workers", "1", "--target", "100000", "--label", "games 14", "--prefer", "none"],
                       capture_output=True, text=True, cwd=HERE)
    assert r.returncode == 2, r.stderr[-2000:]
    assert r.stdout.startswith("games 14 q1 train yes")
    rep = json.loads(open(log).read().splitlines()[-1])
    assert rep["label"] == "games 14" and rep["counts"]["yes"]["train"] == rep["counts"]["no"]["train"]


def _collect_pairs():
    sys.path.insert(0, os.path.join(HERE, "scripts"))
    spec = importlib.util.spec_from_file_location("collect_pairs", os.path.join(HERE, "scripts", "collect_pairs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_collect_pairs_passes_the_shot_cap_to_the_sampler():
    import random
    cp = _collect_pairs()
    cls = cp.sampler_of(True, 7)
    s = cls(0, {1: "ryu", 2: "ken"}, random.Random(0), lambda k, im: "x", {"all": 72}, 120, 3)
    assert isinstance(s, S.ShotSampler) and s.shot_per_game == 7
    assert cp.sampler_of(True, S.SHOT_PER_GAME)(0, {1: "ryu", 2: "ken"}, random.Random(0), lambda k, im: "x",
                                                {"all": 72}, 120, 3).shot_per_game == S.SHOT_PER_GAME
    from sf2.data import pairs_collect as PC
    assert cp.sampler_of(False, 7) is PC.PairSampler


def test_collect_pairs_records_and_forwards_the_shot_cap():
    cp = _collect_pairs()
    args = cp.parser().parse_args(["--mode", "vs", "--shots", "--shot-per-game", "6", "--out", "x", "--games", "33"])
    assert args.shot_per_game == 6
    assert cp.run_record(args, [("ryu", "ken")])["shot_per_game"] == 6
    common = cp.common_args(args)
    assert common[common.index("--shot-per-game") + 1] == "6"
    assert cp.parser().parse_args([]).shot_per_game == S.SHOT_PER_GAME


def test_shot_cells_only_for_thrower_slots(tmp_path):
    """blanka (no projectile) vs ryu: one cell per stage for ryu's slot only; a full cell needs ``cap`` samples."""
    base = tmp_path / "blanka_vs_ryu"
    base.mkdir()
    (base / "games.jsonl").write_text("".join(json.dumps({"game": g}) + "\n" for g in (31, 32)))
    shots = [{"game": 32, "slot": 2, "flight_stage": "start"}] * 2 + [{"game": 32, "slot": 2, "flight_stage": "end"}]
    (base / "shots.jsonl").write_text("".join(json.dumps(s) + "\n" for s in shots))
    assert F.shot_cells(str(tmp_path), from_game=32, cap=2) == {"cells": 3, "not_full": 2}
    assert F.shot_cells(str(tmp_path), from_game=31, cap=1) == {"cells": 6, "not_full": 4}
