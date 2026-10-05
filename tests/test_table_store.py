"""Per-character value-table store (owner 2026-10-05: "support multiple characters and a storage design"). One JSON
per character under a root dir (the unchanged value_table structure, so VT math is untouched) + a manifest tracking
each character's accumulated games / last-updated / source run / cell count. This module only resolves paths,
loads/saves by character, and pools a run's worker tables into the stored one (seed-dedup). Pure path + JSON I/O."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from sf2.system1 import table_store as TS                                 # noqa: E402
from sf2.system1 import value_table as VT                                 # noqa: E402


def acc(n, mean):
    return [n, n * mean, n * mean * mean]


def tbl(cells):
    t = VT.blank()
    t["cells"] = {w: {a: list(s) for a, s in mv.items()} for w, mv in cells.items()}
    return t


def test_path_resolves_per_character_under_the_root():
    assert TS.path("ryu", root="runs/tables").endswith(os.path.join("runs", "tables", "ryu.json"))


def test_load_absent_character_is_blank():
    t = TS.load("nobody", root="/tmp/does-not-exist-xyz")
    assert t == VT.blank()


def test_save_then_load_round_trips_the_table(tmp_path):
    root = str(tmp_path / "tables")
    t = tbl({"close|standing|0": {"throw_F+mp": acc(40, 25.0)}})
    TS.save("chunli", t, root=root, games=20, run="A2")
    assert TS.load("chunli", root=root) == t


def test_manifest_accumulates_games_and_records_cells_and_run(tmp_path):
    root = str(tmp_path / "tables")
    TS.save("chunli", tbl({"mid|standing|0": {"s.mp": acc(30, 3.0)}}), root=root, games=20, run="A2")
    TS.save("chunli", tbl({"mid|standing|0": {"s.mp": acc(30, 3.0)}, "far|standing|0": {"s.hk": acc(10, 1.0)}}),
            root=root, games=12, run="A3")
    man = json.load(open(TS.manifest_path(root)))
    assert man["chunli"]["games"] == 32                    # accumulates across runs
    assert man["chunli"]["cells"] == 2 and man["chunli"]["run"] == "A3"


def test_two_characters_are_isolated_files(tmp_path):
    root = str(tmp_path / "tables")
    TS.save("chunli", tbl({"mid|standing|0": {"s.mp": acc(30, 3.0)}}), root=root, games=20, run="A2")
    TS.save("ryu", tbl({"mid|standing|0": {"hadoken": acc(30, 4.0)}}), root=root, games=10, run="R1")
    assert os.path.exists(TS.path("chunli", root)) and os.path.exists(TS.path("ryu", root))
    assert set(json.load(open(TS.manifest_path(root)))) == {"chunli", "ryu"}
    assert "hadoken" not in json.dumps(TS.load("chunli", root=root))     # isolation


def test_pool_into_seed_dedups_parallel_workers(tmp_path):
    root = str(tmp_path / "tables")
    seed = tbl({"close|standing|0": {"throw_F+mp": acc(40, 25.0)}})       # the common carried seed
    workers = [seed, seed]                                                # two workers that merely carried it
    merged = TS.pool_into("chunli", workers, seed=seed, root=root, games=24, run="A2")
    assert merged["cells"]["close|standing|0"]["throw_F+mp"][0] == 40     # 40, not 80 (dedup)
    assert TS.load("chunli", root=root)["cells"]["close|standing|0"]["throw_F+mp"][0] == 40
