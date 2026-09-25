"""Training loop plumbing: val sizing, time budgets, queue scheduling, collection failures. No model or ROM needed."""
import argparse
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import parallel  # noqa: E402
from sf2 import headless  # noqa: E402
import train  # noqa: E402
import training_queue as tq  # noqa: E402


def test_val_limit_caps_the_held_out_slice_when_no_val_jsonl():
    # every Chun-Li dataset lacks val.jsonl, so this fallback is the path real runs take
    tr, val = train.split_val(list(range(34274)), [], limit=600, seed=0)
    assert len(val) == 600
    assert len(tr) + len(val) == 34274  # frames not held out go back to training
    assert not set(tr) & set(val)


def test_val_limit_caps_an_explicit_val_split():
    tr, val = train.split_val(list(range(100)), list(range(1000, 3000)), limit=600, seed=0)
    assert len(val) == 600 and len(tr) == 100


def test_training_budget_covers_load_eval0_and_the_final_eval():
    # 30 min wall budget; load + eval 0 took 4.5 min; one eval takes 90 s
    budget = train.training_minutes(30.0, spent_s=270, eval_s=90)
    assert 4.5 + budget + 1.5 <= 30.0
    assert budget > 20.0


def test_training_budget_never_goes_to_zero():
    assert train.training_minutes(5.0, spent_s=600, eval_s=90) >= 1.0


def _create(tmp_path, collect_name, train_data, rom="ROM"):
    rom_path = tmp_path / "sf2.sfc"
    rom_path.write_bytes(b"")
    mesen = tmp_path / "Mesen"
    mesen.write_bytes(b"")
    args = argparse.Namespace(queue=str(tmp_path / "q.json"), minutes=30, collect_name=collect_name,
                              collect_decisions=100, base_port=47940, eps=0.2, savestate="states/x.state",
                              me="chunli", opp="dhalsim", train_out="runs/zz_test", train_init="runs/init",
                              train_data=train_data, rom=str(rom_path) if rom else None, mesen=str(mesen))
    tq.create(args)
    return {t["id"]: t for t in tq.load(tmp_path / "q.json")["tasks"]}


def test_train_runs_alongside_a_collection_it_does_not_use(tmp_path):
    tasks = _create(tmp_path, "seed_r5", ["data/seed_r4", "data/dagger_r2"])
    assert tasks["train"]["depends"] == []


def test_train_waits_for_a_collection_it_trains_on(tmp_path):
    tasks = _create(tmp_path, "seed_r5", ["data/seed_r4", "data/seed_r5/"])
    assert tasks["train"]["depends"] == ["collect"]


def test_train_finishes_before_the_queue_deadline_kills_it(tmp_path):
    tasks = _create(tmp_path, "seed_r5", ["data/seed_r4"])
    argv = tq.command(tasks["train"], remaining=1800)
    assert float(argv[argv.index("--max-minutes") + 1]) < 30.0


def test_collection_is_told_where_the_rom_is(tmp_path):
    tasks = _create(tmp_path, "seed_r5", ["data/seed_r4"])
    argv = tq.command(tasks["collect"], remaining=1800)
    assert argv[argv.index("--rom") + 1] == str(tmp_path / "sf2.sfc")


def test_collection_is_told_where_mesen_is(tmp_path):
    tasks = _create(tmp_path, "seed_r5", ["data/seed_r4"])
    argv = tq.command(tasks["collect"], remaining=1800)
    assert argv[argv.index("--mesen") + 1] == str(tmp_path / "Mesen")


def test_find_mesen_checks_the_per_user_applications_folder(tmp_path, monkeypatch):
    # a drag-installed "Mesen 2.app" in ~/Applications, with SF2_MESEN unset
    exe = tmp_path / "Applications" / "Mesen 2.app" / "Contents" / "MacOS" / "Mesen"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SF2_MESEN", raising=False)
    monkeypatch.setattr(headless, "MAC_MESEN", str(tmp_path / "nowhere" / "Mesen"))
    assert headless.find_mesen() == str(exe)


def test_queue_refuses_a_collection_without_a_rom(tmp_path, monkeypatch):
    monkeypatch.delenv("SF2_ROM", raising=False)
    with pytest.raises(SystemExit):
        _create(tmp_path, "seed_r5", ["data/seed_r4"], rom=None)
    assert not (tmp_path / "q.json").exists()


class _FailedWorker:
    def __init__(self, *a, **kw):
        pass

    def wait(self):
        return 1


def test_all_workers_failing_leaves_no_ready_dataset(tmp_path, monkeypatch):
    # a merged-but-empty dataset marked _READY looks trainable and blocks re-running the name
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(parallel.subprocess, "Popen", _FailedWorker)
    monkeypatch.setattr(sys, "argv", ["parallel.py", "--workers", "2", "collect_teacher", "--name", "seed_x",
                                      "--decisions", "10"])
    with pytest.raises(SystemExit) as e:
        parallel.main()
    assert e.value.code
    assert not (tmp_path / "data" / "seed_x" / "_READY").exists()
