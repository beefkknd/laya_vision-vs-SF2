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


def _frames(n_rounds, per_round, dataset="d"):
    exs, info = [], {}
    for r in range(n_rounds):
        for f in range(per_round):
            i = "%s-%d-%d" % (dataset, r, f)
            exs.append({"id": i, "dataset": dataset})
            info[(dataset, i)] = {"id": i, "dataset": dataset, "episode": 0,
                                  "meta": {"episode": 0, "round": r, "frame": f}}
    return exs, info


def test_fallback_val_holds_out_whole_rounds_and_keeps_them_out_of_training():
    # no val.jsonl anywhere (true of every Chun-Li dataset): neighbouring frames must not straddle the split
    exs, info = _frames(40, 50)
    tr, val = train.split_val(exs, [], limit=600, seed=0, info=info)
    val_rounds = {train.info_for(info, e)["meta"]["round"] for e in val}
    train_rounds = {train.info_for(info, e)["meta"]["round"] for e in tr}
    assert val and not val_rounds & train_rounds
    assert len(val) <= 600


def test_val_limit_caps_an_explicit_val_split():
    exs, info = _frames(4, 25)
    val = [{"id": "v%d" % i} for i in range(2000)]
    tr, v = train.split_val(exs, val, limit=600, seed=0, info=info)
    assert len(v) == 600 and len(tr) == 100


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
    stdout = iter(())  # parallel.py reads each worker's output from its pipe

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


def test_runner_status_lines_land_in_the_training_log(tmp_path, monkeypatch):
    # people follow one file with tail -f; queue events belong there next to the task output
    log = tmp_path / "training.log"
    monkeypatch.setattr(tq, "LOG", log)
    monkeypatch.setattr(tq, "command", lambda task, remaining: [sys.executable, "-c", "print('task output')"])
    q = tmp_path / "q.json"
    tq.save(q, {"version": 1, "state": "running", "created_at": 0, "deadline": 4e9, "history": [],
                "tasks": [{"id": "collect", "kind": "collect", "resource": "cpu", "slots": 6, "state": "pending"}]})
    tq.run(q, poll_seconds=0.05)
    text = log.read_text()
    for line in ("task output", "started collect", "collect done", "queue complete"):
        assert line in text


def test_queue_without_minutes_has_no_deadline_and_train_runs_to_the_end(tmp_path):
    rom, mesen = tmp_path / "sf2.sfc", tmp_path / "Mesen"
    rom.write_bytes(b"")
    mesen.write_bytes(b"")
    args = argparse.Namespace(queue=str(tmp_path / "q.json"), minutes=None, collect_name="seed_r5",
                              collect_decisions=100, base_port=47940, eps=0.2, savestate="states/x.state",
                              me="chunli", opp="dhalsim", train_out="runs/zz_test", train_init="runs/init",
                              train_data=["data/seed_r4"], rom=str(rom), mesen=str(mesen))
    tq.create(args)
    queue = tq.load(tmp_path / "q.json")
    assert queue["deadline"] is None
    train_task = next(t for t in queue["tasks"] if t["id"] == "train")
    assert "--max-minutes" not in tq.command(train_task, remaining=None)


def test_runner_finishes_a_queue_with_no_deadline(tmp_path, monkeypatch):
    monkeypatch.setattr(tq, "LOG", tmp_path / "training.log")
    monkeypatch.setattr(tq, "command", lambda task, remaining: [sys.executable, "-c", "pass"])
    q = tmp_path / "q.json"
    tq.save(q, {"version": 1, "state": "running", "created_at": 0, "deadline": None, "history": [],
                "tasks": [{"id": "collect", "kind": "collect", "resource": "cpu", "slots": 6, "state": "pending"}]})
    tq.run(q, poll_seconds=0.05)
    assert tq.load(q)["state"] == "done"


def test_parallel_workers_write_inside_their_batch_dir():
    # a batch must be one directory: copied from another machine, its image paths must still resolve
    dirs = parallel.worker_dirs("data", "seed_x", 3)
    assert all(os.path.dirname(d) == os.path.join("data", "seed_x") for d in dirs)
    assert len(set(dirs)) == 3


def test_worker_names_carry_the_batch_name_so_row_ids_stay_unique():
    a = parallel.worker_dirs("data", "batch_a", 2)
    b = parallel.worker_dirs("data", "batch_b", 2)
    assert not {os.path.basename(d) for d in a} & {os.path.basename(d) for d in b}


def test_a_subset_sharing_ids_keeps_each_datasets_own_metadata():
    # dagger_hot holds copies of some dagger rows (same ids); its annotation must not overwrite dagger's
    rows = lambda: [{"id": "e0-%d" % f, "episode": 0, "meta": {"episode": 0, "round": 0, "frame": f}}  # noqa: E731
                    for f in range(0, 200, 4)]
    info = {}
    train.add_info(info, "dagger", rows())
    train.add_info(info, "dagger_hot", [r for r in rows() if r["meta"]["frame"] >= 100])
    rec = train.info_for(info, {"id": "e0-120", "dataset": "dagger"})
    assert rec["dataset"] == "dagger" and rec["meta"]["t_round"] == 2.0        # 120 frames into the round
    assert train.info_for(info, {"id": "e0-120", "dataset": "dagger_hot"})["meta"]["t_round"] == 20 / 60


def test_each_selection_metric_keeps_its_own_best_step():
    # accuracy peaks at step 500, soft cross-entropy (lower is better) at 1000, t_of_pred at 1500
    best = {}
    assert sorted(train.track_best(best, {"acc": 0.50, "xent": 1.20, "tpred": 0.40}, 0)) == ["acc", "tpred", "xent"]
    assert train.track_best(best, {"acc": 0.60, "xent": 1.30, "tpred": 0.39}, 500) == ["acc"]
    assert train.track_best(best, {"acc": 0.55, "xent": 1.10, "tpred": 0.38}, 1000) == ["xent"]
    assert train.track_best(best, {"acc": 0.55, "xent": 1.15, "tpred": 0.45}, 1500) == ["tpred"]
    assert train.track_best(best, {"acc": 0.55, "xent": 1.15, "tpred": 0.45}, 2000) == []  # ties do not count
    assert {k: v["step"] for k, v in best.items()} == {"acc": 500, "xent": 1000, "tpred": 1500}


def test_the_selected_metric_saves_best_and_the_others_best_name():
    assert train.checkpoint_name("xent", select="xent") == "best"
    assert train.checkpoint_name("acc", select="xent") == "best_acc"
