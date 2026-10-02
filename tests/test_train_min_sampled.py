"""scripts/train.py --min-sampled-train / --min-sampled-val (round 3, docs/prereg_movement_finetunes.md)."""
import importlib.util
import os
from types import SimpleNamespace

from sf2.data import train_data as TD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _train_module():
    import sys
    if os.path.join(ROOT, "scripts") not in sys.path:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
    spec = importlib.util.spec_from_file_location("train_script", os.path.join(ROOT, "scripts", "train.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rows(name, n, split):
    return [{"dataset": name, "split": split, "id": "%s-%s-%d" % (name, split, i)} for i in range(n)]


def test_defaults_are_the_old_minimums():
    ap_args = _train_module()
    assert TD.MIN_SAMPLED_TRAIN == 1000 and TD.MIN_SAMPLED_VAL == 100
    assert ap_args.MIN_SAMPLED_TRAIN == 1000


def test_lowered_minimum_passes_and_default_fails():
    dirs = ["d/a", "d/b"]
    train = _rows("a", 300, "train") + _rows("b", 300, "train")
    val = _rows("a", 40, "val") + _rows("b", 40, "val")
    assert TD.sampling_problems(train, val, dirs)                                  # 1000 / 100 -> refused
    assert not [p for p in TD.sampling_problems(train, val, dirs, 250, 35) if "rows" in p]


def test_train_passes_the_flags_through(monkeypatch):
    mod = _train_module()
    seen = {}

    def fake(train, val, dirs, min_train=0, min_val=0):
        seen["mins"] = (min_train, min_val)
        return []
    monkeypatch.setattr(mod, "sampling_problems", fake)
    monkeypatch.setattr(mod, "coverage_problems", lambda *a, **k: [])
    args = SimpleNamespace(balance="sampling", data=["d/a"], min_sampled_train=250, min_sampled_val=35)
    mod.data_problems([], [], args)
    assert seen["mins"] == (250, 35)
