"""scripts/train.py --init (docs/prereg_u_round2.md): continue from a checkpoint saved by train.py instead of BASE; its
tags must agree with what the data teaches; the start point is recorded. Default: BASE, unchanged."""
import importlib.util
import json
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
spec = importlib.util.spec_from_file_location("train_script", os.path.join(HERE, "..", "scripts", "train.py"))
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

TAGS = {"note_version": 3, "value_questions": False, "perception": True}


def ckpt(tmp_path, cfg):
    d = tmp_path / "best"
    d.mkdir()
    (d / "vlm_agent_config.json").write_text(json.dumps(cfg))
    (d / "model.safetensors").write_bytes(b"weights")
    return str(d)


def test_default_is_base_unchanged():
    args = tr.parse_args(["--data", "x", "--out", "y"])
    assert args.init is None
    path, kwargs = tr.init_source(args)
    assert path == tr.BASE_MODEL and kwargs == tr.IMAGE_CFG


def test_init_from_checkpoint_loads_it_without_overriding_its_prep(tmp_path):
    c = ckpt(tmp_path, dict(TAGS))
    args = tr.parse_args(["--data", "x", "--out", "y", "--init", c])
    assert tr.init_source(args) == (c, {})


def test_init_tags_must_agree_with_the_data(tmp_path):
    tr.check_init(ckpt(tmp_path, dict(TAGS)), TAGS)            # agrees: no error
    bad = tmp_path / "other"
    bad.mkdir()
    with pytest.raises(SystemExit):
        tr.check_init(ckpt(bad, {"note_version": 2, "value_questions": True}), TAGS)


def test_init_must_be_a_train_py_checkpoint(tmp_path):
    with pytest.raises(SystemExit):
        tr.check_init(str(tmp_path / "missing"), TAGS)


def test_init_record_names_path_and_weights_hash(tmp_path):
    c = ckpt(tmp_path, dict(TAGS))
    rec = tr.init_record(c)
    assert rec["path"] == c and len(rec["weights_sha256"]) == 64
    assert tr.init_record(None) is None
