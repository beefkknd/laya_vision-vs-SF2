"""Frozen vision-feature cache and without-replacement sampling. The model-equivalence test needs a checkpoint
and runs only with SF2_SLOW=1; the rest need just the processor."""
import os
import pickle
import random

import numpy as np
import pytest

from sf2 import vision_cache as vc

CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "chunli_r3", "best")


def test_epoch_order_shards_one_pass_across_workers_without_overlap():
    shards = [vc.epoch_order(103, seed=7, pass_idx=0, worker=w, n_workers=4) for w in range(4)]
    flat = [i for s in shards for i in s]
    assert sorted(flat) == list(range(103))  # every example exactly once per pass


def test_epoch_order_is_deterministic_and_reshuffles_each_pass():
    a = vc.epoch_order(50, seed=1, pass_idx=0, worker=0, n_workers=1)
    assert a == vc.epoch_order(50, seed=1, pass_idx=0, worker=0, n_workers=1)
    assert a != vc.epoch_order(50, seed=1, pass_idx=1, worker=0, n_workers=1)


def _fake_encode(paths):
    # feature rows that identify their image: every value = the image's number
    return np.stack([np.full((4, 3), float(os.path.basename(p).split(".")[0]), np.float16) for p in paths])


def _images(tmp_path, n):
    from PIL import Image

    d = tmp_path / "ds" / "images"
    d.mkdir(parents=True)
    paths = []
    for i in range(n):
        p = d / ("%d.png" % i)
        Image.new("RGB", (256, 224), (i, 0, 0)).save(p)
        paths.append(str(p))
    return paths


def _examples(paths, dataset="g"):
    return [{"id": "e%d" % i, "dataset": dataset, "label": 0, "target": [0.7, 0.3],
             "state": {"images": [paths[i], paths[i + 1]], "context": "me=chunli hp=%d" % i},
             "q": {"t": "choice", "ins": "Which move?", "crit": {"left": None, "right": None}}}
            for i in range(len(paths) - 1)]


def test_cache_builds_once_and_serves_rows_by_image(tmp_path):
    paths = _images(tmp_path, 5)
    ds = str(tmp_path / "ds")
    calls = []
    enc = lambda ps: (calls.append(len(ps)), _fake_encode(ps))[1]  # noqa: E731
    vc.build(ds, _examples(paths), enc, "fp1", batch=2)
    assert sum(calls) == 5  # each distinct image encoded once, though 4 of 5 appear in two rows
    vc.build(ds, _examples(paths), enc, "fp1", batch=2)
    assert sum(calls) == 5  # already complete: nothing recomputed
    cache = vc.load([ds], "fp1")
    got = cache.get([paths[3], paths[1]])
    assert got.shape == (2, 4, 3) and float(got[0, 0, 0]) == 3.0 and float(got[1, 0, 0]) == 1.0


def test_cache_for_another_fingerprint_is_missing(tmp_path):
    paths = _images(tmp_path, 3)
    ds = str(tmp_path / "ds")
    vc.build(ds, _examples(paths), _fake_encode, "fp1")
    assert vc.missing([ds], "fp1") == [] and vc.missing([ds], "fp2") == [ds]


def test_cache_still_serves_rows_after_the_data_dir_moves(tmp_path):
    # a data dir copied to another machine (or folder) must keep its cache: rows are keyed by image path
    import shutil

    paths = _images(tmp_path, 4)
    vc.build(str(tmp_path / "ds"), _examples(paths), _fake_encode, "fp1")
    shutil.move(str(tmp_path / "ds"), str(tmp_path / "moved"))
    moved = str(tmp_path / "moved")
    assert vc.missing([moved], "fp1") == []
    got = vc.load([moved], "fp1").get([os.path.join(moved, "images", "2.png")])
    assert float(got[0, 0, 0]) == 2.0


def test_old_cache_keyed_by_another_machines_paths_is_rebuilt(tmp_path):
    import json

    paths = _images(tmp_path, 3)
    ds = str(tmp_path / "ds")
    vc.build(ds, _examples(paths), _fake_encode, "fp1")
    idx = os.path.join(ds, "vision", "fp1.json")
    with open(idx, "w") as f:  # the pre-fix format: absolute paths from the machine that built it
        json.dump({"fingerprint": "fp1", "images": ["/Users/claw/x/ds/images/%d.png" % i for i in range(3)]}, f)
    assert vc.missing([ds], "fp1") == [ds]
    vc.build(ds, _examples(paths), _fake_encode, "fp1")
    assert float(vc.load([ds], "fp1").get([paths[1]])[0, 0, 0]) == 1.0


def test_cache_pickles_small_for_loader_workers(tmp_path):
    paths = _images(tmp_path, 40)
    ds = str(tmp_path / "ds")
    vc.build(ds, _examples(paths), lambda ps: np.zeros((len(ps), 64, 576), np.float16), "fp1")
    cache = vc.load([ds], "fp1")
    cache.get([paths[0]])  # opens the memmap in this process
    blob = pickle.dumps(cache)
    assert len(blob) < 64_000  # the index travels, the 40 x 72 KB of features do not
    assert pickle.loads(blob).get([paths[5]]).shape == (1, 64, 576)


@pytest.fixture(scope="module")
def processor():
    if not os.path.isdir(os.path.join(CKPT, "processor")):
        pytest.skip("no local checkpoint processor")
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(os.path.join(CKPT, "processor"))


def test_cached_item_has_the_same_tokens_as_the_pixel_path(tmp_path, processor):
    import laya.vlm_train as vt

    paths = _images(tmp_path, 3)
    ex = _examples(paths)[0]
    vc.build(str(tmp_path / "ds"), [ex], lambda ps: np.zeros((len(ps), 64, 576), np.float16), "fp1")
    cache = vc.load([str(tmp_path / "ds")], "fp1")
    ref = vt.make_item(processor, ex, random.Random(3))
    it = vc.cached_item(processor, ex, random.Random(3), cache, image_seq_len=64)
    for k in ("ids", "markers", "option_span", "target", "label", "order", "qtype", "n_images"):
        assert it[k] == ref[k], k
    assert tuple(it["feats"].shape) == (2, 64, 576)


def test_collate_stacks_features_in_row_then_image_order(tmp_path, processor):
    paths = _images(tmp_path, 4)
    exs = _examples(paths)
    vc.build(str(tmp_path / "ds"), exs, _fake_encode, "fp1")
    cache = vc.load([str(tmp_path / "ds")], "fp1")
    items = [vc.cached_item(processor, e, random.Random(0), cache, image_seq_len=4) for e in exs[:2]]
    for it in items:
        it["dataset"] = "g"
    b = vc.collate_train(items, processor.tokenizer.pad_token_id)
    assert b["pixel_values"] is None
    assert [float(x) for x in b["image_hidden_states"][:, 0, 0]] == [0.0, 1.0, 1.0, 2.0]


def test_stream_draws_each_example_once_per_pass(tmp_path, processor):
    paths = _images(tmp_path, 31)
    exs = _examples(paths)
    vc.build(str(tmp_path / "ds"), exs, _fake_encode, "fp1")
    cache = vc.load([str(tmp_path / "ds")], "fp1")
    stream = vc.CachedItemStream(processor, exs, seed=0, cache=cache, image_seq_len=4)
    it = iter(stream)
    first_pass = [next(it)["ids"] for _ in range(len(exs))]
    assert len({tuple(i) for i in first_pass}) == len(exs)  # no repeats before every example was drawn


@pytest.mark.skipif(not os.environ.get("SF2_SLOW"), reason="loads the model; set SF2_SLOW=1")
def test_cached_features_give_the_pixel_path_logits(tmp_path):
    import torch

    import laya
    import laya.vlm_train as vt

    agent = laya.load_vlm(CKPT, device="cpu")
    model, proc = agent.model.eval(), agent.processor
    data = os.path.join(os.path.dirname(CKPT), "..", "..", "data", "seed_chunli_r5")
    exs = vt.load_jsonl_examples(os.path.dirname(data), "seed_chunli_r5", "train")[:3]
    fp = vc.fingerprint(model, agent.prep)
    vc.build(str(tmp_path), exs, vc.encoder(model, proc), fp)
    cache = vc.load([str(tmp_path)], fp)
    ref = [vt.make_item(proc, e, random.Random(0), shuffle=False) for e in exs]
    cached = [vc.cached_item(proc, e, random.Random(0), cache, agent.prep.image_seq_len, shuffle=False) for e in exs]
    for it in ref + cached:
        it["dataset"] = "g"
    pad = proc.tokenizer.pad_token_id
    with torch.no_grad():
        a, _ = vt._forward(model, vt._collate_train(ref, pad))
        b, _ = vc.forward(model, vc.collate_train(cached, pad))
    mask = vt._collate_train(ref, pad)["marker_mask"]
    assert float((a - b)[mask].abs().max()) < 5e-3  # fp16 storage of the features
