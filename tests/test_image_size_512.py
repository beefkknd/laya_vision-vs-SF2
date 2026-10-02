"""scripts/train.py --image-size {256,512}: 256 (default) is today's exact identity; 512 must make laya's own image
prep an EXACT nearest-neighbour 2x upscale of our 256x256 frames (every pixel an identical 2x2 block, no blur), so
the encoder gets 64 tokens per frame instead of 16. Run through laya's real prep (ImagePrep.from_config, the object
VLMAgent builds from the same config keys and training calls pixel_values on), not a reimplementation."""
import importlib.util
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from laya.preprocess import ImagePrep, prefix_ids
from sf2.config import IMAGE_CFG, image_cfg

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
spec = importlib.util.spec_from_file_location("train_script_512", os.path.join(ROOT, "scripts", "train.py"))
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)

DATA = os.path.join(ROOT, "test_data_eye_q3v2b_act", "walk")


def real_frame() -> np.ndarray:
    with open(os.path.join(DATA, "test.jsonl")) as f:
        row = json.loads(f.readline())
    with Image.open(os.path.join(DATA, row["images"][-1])) as im:
        img = np.asarray(im.convert("RGB"))
    assert img.shape == (256, 256, 3) and img.dtype == np.uint8
    assert len(np.unique(img.reshape(-1, 3), axis=0)) > 8   # a real screen, not a blank pad
    return img


def laya_pixels(cfg, img):
    """laya's own prep as an agent built with these overrides runs it (VLMAgent: ImagePrep.from_config(cfg))."""
    prep = ImagePrep.from_config(dict(cfg), default_backend="gpu")
    pv, mask = prep.pixel_values(img)
    assert bool(mask.all())
    return prep, pv


def normalised(img_hwc: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(np.ascontiguousarray(img_hwc)).permute(2, 0, 1)
    return (x.to(torch.float32) - 127.5) / 127.5


def test_512_is_exact_nearest_2x_upscale():
    img = real_frame()
    prep, pv = laya_pixels(image_cfg(512), img)
    assert prep.image_size == 512 and tuple(pv.shape) == (3, 512, 512)
    want = np.repeat(np.repeat(img, 2, 0), 2, 1)
    assert torch.equal(pv, normalised(want))
    back = (pv * 127.5 + 127.5).round().to(torch.uint8).permute(1, 2, 0).numpy()   # the uint8 before normalising
    assert np.array_equal(back, want)


def test_256_is_still_the_identity():
    img = real_frame()
    assert IMAGE_CFG == image_cfg(256) and image_cfg() == IMAGE_CFG
    _, pv = laya_pixels(IMAGE_CFG, img)
    assert torch.equal(pv, normalised(img))


def test_image_cfg_returns_fresh_dicts_and_rejects_other_sizes():
    a = image_cfg(512)
    a["image_size"] = 1
    assert image_cfg(512)["image_size"] == 512 and IMAGE_CFG["image_size"] == 256
    for bad in (128, 384, 1024, "512"):
        with pytest.raises(ValueError):
            image_cfg(bad)


def fake_agent(cfg):
    """What train.py's prep check reads off a loaded agent, built by laya's own ImagePrep.apply."""
    processor = SimpleNamespace(image_processor=SimpleNamespace(max_image_size={}, size={}))
    prep = ImagePrep.from_config(dict(cfg), default_backend="gpu").apply(processor)
    return SimpleNamespace(model=SimpleNamespace(prep=prep), processor=processor)


def test_train_default_stays_256():
    args = tr.parse_args(["--data", "x", "--out", "y"])
    assert args.image_size == 256
    assert tr.init_source(args) == (tr.BASE_MODEL, IMAGE_CFG)


def test_train_512_overrides_and_prep_check():
    args = tr.parse_args(["--data", "x", "--out", "y", "--image-size", "512"])
    assert args.image_size == 512
    source, overrides = tr.init_source(args)
    assert source == tr.BASE_MODEL and overrides == image_cfg(512)
    assert tr.prep_problem(fake_agent(overrides), image_cfg(512)) is None
    assert tr.prep_problem(fake_agent(IMAGE_CFG), image_cfg(512)) is not None     # a 256 agent fails a 512 run
    assert tr.prep_problem(fake_agent(overrides), IMAGE_CFG) is not None
    assert tr.prep_problem(fake_agent(dict(overrides, image_interpolation="bilinear")), image_cfg(512)) is not None


def test_train_init_with_512_keeps_checkpoint_prep():
    args = tr.parse_args(["--data", "x", "--out", "y", "--image-size", "512", "--init", "ck"])
    assert tr.init_source(args) == ("ck", {})


@pytest.mark.parametrize("bad", ["384", "1024", "abc"])
def test_train_rejects_other_sizes(bad, capsys):
    with pytest.raises(SystemExit):
        tr.parse_args(["--data", "x", "--out", "y", "--image-size", bad])
    assert "--image-size" in capsys.readouterr().err   # refused for its value, not as an unknown option
    if bad.isdigit():
        with pytest.raises(SystemExit):
            tr.parse_args(["--data", "x", "--out", "y", "--image-size", bad])
        assert "invalid choice" in capsys.readouterr().err


def test_tokens_per_frame_64_at_512_vs_16_at_256():
    p512 = ImagePrep.from_config(image_cfg(512), default_backend="gpu")
    p256 = ImagePrep.from_config(IMAGE_CFG, default_backend="gpu")
    assert (p512.image_seq_len, p256.image_seq_len) == (64, 16)


def test_processor_writes_64_image_tokens_per_frame():
    """The real processor (base checkpoint's, from the HF cache) after laya's apply: <image> tokens per frame."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor
    try:
        d = snapshot_download(tr.BASE_MODEL)
    except Exception as e:  # noqa: BLE001
        pytest.skip("base checkpoint not in the HF cache: %s" % e)
    counts = {}
    for size in (256, 512):
        processor = AutoProcessor.from_pretrained(os.path.join(d, "processor"))
        prep = ImagePrep.from_config(image_cfg(size), default_backend="gpu").apply(processor)
        prep.check(processor)
        image_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        ids = prefix_ids(processor, "User:", 2)
        counts[size] = ids.count(image_id) // 2
    assert counts == {256: 16, 512: 64}
