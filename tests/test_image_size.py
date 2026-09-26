"""--image-size reaches load_vlm in train.py and cache_vision.py, and a checkpoint keeps it. No model needed."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import cache_vision  # noqa: E402
import train  # noqa: E402

laya = pytest.importorskip("laya")


class _Loaded(Exception):
    pass


def _load_kwargs(monkeypatch, script, argv):
    seen = {}

    def fake_load_vlm(path, **kw):
        seen.update(kw, path=path)
        raise _Loaded()

    monkeypatch.setattr(laya, "load_vlm", fake_load_vlm)
    monkeypatch.setattr(sys, "argv", [script.__file__] + argv)
    with pytest.raises(_Loaded):
        script.main()
    return seen


@pytest.mark.parametrize("script,argv", [(train, ["--data", "d", "--out", "o"]),
                                         (cache_vision, ["--init", "ckpt", "--data", "d"])])
def test_image_size_is_passed_to_load_vlm(monkeypatch, script, argv):
    assert _load_kwargs(monkeypatch, script, argv + ["--image-size", "256"])["image_size"] == 256


@pytest.mark.parametrize("script,argv", [(train, ["--data", "d", "--out", "o"]),
                                         (cache_vision, ["--init", "ckpt", "--data", "d"])])
def test_without_the_flag_the_checkpoint_keeps_its_own_size(monkeypatch, script, argv):
    assert "image_size" not in _load_kwargs(monkeypatch, script, argv)


def test_a_saved_checkpoint_records_the_overridden_size():
    # VLMAgent._load applies overrides to the saved config before building ImagePrep; save() writes prep.to_config()
    from laya.preprocess import ImagePrep

    saved = {"image_size": 512, "preprocess": "processor"}
    prep = ImagePrep.from_config(dict(saved, image_size=256))
    assert prep.to_config()["image_size"] == 256
    assert prep.image_seq_len == 16  # 64 image tokens per frame at 512
    assert ImagePrep.from_config(prep.to_config()).image_size == 256
