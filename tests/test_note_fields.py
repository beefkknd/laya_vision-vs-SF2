"""A model trained without some note fields (train.py --drop-note-field) is played without them too."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import train  # noqa: E402
from sf2 import policy  # noqa: E402

laya = pytest.importorskip("laya")
NOTE = ("me=chunli stand hp=100 opp=dhalsim attack hp=90 dist=mid facing=right corner=none time=early "
        "last=hk fireball=none")


def test_drop_fields_removes_only_the_named_field():
    out = policy.drop_fields(NOTE, ["last"])
    assert "last=" not in out
    assert out == NOTE.replace(" last=hk", "")
    assert policy.drop_fields(NOTE, []) == NOTE


def test_training_examples_lose_the_field():
    exs = [{"state": {"images": ["a", "b"], "context": NOTE}}]
    train.drop_note_fields(exs, ["last"])
    assert "last=" not in exs[0]["state"]["context"]


class _Agent:
    def __init__(self):
        self.seen = []

    def predict(self, state, q):
        self.seen.append(state["context"])
        return {"answers": {"action": {"choice": "idle", "probabilities": {"idle": 1.0}}}}


def _policy(monkeypatch, path):
    agent = _Agent()
    monkeypatch.setattr(laya, "load_vlm", lambda p, **kw: agent)
    pol = policy.LayaPolicy(str(path))
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    pol.act(img, img, NOTE)
    return agent.seen[0]


def test_a_checkpoint_saved_without_last_is_played_without_it(monkeypatch, tmp_path):
    train.write_note(str(tmp_path), ["last"])
    assert "last=" not in _policy(monkeypatch, tmp_path)


def test_an_ordinary_checkpoint_sees_the_whole_note(monkeypatch, tmp_path):
    assert _policy(monkeypatch, tmp_path) == NOTE
