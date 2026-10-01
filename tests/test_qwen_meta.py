"""Every lesson-loop run records which Qwen served it (owner 2026-10-01 chose to switch servers without re-running the
locked arms: the mix must be visible in every run's records, never inferred)."""
import importlib.util
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
spec = importlib.util.spec_from_file_location("qwen_lessons", os.path.join(HERE, "..", "scripts", "qwen_lessons.py"))
ql = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ql)


def test_qwen_meta_names_server_and_model(monkeypatch):
    monkeypatch.setattr(ql, "QWEN_URL", "http://host:8080/v1/chat/completions")
    monkeypatch.setattr(ql, "QWEN_MODEL", "some-model.gguf")
    assert ql.qwen_meta() == {"qwen_url": "http://host:8080/v1/chat/completions", "qwen_model": "some-model.gguf"}


def test_qwen_meta_never_carries_the_key(monkeypatch):
    monkeypatch.setenv("SF2_QWEN_API_KEY", "secret-xyz")
    assert "secret-xyz" not in repr(ql.qwen_meta())
