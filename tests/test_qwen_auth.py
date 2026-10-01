"""Qwen's server may need an API key (owner 2026-10-01: a faster server on the LAN). The key is read at request time from
$SF2_QWEN_API_KEY or a private file (config.QWEN_KEY_FILE), sent as a Bearer header, and never logged."""
import json
import os

import pytest

from sf2.system2 import qwen


class FakeResp:
    def __init__(self, body):
        self.body = body

    def read(self):
        return json.dumps(self.body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def run_chat(monkeypatch, tmp_path, env_key=None, file_key=None):
    seen = {}

    def fake_urlopen(req, timeout):
        seen["headers"] = dict(req.header_items())
        return FakeResp({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})

    monkeypatch.setattr(qwen.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(qwen, "LOG_DIR", str(tmp_path / "logs"))
    keyfile = tmp_path / "qwen_api_key"
    if file_key is not None:
        keyfile.write_text(file_key + "\n")
    monkeypatch.setattr(qwen, "QWEN_KEY_FILE", str(keyfile))
    if env_key is None:
        monkeypatch.delenv("SF2_QWEN_API_KEY", raising=False)
    else:
        monkeypatch.setenv("SF2_QWEN_API_KEY", env_key)
    assert qwen.chat([{"role": "user", "content": "hi"}], "t") == "ok"
    logged = "".join(p.read_text() for p in (tmp_path / "logs").iterdir())
    return seen["headers"], logged


def test_no_key_no_header(monkeypatch, tmp_path):
    h, _ = run_chat(monkeypatch, tmp_path)
    assert "Authorization" not in h


def test_key_file_sent_as_bearer_and_never_logged(monkeypatch, tmp_path):
    h, logged = run_chat(monkeypatch, tmp_path, file_key="secret-from-file")
    assert h["Authorization"] == "Bearer secret-from-file"
    assert "secret-from-file" not in logged


def test_env_wins_over_file(monkeypatch, tmp_path):
    h, logged = run_chat(monkeypatch, tmp_path, env_key="secret-env", file_key="secret-file")
    assert h["Authorization"] == "Bearer secret-env" and "secret-env" not in logged
