"""The brain panel's buttons write only a --fresh run's memory (memory_runs/<name>), never memory/ (only Qwen writes
that), and only from the panel's own page."""
import importlib.util
import json
import os
import sys

import pytest

from sf2.demo import demo_cheat

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for rel in ("playbook/chunli.json", "short/chunli_vs_ryu.json"):
        path = tmp_path / demo_cheat.SEED / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"me": "chunli", "lessons": [{"text": "use more sweep at mid range"}]}))
    (tmp_path / "memory" / "short").mkdir(parents=True)
    (tmp_path / "memory_runs" / "video1").mkdir(parents=True)
    return tmp_path


def test_cheat_and_clean_write_the_run_memory(repo):
    demo_cheat.apply("memory_runs/video1", "chunli", "short", "cheat", "ryu")
    got = json.load(open(repo / "memory_runs/video1/short/chunli_vs_ryu.json"))
    assert got["lessons"][0]["text"] == "use more sweep at mid range"
    demo_cheat.apply("memory_runs/video1", "chunli", "playbook", "clean", None)
    assert json.load(open(repo / "memory_runs/video1/playbook/chunli.json"))["lessons"] == []
    assert not [p for p in os.listdir(repo / "memory_runs/video1/short") if p.endswith(".tmp")]


@pytest.mark.parametrize("root,what,do,opp", [
    ("memory_runs/../memory", "short", "clean", "ryu"),          # a root that walks out
    ("memory_runs", "short", "clean", "ryu"),                    # the runs folder itself
    ("memory_runs/video1", "short", "clean", "../../memory/short/chunli_vs_ryu"),
    ("memory_runs/video1", "short", "clean", "ryu/../../../memory/short/x"),
    ("memory_runs/video1", "short", "clean", "akuma"),           # not a character
    ("memory_runs/video1", "short", "clean", None),              # a short memory needs an opponent
    ("memory_runs/video1", "notes", "clean", "ryu"),
    ("memory_runs/video1", "short", "wipe", "ryu"),
])
def test_bad_requests_never_write(repo, root, what, do, opp):
    before = sorted(str(p) for p in repo.rglob("*"))
    with pytest.raises(ValueError):
        demo_cheat.apply(root, "chunli", what, do, opp)
    with pytest.raises(ValueError):
        demo_cheat.add_pending(what, do, opp) if root == "memory_runs/video1" else demo_cheat.apply(
            root, "chunli", what, do, opp)
    assert sorted(str(p) for p in repo.rglob("*")) == before


def test_a_symlinked_run_folder_cannot_reach_memory(repo):
    os.symlink(repo / "memory", repo / "memory_runs" / "sneaky")
    with pytest.raises(ValueError):
        demo_cheat.apply("memory_runs/sneaky", "chunli", "short", "clean", "ryu")


def panel():
    if os.path.join(HERE, "scripts") not in sys.path:
        sys.path.insert(0, os.path.join(HERE, "scripts"))
    spec = importlib.util.spec_from_file_location("brain_panel", os.path.join(HERE, "scripts", "brain_panel.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("host,origin,ok", [
    ("127.0.0.1:8765", "http://127.0.0.1:8765", True),
    ("localhost:8765", "http://localhost:8765", True),
    ("127.0.0.1:8765", None, True),                              # curl / the page's own same-origin request
    ("127.0.0.1:8765", "https://evil.example", False),           # another site posting to the panel
    ("127.0.0.1:8765", "null", False),
    ("evil.example:8765", None, False),                          # DNS rebinding: a foreign Host
    ("127.0.0.1:8765", "http://127.0.0.1:9999", False),
])
def test_the_panel_takes_buttons_only_from_its_own_page(host, origin, ok):
    assert panel().request_ok(host, origin, 8765) is ok


def test_a_cheat_with_no_seed_file_is_refused_when_pressed(repo):         # found by the blind review, 2026-09-29
    with pytest.raises(ValueError, match="no seed"):
        demo_cheat.add_pending("short", "cheat", "ken")
    assert demo_cheat.pending() == {}


def test_a_bad_pending_button_never_stops_a_fresh_start(repo):
    demo_cheat.add_pending("short", "cheat", "ryu")
    os.remove(repo / demo_cheat.SEED / "short/chunli_vs_ryu.json")          # the seed went away meanwhile
    done = demo_cheat.apply_pending("memory_runs/video1", "chunli")
    assert len(done) == 1 and "could not" in done[0] and not os.path.exists(demo_cheat.PENDING)
    os.makedirs(os.path.dirname(demo_cheat.PENDING), exist_ok=True)
    open(demo_cheat.PENDING, "w").write("{half")
    assert demo_cheat.pending() == {}
    assert demo_cheat.apply_pending("memory_runs/video1", "chunli") == [] and not os.path.exists(demo_cheat.PENDING)


@pytest.mark.parametrize("body", [b"[]", b'"x"', b'{"what": "short", "do": "clean", "opp": ["x"]}',
                                  b'{"what": ["short"], "do": "clean", "opp": "ryu"}', b"{half", b"null"])
def test_odd_bodies_are_a_400_not_a_dropped_connection(body):
    with pytest.raises(ValueError):
        panel().button(body)


def test_a_good_body_reads():
    assert panel().button(b'{"what": "short", "do": "clean", "opp": "ryu"}') == ("short", "clean", "ryu")
