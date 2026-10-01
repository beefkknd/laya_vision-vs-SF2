"""--eye through the runners (docs/prereg_u_perception.md "In play"): qwen_lessons.py and ab_memory.py play System 1
with a perception checkpoint (the U arm); run.json and the verdict name it and its sha256, run names end "+eye", a
checkpoint that is not tagged perception is refused as --eye and a tagged one is refused as --model; factorial_report
pairs U runs with the A and T runs of the same (opponent, seed) for a third ranking (U0-T0, U0-A0, U1-T1, U1-A1),
while its 2x2 report stays as it was."""
import contextlib
import json
import os
from types import SimpleNamespace

import pytest

from sf2.system1 import eye as E
from tests.test_ab_fixed import ab
from tests.test_factorial_report import LABEL, a_run, fr, rounds, world
from tests.test_lock import qwen_lessons
from tests.test_table_runs import loop_args


def ckpt(tmp_path, name="u"):
    d = tmp_path / name
    os.makedirs(d)
    (d / "model.safetensors").write_bytes(b"weights-" + name.encode())
    (d / "vlm_agent_config.json").write_text(json.dumps({"perception": True, "note_version": 3}))
    return str(d)


# ---- the checkpoint's identity -------------------------------------------------------------------------------------

def test_checkpoint_sha_covers_weights_and_config(tmp_path):
    a = ckpt(tmp_path)
    h = E.checkpoint_sha256(a)
    assert len(h) == 64 and h == E.checkpoint_sha256(a)
    with open(os.path.join(a, "vlm_agent_config.json"), "a") as f:
        f.write(" ")
    assert E.checkpoint_sha256(a) != h
    with pytest.raises(SystemExit):
        E.checkpoint_sha256(str(tmp_path / "missing"))


# ---- qwen_lessons ----------------------------------------------------------------------------------------------------

def test_without_eye_nothing_changes(tmp_path):
    q = qwen_lessons()
    args = loop_args(tmp_path)
    assert q.eye_meta(args) == {} and q.run_name(args, "S") == "S_ryu_character_fgc"
    assert all("--eye" not in c for _, c in q.arm_cmds(args, "ROOT"))


def test_eye_is_recorded_suffixed_and_passed_to_the_arms(tmp_path):
    q = qwen_lessons()
    u = ckpt(tmp_path)
    args = loop_args(tmp_path, eye=u, book="lessons/book.json")
    assert q.run_name(args, "S") == "S_ryu_character_fgc_book+eye"
    assert q.eye_meta(args) == {"eye": u, "eye_sha256": E.checkpoint_sha256(u)}
    for _, c in q.arm_cmds(args, "ROOT"):
        assert c[c.index("--eye") + 1] == u


def play(q, monkeypatch, args, out, has_eye):
    made = []

    class S1:
        def __init__(self, model, me, **kw):
            made.append((model, kw.get("oracle")))
            self.short, self.eye = None, (object() if has_eye else None)

    monkeypatch.setattr(q, "Advisor", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(q, "open_fight", lambda *a, **k: contextlib.nullcontext((None, None)))
    monkeypatch.setattr(q, "System1", S1)
    monkeypatch.setattr(q, "play_round", lambda *a: SimpleNamespace(log=[], result="win",
                                                                    summary={"result": "win", "dealt": 1, "taken": 0}))
    monkeypatch.setattr(q, "chat", lambda *a: (_ for _ in ()).throw(RuntimeError("qwen down")))
    return q.play_arm(args, "none", 0, out), made


def test_an_eye_arm_plays_the_checkpoint_and_records_it(tmp_path, monkeypatch):
    q = qwen_lessons()
    u = ckpt(tmp_path)
    out = str(tmp_path / "none")
    code, made = play(q, monkeypatch, loop_args(tmp_path, eye=u), out, True)
    assert code == 0 and made == [(u, None)]
    run = json.load(open(os.path.join(out, "run.json")))
    assert run["eye"] == u and run["eye_sha256"] == E.checkpoint_sha256(u) and run["model"] == u


def test_eye_refuses_an_untagged_checkpoint(tmp_path, monkeypatch):
    q = qwen_lessons()
    with pytest.raises(SystemExit):
        play(q, monkeypatch, loop_args(tmp_path, eye=ckpt(tmp_path)), str(tmp_path / "none"), False)


def test_a_perception_checkpoint_as_model_is_refused(tmp_path, monkeypatch):
    q = qwen_lessons()
    with pytest.raises(SystemExit):
        play(q, monkeypatch, loop_args(tmp_path, model=ckpt(tmp_path)), str(tmp_path / "none"), True)


def test_eye_with_a_table_is_refused(tmp_path, monkeypatch):
    q = qwen_lessons()
    with pytest.raises(SystemExit):
        play(q, monkeypatch, loop_args(tmp_path, eye=ckpt(tmp_path), oracle="x.json"), str(tmp_path / "none"), True)


# ---- ab_memory ---------------------------------------------------------------------------------------------------------

def test_ab_memory_eye_run_folder_meta_and_arm(tmp_path, monkeypatch):
    m = ab()
    u = ckpt(tmp_path)
    assert m.run_root(5, str(tmp_path), "S", eye=True).endswith("S_s5+eye")
    assert m.run_meta(SimpleNamespace(model="M", oracle=None, eye=u)) == {"model": u, "eye": u,
                                                                         "eye_sha256": E.checkpoint_sha256(u)}
    assert m.run_meta(SimpleNamespace(model="M", oracle=None)) == {"model": "M"}
    made = []

    class S1:
        def __init__(self, model, me, **kw):
            made.append(model)
            self.eye = object()

    monkeypatch.setattr(m, "Advisor", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(m, "open_fight", lambda *a, **k: contextlib.nullcontext((None, None)))
    monkeypatch.setattr(m, "System1", S1)
    args = SimpleNamespace(char="chunli", advisor="a", model="runs/all8/best", seed=1, rounds=0, oracle=None, eye=u)
    assert m.play_arm(args, "ryu", "none", 0, str(tmp_path / "ryu_none")) == 0 and made == [u]


# ---- factorial_report: the third ranking --------------------------------------------------------------------------

def u_world(root, opps=("ryu", "ken"), seeds=(1, 2, 3), n=10, sha="u1", commit="c2"):
    """A0 = 0, A1 = +10, T0 = +30, T1 = +35 (world) and U0 = +20, U1 = +32 hp per round."""
    world(root, opps, seeds, n)
    for opp in opps:
        for s in seeds:
            a_run(root, "2026100%d-000002_%s_character_fgc_book+eye" % (s, opp), s, rounds([32] * n),
                  rounds([20] * n), False, extra={"eye": "runs/u/best", "eye_sha256": sha, "commit": commit})


def test_the_2x2_report_ignores_eye_runs(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    world(a)
    u_world(b)
    ra, rb = fr().report([a], LABEL, min_rounds=10), fr().report([b], LABEL, min_rounds=10)
    assert rb["problems"] == [] and [p[:2] for p in ra["pairs"]] == [p[:2] for p in rb["pairs"]]
    assert ra["effects"] == rb["effects"] and ra["cells"] == rb["cells"]


def test_u_report_pairs_three_rankings_and_its_effects(tmp_path):
    r = str(tmp_path)
    u_world(r)
    rep = fr().u_report([r], LABEL, min_rounds=10)
    assert rep["problems"] == [] and len(rep["triples"]) == 6
    eff = {k: v["pooled"]["mean"] for k, v in rep["effects"].items()}
    assert eff == pytest.approx({"U0-T0": -10, "U0-A0": 20, "U1-T1": -3, "U1-A1": 22})
    assert rep["cells"]["U0"]["hp"] == pytest.approx(20) and rep["cells"]["T1"]["hp"] == pytest.approx(35)
    assert rep["commits"] == {"eye": ["c2"]}


def test_u_report_refuses_two_eye_checkpoints(tmp_path):
    r = str(tmp_path)
    u_world(r, seeds=(1,))
    a_run(r, "20261002-000002_ryu_character_fgc_book+eye", 2, rounds([1] * 10), rounds([1] * 10), False,
          extra={"eye": "runs/u2/best", "eye_sha256": "u2"})
    world(r, opps=("ryu",), seeds=(2,))
    rep = fr().u_report([r], LABEL, min_rounds=10)
    assert any("eye_sha256" in p for p in rep["problems"])


def test_u_report_refuses_a_triple_whose_settings_differ(tmp_path):
    r = str(tmp_path)
    fl = {"forward_lessons": False}
    a_run(r, "20261001-000000_ryu_character_fgc_book", 1, rounds([1] * 10), rounds([1] * 10), False, extra=fl)
    a_run(r, "20261001-000001_ryu_character_fgc_book+table", 1, rounds([1] * 10), rounds([1] * 10), True, extra=fl)
    a_run(r, "20261001-000002_ryu_character_fgc_book+eye", 1, rounds([1] * 10), rounds([1] * 10), False,
          extra={"eye": "runs/u/best", "eye_sha256": "u1", "forward_lessons": True})
    rep = fr().u_report([r], LABEL, min_rounds=10)
    assert rep["triples"] == [] and any("forward_lessons" in p for p in rep["problems"])


def test_main_prints_the_u_report_only_when_asked(tmp_path, capsys):
    r = str(tmp_path)
    u_world(r, seeds=(1,))
    fr().main(["--min-rounds", "10", r])
    assert "U0-T0" not in capsys.readouterr().out
    fr().main(["--min-rounds", "10", "--u", r])
    assert "U0-T0" in capsys.readouterr().out


def test_u_runs_on_newer_code_still_pair_and_list_their_commits(tmp_path):
    r = str(tmp_path)
    old = {"commit": "c1"}
    a_run(r, "20261001-000000_ryu_character_fgc_book", 1, rounds([0] * 10), rounds([0] * 10), False, extra=old)
    a_run(r, "20261001-000001_ryu_character_fgc_book+table", 1, rounds([5] * 10), rounds([5] * 10), True, extra=old)
    a_run(r, "20261001-000002_ryu_character_fgc_book+eye", 1, rounds([3] * 10), rounds([3] * 10), False,
          extra={"eye": "runs/u/best", "eye_sha256": "u1", "commit": "c2"})
    rep = fr().u_report([r], LABEL, min_rounds=10)
    assert rep["problems"] == [] and len(rep["triples"]) == 1
    assert rep["commits"] == {"all8": ["c1"], "table": ["c1"], "eye": ["c2"]}
