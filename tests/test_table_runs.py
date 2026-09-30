"""--oracle through the runners (docs/prereg_2x2.md): qwen_lessons.py and ab_memory.py play System 1 with the lookup
table and no laya-vision; the run file, run.json and the verdict name the table and its sha256; run names get "+table"
so the reports can tell the rankings apart; --shared-text-laya (qwen_lessons) uses the one shared text laya server and
reserves config.RUN_JOB_SHARED_GB per run. System 2's readers of the logs neither crash on table runs nor mix them
with runs/all8's."""
import contextlib
import hashlib
import json
import os
from types import SimpleNamespace

from sf2.config import MODEL_JOB_GB, RUN_JOB_SHARED_GB
from sf2.data import value_oracle
from sf2.system2 import track_record as T
from tests.test_ab_fixed import ab
from tests.test_lock import qwen_lessons

CELL = ("chunli", "close", 0, 0)


def table_file(tmp_path):
    p = str(tmp_path / "oracle.json")
    value_oracle.save({CELL: {"throw": 6.0, "lk": 2.0}}, p)
    return p, hashlib.sha256(open(p, "rb").read()).hexdigest()


def loop_args(tmp_path, **kw):
    base = dict(opp="ryu", seed=7, lock=None, prompt="character_fgc", track=None, book=None, forward_lessons=False,
                history=False, games=1, rounds=1, advisor="a", model="runs/all8/best", state=None, oracle=None,
                shared_text_laya=False, base_port=9000)
    return SimpleNamespace(**dict(base, **kw))


# ---- qwen_lessons ----

def test_without_oracle_nothing_is_recorded_or_suffixed(tmp_path):
    q = qwen_lessons()
    args = loop_args(tmp_path)
    assert q.oracle_meta(args) == {} and q.load_oracle(None) == (None, None)
    assert q.run_name(args, "S") == "S_ryu_character_fgc"
    assert q.job_gb(args) == MODEL_JOB_GB


def test_oracle_is_recorded_with_its_sha_and_suffixes_the_run(tmp_path):
    q = qwen_lessons()
    path, digest = table_file(tmp_path)
    args = loop_args(tmp_path, oracle=path, book="lessons/book.json")
    table, got = q.load_oracle(path)
    assert table == {CELL: {"throw": 6.0, "lk": 2.0}} and got == digest
    assert q.oracle_meta(args) == {"oracle": path, "oracle_sha256": digest}
    assert q.run_name(args, "S") == "S_ryu_character_fgc_book+table"


def test_shared_text_laya_reserves_the_shared_run_budget(tmp_path):
    q = qwen_lessons()
    assert q.job_gb(loop_args(tmp_path, shared_text_laya=True)) == RUN_JOB_SHARED_GB


def test_the_children_get_oracle_and_shared_flags(tmp_path):
    q = qwen_lessons()
    path, _ = table_file(tmp_path)
    cmds = q.arm_cmds(loop_args(tmp_path, oracle=path, shared_text_laya=True), "ROOT")
    assert [k for k, _ in cmds] == [("ryu", "loop"), ("ryu", "none")]
    for _, c in cmds:
        assert c[c.index("--oracle") + 1] == path and "--shared-text-laya" in c
    plain = q.arm_cmds(loop_args(tmp_path), "ROOT")
    assert all("--oracle" not in c and "--shared-text-laya" not in c for _, c in plain)


def play_loop_arm(q, monkeypatch, args, out):
    made, advisors = [], []

    class S1:
        def __init__(self, model, me, **kw):
            made.append((model, kw.get("oracle")))
            self.short = None

    def advisor(*a, **k):
        advisors.append(k)
        return contextlib.nullcontext()

    def play_round(*a):
        return SimpleNamespace(log=[], result="win", summary={"result": "win", "dealt": 10, "taken": 0})

    monkeypatch.setattr(q, "Advisor", advisor)
    monkeypatch.setattr(q, "open_fight", lambda *a, **k: contextlib.nullcontext((None, None)))
    monkeypatch.setattr(q, "System1", S1)
    monkeypatch.setattr(q, "play_round", play_round)
    monkeypatch.setattr(q, "chat", lambda *a: (_ for _ in ()).throw(RuntimeError("qwen down")))
    assert q.play_arm(args, "loop", 0, out) == 0
    return made, advisors


def test_a_table_arm_plays_without_laya_vision_and_records_the_table(tmp_path, monkeypatch):
    q = qwen_lessons()
    path, digest = table_file(tmp_path)
    out = str(tmp_path / "loop")
    made, advisors = play_loop_arm(q, monkeypatch, loop_args(tmp_path, oracle=path, shared_text_laya=True), out)
    assert made == [(None, {CELL: {"throw": 6.0, "lk": 2.0}})]
    assert advisors == [{"shared": True}]
    run = json.load(open(os.path.join(out, "run.json")))
    assert run["oracle"] == path and run["oracle_sha256"] == digest
    assert (run["games"], run["rounds"], run["advisor"], run["model"]) == (1, 1, "a", None)
    assert len(run["commit"]) == 40                  # the code the run played (factorial_report pairs only equals)


def test_an_all8_arm_plays_as_before(tmp_path, monkeypatch):
    q = qwen_lessons()
    out = str(tmp_path / "loop")
    made, advisors = play_loop_arm(q, monkeypatch, loop_args(tmp_path), out)
    assert made == [("runs/all8/best", None)] and advisors == [{}]
    run = json.load(open(os.path.join(out, "run.json")))
    assert "oracle" not in run and run["model"] == "runs/all8/best" and run["games"] == 1


# ---- ab_memory ----

def test_ab_memory_table_run_folder_is_suffixed(tmp_path):
    m = ab()
    assert m.run_root(5, str(tmp_path), "S", table=True).endswith("S_s5+table")
    assert m.run_root(5, str(tmp_path), "T").endswith("T_s5")


def test_ab_memory_table_arm_plays_without_laya_vision(tmp_path, monkeypatch):
    m = ab()
    path, digest = table_file(tmp_path)
    made = []

    class S1:
        def __init__(self, model, me, **kw):
            made.append((model, kw.get("oracle")))

    monkeypatch.setattr(m, "Advisor", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(m, "open_fight", lambda *a, **k: contextlib.nullcontext((None, None)))
    monkeypatch.setattr(m, "System1", S1)
    args = SimpleNamespace(char="chunli", advisor="a", model="runs/all8/best", seed=1, rounds=0, oracle=path)
    assert m.play_arm(args, "ryu", "none", 0, str(tmp_path / "ryu_none")) == 0
    assert made == [(None, {CELL: {"throw": 6.0, "lk": 2.0}})]
    assert m.run_meta(args) == {"model": None, "oracle": path, "oracle_sha256": digest}
    assert m.run_meta(SimpleNamespace(model="M", oracle=None)) == {"model": "M"}


# ---- System 2's readers ----

def a_run(root, name, hp, lines, verdict):
    d = os.path.join(root, name)
    rounds = {"loop": [{"dealt": max(h, 0), "taken": max(-h, 0), "lines": lines, "result": "win"} for h in hp],
              "none": [{"dealt": 0, "taken": 0, "lines": [], "result": "loss"} for _ in hp]}
    for arm, rs in rounds.items():
        os.makedirs(os.path.join(d, arm))
        with open(os.path.join(d, arm, "rounds.jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rs))
    with open(os.path.join(d, "verdict.json"), "w") as f:
        json.dump(verdict, f)


def test_the_track_record_keeps_table_runs_apart(tmp_path):
    """A lesson's rounds under the table ranking are not evidence about it under runs/all8 (and back)."""
    r = str(tmp_path)
    line = "use more hp up close"
    for i in range(3):
        a_run(r, "2026093%d-000000_ryu" % i, [10] * 10, [line], {"seed": i})
        a_run(r, "2026093%d-000001_ryu+table" % i, [-50] * 10, [line], {"seed": i, "oracle": "t.json"})
    all8 = T.build([r])
    assert all8["opponents"]["ryu"][line]["mean"] == 10 and len(all8["sources"]) == 3
    assert len(all8["other_ranking"]) == 3
    table = T.build([r], ranking="table")
    assert table["opponents"]["ryu"][line]["mean"] == -50 and len(table["sources"]) == 3


def test_a_track_record_without_table_runs_is_unchanged(tmp_path):
    r = str(tmp_path)
    for i in range(3):
        a_run(r, "2026093%d-000000_ryu" % i, [10] * 10, ["x"], {"seed": i})
    assert set(T.build([r])) == {"opponents", "sources", "skipped", "short"}


def test_compare_prompts_labels_table_runs(tmp_path):
    from tests.test_compare_prompts import cp
    r = str(tmp_path)
    a_run(r, "1_ryu_character_fgc_book", [10] * 10, [], {"seed": 3, "prompt": "character_fgc", "book": "b"})
    a_run(r, "2_ryu_character_fgc_book+table", [20] * 10, [], {"seed": 3, "prompt": "character_fgc", "book": "b",
                                                               "oracle": "t"})
    labels = sorted(x[3] for x in cp()._runs([r]))
    assert labels == ["character_fgc+book", "character_fgc+book+table"]
