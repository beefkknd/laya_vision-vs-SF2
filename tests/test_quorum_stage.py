"""Stage (train vs eval) as a first-class invariant of the bee quorum (sf2/quorum/config.py, frontier.py; the driver's
resolve_stage). The bug this pins: an "eval" without a hand-built bees-off JSON silently ran all six explore bees, so
a trained table measured 39% when it was 94%. TRAIN behaviour is pinned byte-for-byte by the golden below."""
import contextlib
import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from looptools import FollowerLaya, load_driver, make_moment            # noqa: E402
from test_quorum import FakeEmu, Prefers, acc, base, fake_play, fake_score   # noqa: E402

from sf2.quorum.config import EXPLORE_BEES, QuorumConfig                   # noqa: E402
from sf2.quorum.decider import quorum_decider                              # noqa: E402
from sf2.system1 import value_table as VT                                  # noqa: E402

# four contexts so every explore bee votes somewhere: attacking|fb -> frontier+fireball+pressure; stunned -> punish;
# crouching -> vs_crouch; jumping|fb -> fireball+antiair
CONTEXTS = {"attacking_fb": dict(dx=36, doing="attacking", fireball=True), "stunned": dict(dx=36, doing="stunned"),
            "crouching": dict(dx=36, doing="crouching"), "jumping_fb": dict(dx=36, doing="jumping", fireball=True)}
GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "quorum_train_golden.json")


def _decide(cfg, ctx):
    """One deterministic decision in context ``ctx``: a 3-move cell + voter weights keyed on that context's 'when'."""
    mk = lambda t, rel: quorum_decider(t, rel, "chunli", random.Random(0), base(), Prefers("throw"),
                                       Prefers("throw_F+hp", "block_low", "walk_forward"), cfg, None)
    m = make_moment(**ctx)
    when = mk(VT.blank(), {"rel": {}})(m)["when"]
    t = VT.blank()
    t["cells"][when] = {"block_high": acc(5, 0.0), "cl.lk": acc(6, 4.0), "throw_F+mp": acc(2, 9.0)}
    d = mk(t, {"rel": {when: {"frontier": [1.5, 3], "table": [2.0, 4]}}})(m)
    return {"action": d["action"], "source": d["source"], "when": d["when"], "quorum": d["quorum"]}


# ------------------------------------------------------------------------------------------------ TRAIN golden
def test_train_decisions_match_the_golden():
    """Pinned against the PRE-refactor code (git tag stage-refactor-base): every proposal with voter / action /
    confidence / weight, the scores, share and pick, in all four contexts. Regenerate ONLY by explicit decision:
    dump {ctx: _decide(cfg, CONTEXTS[ctx])} with sort_keys=True, indent=1 to GOLDEN_PATH."""
    cfg = QuorumConfig(mode="vote", epsilon=0.0, theta=0.5, k=8.0, table_min_n=3)
    got = {k: _decide(cfg, v) for k, v in CONTEXTS.items()}
    assert json.dumps(got, sort_keys=True) == json.dumps(json.load(open(GOLDEN_PATH)), sort_keys=True)
    assert {p[0] for g in got.values() for p in g["quorum"]["proposals"]} >= set(EXPLORE_BEES)   # all six bees voted


# ------------------------------------------------------------------------------------------- config invariant
def test_eval_config_fails_fast_and_for_eval_keeps_the_genome():
    with pytest.raises(ValueError):
        QuorumConfig(stage="eval", frontier=True)
    with pytest.raises(ValueError):
        QuorumConfig(stage="eval", frontier=False, fireball=False, pressure=False, punish=False, vs_crouch=False,
                     antiair=False, epsilon=0.1)
    with pytest.raises(ValueError):
        QuorumConfig(stage="play")
    genome = QuorumConfig.from_dict({"mode": "vote", "theta": 0.7, "beta": 1.3, "k": 12.0, "epsilon": 0.1,
                                     "priors": {"table": 2.5}})
    ev = QuorumConfig.for_eval(genome)
    assert ev.stage == "eval" and ev.epsilon == 0.0 and not ev.qwen and not any(getattr(ev, b) for b in EXPLORE_BEES)
    assert (ev.theta, ev.beta, ev.k, ev.mode, ev.priors) == (0.7, 1.3, 12.0, "vote", genome.priors)
    assert QuorumConfig.from_dict({"mode": "vote"}).stage == "train"        # a legacy JSON (no stage) is a train config
    voters = {p[0] for c in CONTEXTS.values() for p in _decide(ev, c)["quorum"]["proposals"]}
    assert voters and voters <= {"laya", "table"}


# ---------------------------------------------------------------------------------------- policy matrix (eval)
def _spied_loop(monkeypatch, driver, policy, stage, **kw):
    """Run the fake loop under ``stage``; return (verdict, decision records, spy counts, decider kwargs seen)."""
    calls = {"credit": 0, "rel": 0, "coach": 0}
    seen = {}
    monkeypatch.setattr(driver.VT, "credit", lambda *a, **k: calls.__setitem__("credit", calls["credit"] + 1) or a[0])
    monkeypatch.setattr(driver.QR, "credit", lambda *a, **k: calls.__setitem__("rel", calls["rel"] + 1) or a[0])
    stub = {"refused": [], "added": [], "removed": [], "promoted": [], "retired": []}
    idxs = seen.setdefault("idx", [])                      # the churn counter update() was handed, per Coach call
    monkeypatch.setattr(driver, "update", lambda *a, **k: calls.__setitem__("coach", calls["coach"] + 1)
                        or idxs.append(a[2]) or (a[0], dict(stub)))
    dec, hyb = driver.VT.decider, driver.VT.hybrid_decider
    monkeypatch.setattr(driver.VT, "decider", lambda *a, **k: seen.update(k) or dec(*a, **k))
    monkeypatch.setattr(driver.VT, "hybrid_decider", lambda *a, **k: seen.update(k) or hyb(*a, **k))
    out = kw.pop("out")
    laya = FollowerLaya()
    v = driver.run_loop("ryu", laya, laya, lambda messages, task: "", games=1, rounds=2, seed_lines=[], out=out,
                        play_round_fn=fake_play, state=b"x", state_id={"path": "p", "sha256": "0"}, emu=FakeEmu(),
                        score_fn=fake_score, seed_rng=1, policy=policy, stage=stage, **kw)
    recs = [json.loads(l) for l in open(os.path.join(out, "trace.jsonl")) if '"decision"' in l]
    return v, recs, calls, seen


@pytest.mark.parametrize("policy", ["quorum", "table", "hybrid"])
def test_eval_stage_makes_no_explore_proposal_and_learns_nothing(monkeypatch, tmp_path, policy):
    driver = load_driver()
    v, recs, calls, seen = _spied_loop(monkeypatch, driver, policy, "eval", out=str(tmp_path / policy),
                                       quorum=QuorumConfig.for_eval(QuorumConfig(mode="vote", epsilon=0.1)))
    assert recs and calls == {"credit": 0, "rel": 0, "coach": 0}
    assert v["stage"] == "eval" and v["learn"] is False and v["explore_bees"] == [] and v["explore_rate"] == 0.0
    if policy == "quorum":
        assert all({p[0] for p in r["quorum"]["proposals"]} <= {"laya", "table"} for r in recs)
    else:
        assert seen["eps0" if policy == "table" else "explore"] == 0.0      # the knob WAS passed, and it is 0
        assert all(r["source"] != "table-explore" for r in recs)


def test_train_stage_still_explores_and_learns_and_no_learn_only_stops_learning(monkeypatch, tmp_path):
    driver = load_driver()
    v, recs, calls, seen = _spied_loop(monkeypatch, driver, "quorum", "train", out=str(tmp_path / "t"),
                                       quorum=QuorumConfig(mode="vote", epsilon=0.0))
    assert calls["credit"] == 2 and calls["rel"] == 2 and calls["coach"] == 2 and seen["idx"] == [0, 1]   # idx feeds update()
    assert v["stage"] == "train" and v["learn"] is True and set(v["explore_bees"]) == set(EXPLORE_BEES)
    assert any("frontier" in {p[0] for p in r["quorum"]["proposals"]} for r in recs)
    v, recs, calls, _ = _spied_loop(monkeypatch, driver, "quorum", "train", out=str(tmp_path / "nl"), no_learn=True,
                                    quorum=QuorumConfig(mode="vote", epsilon=0.0))
    assert calls == {"credit": 0, "rel": 0, "coach": 0} and v["learn"] is False     # ...so idx is never consumed
    assert any("frontier" in {p[0] for p in r["quorum"]["proposals"]} for r in recs)   # but the bees still explore


def test_main_table_eval_forwards_the_one_resolved_quorum(monkeypatch, tmp_path):
    """main's --policy table branch must hand run_loop the resolved (eval) quorum config, else run_loop's own
    resolve_stage sees a default TRAIN config at eval and raises -- the real bug the blind review caught."""
    driver, got = load_driver(), {}
    import sf2.system1.screen_emu as SE
    monkeypatch.setattr(SE, "open_screen", lambda *a, **k: contextlib.nullcontext(FakeEmu()))
    monkeypatch.setattr(driver, "run_loop", lambda *a, **k: got.update(k) or {"registry_end": []})
    monkeypatch.chdir(tmp_path)
    os.makedirs("states")
    open(os.path.join("states", "p1_chunli_vs_ryu.state"), "wb").write(b"x")
    monkeypatch.setattr(sys, "argv", ["play", "--policy", "table", "--stage", "eval", "--opp", "ryu", "--no-score",
                                      "--out", str(tmp_path / "o")])
    assert driver.main() == 0 and got["policy"] == "table"
    _, rt = driver.resolve_stage(got["stage"], got.get("quorum"), got.get("explore_rate"), got.get("no_learn"))  # as run_loop
    assert rt["stage"] == "eval" and rt["eps0"] == 0.0 and rt["learn"] is False and rt["explore_bees"] == []


@pytest.mark.parametrize("knob", [{"epsilon": 0.1}, {"qwen": True}] + [{b: True} for b in EXPLORE_BEES])
def test_resolve_stage_raises_on_any_train_knob_at_eval(knob):
    driver, ev = load_driver(), QuorumConfig.for_eval(QuorumConfig())
    assert driver.resolve_stage("eval", ev)[0] is ev                        # a clean eval config passes through as-is
    for k, v in knob.items():
        setattr(ev, k, v)                                                   # a post-hoc mutation: re-validated, raises
    with pytest.raises(ValueError):
        driver.resolve_stage("eval", ev)


def test_resolve_stage_has_one_authority_and_never_sanitizes():
    driver = load_driver()
    with pytest.raises(ValueError):
        driver.resolve_stage("eval", QuorumConfig(epsilon=0.1))             # a TRAIN config at eval: raised, not zeroed
    with pytest.raises(ValueError):
        driver.resolve_stage("eval", QuorumConfig.for_eval(QuorumConfig()), explore_rate=0.2)
    with pytest.raises(ValueError):
        driver.resolve_stage("train", QuorumConfig.for_eval(QuorumConfig()))   # an EVAL config at train: conflict
    with pytest.raises(ValueError):
        driver.resolve_stage("play", QuorumConfig())
    qcfg, rt = driver.resolve_stage("train", None, None, no_learn=True)   # --no-learn keeps its meaning: train, no credit
    assert rt["learn"] is False and qcfg.stage == "train" and rt["explore_rate"] == driver.HYBRID_EXPLORE
