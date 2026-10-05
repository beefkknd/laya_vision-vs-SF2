"""The bee-quorum System 1 (sf2/quorum): config, tally, reliability, flavour voters, the decider's three modes, the
Qwen hook, the genome, the driver switch (run_loop policy="quorum") and the evolution loop. Pure: fake advisors and
a fake play_round, no emulator, no models, no Qwen."""
import json
import os
import random
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
from looptools import FollowerLaya, load_driver, make_moment            # noqa: E402

from sf2.quorum import genome as G                                        # noqa: E402
from sf2.quorum import reliability as R                                    # noqa: E402
from sf2.quorum.config import QuorumConfig                                 # noqa: E402
from sf2.quorum.decider import quorum_decider                              # noqa: E402
from sf2.quorum.escalate import make_qwen_pick, parse                      # noqa: E402
from sf2.quorum.tally import Proposal, ranking, score, table_proposal      # noqa: E402
from sf2.quorum.voters import base_proposal, flavor_proposal               # noqa: E402
from sf2.system1 import value_table as VT                                  # noqa: E402
from sf2.system1.advice import char_categories                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CATS = char_categories("chunli")
WHEN = "close|standing|0"


class Prefers:
    """A fake advisor: probability 0.8 on the first preferred option it is offered, the rest shared."""

    def __init__(self, *prefs):
        self.prefs, self.calls = prefs, []

    def ask(self, text, q):
        opts = list(q["criteria"])
        self.calls.append(opts)
        pick = next((p for p in self.prefs if p in opts), opts[0])
        rest = (0.2 / (len(opts) - 1)) if len(opts) > 1 else 0.0
        return {o: (0.8 if o == pick else rest) for o in opts} if len(opts) > 1 else {pick: 1.0}


def base(action="block_high", cat="block"):
    return lambda m: {"action": action, "category": cat, "cat_probs": {cat: 0.9}, "move_probs": {action: 0.5},
                      "prompt_lines": []}


def cell_table(stats):
    t = VT.blank()
    t["cells"][WHEN] = {a: list(s) for a, s in stats.items()}
    return t


def acc(n, mean):
    return [n, n * mean, n * mean * mean]


# ---------------------------------------------------------------------------------------------------------- config
def test_config_validates_and_round_trips(tmp_path):
    with pytest.raises(ValueError):
        QuorumConfig(mode="dance")
    with pytest.raises(ValueError):
        QuorumConfig.from_dict({"thetaa": 0.5})
    c = QuorumConfig.from_dict({"mode": "vote", "priors": {"table": 2.5}})
    assert c.priors["table"] == 2.5 and c.priors["laya"] == 1.0
    p = str(tmp_path / "q.json")
    c.save(p)
    assert QuorumConfig.load(p) == c


# ----------------------------------------------------------------------------------------------------------- tally
def test_table_voter_abstains_without_evidence_or_on_losing_moves():
    cfg = QuorumConfig(table_min_n=3)
    assert table_proposal({"s.hp": acc(2, 9.0)}, ["s.hp"], cfg) is None            # too few samples
    assert table_proposal({"s.hp": acc(9, -4.0)}, ["s.hp"], cfg) is None           # never votes for a losing move
    p = table_proposal({"s.hp": acc(9, 4.0), "s.lk": acc(9, 6.0)}, ["s.hp", "s.lk"], cfg)
    assert p.action == "s.lk" and p.confidence == pytest.approx(9 / 17)


def test_score_weights_votes_recruits_and_inhibits():
    cfg = QuorumConfig(beta=1.0, gamma=1.0, k=1.0, net_scale=10.0)
    rel = R.blank()
    rel["rel"][WHEN] = {"attack": [3.0, 5]}                       # attack has earned a loud voice here
    props = [Proposal("laya", "block_high", 0.5), Proposal("attack", "cl.hp", 0.5)]
    cell = {"cl.hp": acc(9, 10.0), "block_high": acc(9, -10.0)}
    s = score(props, cell, WHEN, rel, cfg)
    assert s["cl.hp"]["votes"] == pytest.approx(1.5) and s["cl.hp"]["recruit"] > 0
    assert s["block_high"]["inhibit"] > 0 and s["block_high"]["score"] < s["block_high"]["votes"]
    order, share = ranking(s)
    assert order[0] == "cl.hp" and 0.5 < share <= 1.0


def test_ranking_share_is_zero_when_nothing_scores_positive():
    order, share = ranking({"a": {"score": -1.0}, "b": {"score": -2.0}})
    assert order == ["a", "b"] and share == 0.0


# ----------------------------------------------------------------------------------------------------- reliability
def test_reliability_rewards_backers_of_the_played_move_only():
    cfg = QuorumConfig(eta=0.5)
    rel = R.blank()
    decs = [{"when": WHEN, "action": "cl.hp",
             "quorum": {"proposals": [["attack", "cl.hp", 0.6, 1.0], ["defend", "block_high", 0.7, 1.0]]}}]
    R.credit(rel, decs, [{"dealt": 20, "taken": 0}], cfg)
    assert R.weight(rel, WHEN, "attack", cfg) > 1.0
    assert R.weight(rel, WHEN, "defend", cfg) == 1.0                 # not observed -> unchanged
    R.credit(rel, [dict(decs[0])], [{"dealt": 0, "taken": 40}], cfg)
    assert rel["rel"][WHEN]["attack"][1] == 2


def test_reliability_merge_pools_counts_in_log_space():
    a = {"rel": {WHEN: {"attack": [4.0, 1]}}}
    b = {"rel": {WHEN: {"attack": [1.0, 1]}, "far|standing|0": {"move": [2.0, 3]}}}
    m = R.merge([a, b])
    assert m["rel"][WHEN]["attack"] == [pytest.approx(2.0), 2]
    assert m["rel"]["far|standing|0"]["move"] == [pytest.approx(2.0), 3]


def test_split_child_falls_back_to_the_base_cell_weight():
    cfg = QuorumConfig()
    rel = {"rel": {WHEN: {"table": [2.0, 4]}}}
    assert R.weight(rel, WHEN + "|attack", "table", cfg) == 2.0


# ---------------------------------------------------------------------------------------------------------- voters
def test_flavor_with_one_category_skips_the_category_model():
    cat, move = Prefers(), Prefers("block_low")
    p = flavor_proposal("defend", ["block"], cat, move, "text", "close", CATS)
    assert p.voter == "defend" and p.action == "block_low" and cat.calls == []


def test_flavor_asks_the_category_model_over_its_own_categories_only():
    cat, move = Prefers("throw"), Prefers("throw_F+hp")
    p = flavor_proposal("attack", ["punch", "kick", "throw", "special", "combo"], cat, move, "text", "close", CATS)
    assert set(cat.calls[0]) <= {"punch", "kick", "throw", "special", "combo"}
    assert p.action == "throw_F+hp" and p.confidence == pytest.approx(0.8 * 0.8)


def test_flavor_abstains_when_the_stance_offers_none_of_its_categories():
    assert flavor_proposal("attack", ["throw"], Prefers(), Prefers(), "t", "standing", CATS) is None   # throws: close only


def test_base_proposal_confidence_is_the_path_probability():
    p = base_proposal(base()(None))
    assert p == Proposal("laya", "block_high", pytest.approx(0.45))


# --------------------------------------------------------------------------------------------------------- decider
def _decide(cfg, table=None, rel=None, qwen=None, seed=0, b=None):
    return quorum_decider(table or VT.blank(), rel or R.blank(), "chunli", random.Random(seed), b or base(),
                          Prefers("throw"), Prefers("throw_F+hp", "block_low", "walk_forward"), cfg, qwen)


def test_shadow_mode_plays_text_laya_and_logs_the_swarm():
    d = _decide(QuorumConfig(mode="shadow"))(make_moment(dx=36))
    assert d["action"] == "block_high" and d["source"] == "laya"          # play unchanged
    q = d["quorum"]
    voters = {p[0] for p in q["proposals"]}
    assert {"laya", "defend", "punish"} <= voters and q["laya_move"] == "block_high"
    assert set(q["scores"]) >= {"block_high", "throw_F+hp"} and d["when"] == WHEN


def test_vote_mode_acts_on_a_quorum():
    rel = {"rel": {WHEN: {"punish": [10.0, 9]}}}                           # punish dominates the tally here
    d = _decide(QuorumConfig(mode="vote", theta=0.5, epsilon=0.0), rel=rel)(make_moment(dx=36))
    assert d["source"] == "quorum" and d["action"] == "throw_F+hp" and d["quorum"]["share"] >= 0.5


def test_vote_mode_split_falls_back_to_laya_or_asks_qwen():
    cfg = QuorumConfig(mode="vote", theta=0.99, epsilon=0.0)
    d = _decide(cfg)(make_moment(dx=36))
    assert d["source"] == "fallback" and d["action"] == "block_high" and d["quorum"]["quorum_move"] is None
    asked = []
    qwen = make_qwen_pick(lambda msgs, task: asked.append(task) or "I pick throw_F+hp.")   # a candidate (punish bee)
    d = _decide(QuorumConfig(mode="vote", theta=0.99, epsilon=0.0, qwen=True), qwen=qwen)(make_moment(dx=36))
    assert asked == ["quorum_pick"] and d["source"] == "qwen" and d["action"] == "throw_F+hp"
    bad = make_qwen_pick(lambda msgs, task: "jump to the moon")         # outside the candidates -> laya
    d = _decide(QuorumConfig(mode="vote", theta=0.99, epsilon=0.0, qwen=True), qwen=bad)(make_moment(dx=36))
    assert d["source"] == "fallback"


def test_vote_mode_explores_the_second_candidate():
    d = _decide(QuorumConfig(mode="vote", epsilon=1.0))(make_moment(dx=36))
    assert d["source"] == "explore" and d["explored"] and d["action"] != d["quorum"]["top"]


def test_candidates_mode_explores_only_among_the_swarms_candidates():
    for seed in range(20):                                 # empty table: every candidate is under-sampled
        d = _decide(QuorumConfig(mode="candidates", epsilon=1.0), seed=seed)(make_moment(dx=36))
        assert d["source"] == "explore" and d["action"] in {p[1] for p in d["quorum"]["proposals"]}


def test_candidates_mode_takes_the_tables_confident_pick_else_laya():
    t = cell_table({"throw_F+hp": acc(10, 8.0), "cl.lk": acc(50, 30.0)})   # the table voter proposes cl.lk
    d = _decide(QuorumConfig(mode="candidates", epsilon=0.0), table=t)(make_moment(dx=36))
    assert d["action"] == "cl.lk" and d["source"] == "table"
    d = _decide(QuorumConfig(mode="candidates", epsilon=0.0))(make_moment(dx=36))   # no evidence: laya stands
    assert d["action"] == "block_high" and d["source"] == "laya"


def test_decider_never_plays_an_unfollowable_move():
    # far: no throws on offer, so the punish flavour cannot propose one and nothing far-illegal reaches the tally
    d = _decide(QuorumConfig(mode="vote", epsilon=0.0, theta=0.0))(make_moment(dx=150))
    assert "throw" not in d["action"] and all("throw" not in p[1] for p in d["quorum"]["proposals"])


# -------------------------------------------------------------------------------------------------------- escalate
def test_parse_reads_only_a_listed_candidate():
    assert parse("s.hp", ["s.hp", "s.mp"]) == "s.hp"
    assert parse("Go with cl.hp now", ["cl.hp", "hp"]) == "cl.hp"
    assert parse("nothing useful", ["s.hp"]) is None
    assert parse("I pick walk_forward.", ["walk_forward", "s.hp"]) == "walk_forward"
    assert parse("throw_F+hp!", ["throw_F+hp", "throw_F+mp"]) == "throw_F+hp"


# ---------------------------------------------------------------------------------------------------------- genome
def test_genome_round_trips_and_mutation_stays_in_bounds():
    g = G.encode(QuorumConfig(mode="vote"))
    assert set(g) == set(G.GENES)
    rng = random.Random(0)
    for _ in range(50):
        g = G.mutate(g, rng, sigma=0.5)
        cfg = G.decode(g, QuorumConfig(mode="vote"))
        assert 0.3 <= cfg.theta <= 0.9 and 0.0 <= cfg.epsilon <= 0.2 and cfg.mode == "vote"


def test_fitness_penalises_turtling_and_escalation():
    win = [{"dealt": 30, "taken": 10, "result": "win"}]
    turtle = [{"dealt": 21, "taken": 0, "result": "loss"}]
    assert G.fitness(win, 0.05) > G.fitness(turtle, 0.05)
    assert G.fitness(win, 0.60) < G.fitness(win, 0.05)


# ------------------------------------------------------------------------------------------------- driver switch
class FakeEmu:
    def new_round(self):
        return self


def fake_play(emu, cat, move, me, opp, state, state_id, delay, lines, out, reader=None, decide=None):
    from sf2.system1.loop_runner import DECISION_KEYS
    os.makedirs(out, exist_ok=True)
    d = decide(make_moment(dx=36, doing="standing", my_bar=120 / 176.0, his_bar=150 / 176.0))
    rec = {"i": 0, "k": 12, "k_prev": 8, "action": d["action"], "situation": ["close", "standing", "half", "half"],
           "moment": {"my_life": 120, "his_life": 150, "doing": "standing", "his_air": False, "his_label": "stand",
                      "side": "left", "dx": 36, "fireball": False},
           **{x: d[x] for x in DECISION_KEYS if x in d}}
    with open(os.path.join(out, "decisions.jsonl"), "w") as f:
        f.write(json.dumps(rec) + "\n")
    return {"decisions": 1}


def fake_score(round_dir):
    return {"result": "win", "dealt": 30, "taken": 0, "hp": 30, "my_life_end": 120, "opp_life_end": 120}


def test_run_loop_policy_quorum_credits_table_and_reliability(tmp_path):
    driver = load_driver()
    out, save_t, save_q = str(tmp_path / "q"), str(tmp_path / "t.json"), str(tmp_path / "rel.json")
    verdict = driver.run_loop("ryu", FollowerLaya(), FollowerLaya(), lambda messages, task: "", games=2, rounds=2,
                              seed_lines=[], out=out, play_round_fn=fake_play, state=b"x",
                              state_id={"path": "p", "sha256": "0"}, emu=FakeEmu(), score_fn=fake_score, seed_rng=1,
                              policy="quorum", save_table=save_t, save_quorum=save_q,
                              quorum=QuorumConfig(mode="vote", epsilon=0.0))
    assert verdict["policy"] == "quorum" and verdict["table_cells"] >= 1
    q = verdict["quorum"]
    assert q["mode"] == "vote" and q["decisions"] == 4 and 0.0 <= q["escalation_rate"] <= 1.0
    rel = json.load(open(save_q))
    assert WHEN in rel["rel"] and any(n >= 1 for _, n in rel["rel"][WHEN].values())   # backers were credited
    events = [json.loads(l) for l in open(os.path.join(out, "trace.jsonl"))]
    assert sum(e.get("event") == "quorum" for e in events) == 4
    assert all(e.get("quorum") for e in events if e.get("event") == "decision")


def test_evolve_dry_run_improves_on_the_fake_bowl(tmp_path):
    out = str(tmp_path / "evo")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/quorum_evolve.py"), "--out", out, "--dry-run",
                        "--generations", "12", "--lam", "6", "--seed", "3"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    log = [json.loads(l) for l in open(os.path.join(out, "log.jsonl"))]
    first = max(e["fitness"] for e in log if e["gen"] == 0)
    last = max(e["fitness"] for e in log if e["gen"] == 11)
    assert last > first
    assert QuorumConfig.load(os.path.join(out, "best_config.json")).mode == "vote"


def test_quorum_report_reads_a_run(tmp_path):
    driver = load_driver()
    out = str(tmp_path / "q")
    driver.run_loop("ryu", FollowerLaya(), FollowerLaya(), lambda messages, task: "", games=1, rounds=3,
                    seed_lines=[], out=out, play_round_fn=fake_play, state=b"x",
                    state_id={"path": "p", "sha256": "0"}, emu=FakeEmu(), score_fn=fake_score, seed_rng=1,
                    policy="quorum", quorum=QuorumConfig(mode="shadow"))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts/quorum_report.py"), out, "--json"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = json.loads(r.stdout)
    assert rep["decisions"] == 3 and rep["sources"] == {"laya": 3}
    assert sum(b["n"] for b in rep["agreement_vs_outcome"]) == 3 and "laya" in rep["voters"]


def test_option2_bee_set_is_pinned():
    # Option-2 (owner 2026-10-05): base laya + defend(anti-pressure) + punish(openings incl. overlooked combos) + table.
    from sf2.quorum.config import FLAVORS, VOTERS
    assert set(VOTERS) == {"laya", "defend", "punish", "combo", "table"}  # dropped broad 'attack'/'move'; combo is its own bee
    assert FLAVORS["defend"] == ["block", "move"]
    assert FLAVORS["punish"] == ["punch", "special", "throw"]
    assert FLAVORS["combo"] == ["combo"]                                 # dedicated: forces combos (air-only) into the vote
