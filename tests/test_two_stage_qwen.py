"""Two-stage Qwen (System 2): Stage 1 SCOUT summarizes a game, Stage 2 COACH strategizes rule changes, and the churn
fix promotes rules that keep winning / retires rules correlated with losing (docs/plan_two_stage_qwen.md). All with a
MOCK Qwen - no network, no emulator.

The single prompt turtled: it kept ADDING blocks and nothing stuck (14 added / 1 removed in 26 rounds, ~23% win). The
split lets each side be prompted for its own job, and the churn fix makes good rules STICK and losing rules DROP.

Seen RED (how each assertion was confirmed able to fail), per test below. The Stage-1/Stage-2/Coach functions did not
exist before this change, so their tests error (AttributeError) against the pre-change module - the new-feature red.
The churn tests have a concrete pre-change twin: the old ``review`` (no ``games_wl``) keeps a loss-correlated rule
registered and never promotes, so both churn assertions were run against ``review(... )`` WITHOUT ``games_wl`` and
seen to give the old state (asserted here as the red twin).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from looptools import FollowerLaya, load_driver, make_moment        # noqa: E402

from sf2.system1.loop_runner import two_stage_decide                # noqa: E402
from sf2.system2 import character_prompt as C                       # noqa: E402
from sf2.system2 import lessons as L                                # noqa: E402

ME, OPP = "chunli", "honda"


# --------------------------------------------------------------- crafted rows (this game) and registry
def row(move, rng, dealt=0, taken=0, doing="attacking", kind="attack"):
    air = doing == "jumping"
    state = {"crouching": "crouch", "attacking": "attack", "stunned": "hit_stun", "jumping": "stand"}.get(doing, "stand")
    return {"action": move, "range": rng, "kind": kind, "dealt": dealt, "taken": taken, "opp_air": air,
            "opp_state": state, "opp_reaction": ["attack"] if taken else ["stand"], "actual": "hit" if dealt else "whiff",
            "opp_move": None, "opp_shot": False}


def turtling_game():
    """She blocks low 30 times (takes a lot, deals nothing), throws a few sweeps that whiff: a losing, turtling game."""
    return ([row("block_low", "close", dealt=0, taken=12, kind="defense") for _ in range(30)]
            + [row("sweep", "close", dealt=0, taken=10) for _ in range(8)]
            + [row("c.mk", "mid", dealt=4, taken=6) for _ in range(6)])


# --------------------------------------------------------------- STAGE 1: the scout digest
def test_stage1_digest_reads_the_facts_from_the_rows():
    """Seen RED: ``C.digest_facts`` did not exist before this change (AttributeError). The values below are computed
    from the crafted rows, so a wrong digest (e.g. summing the wrong column) fails the equality asserts."""
    rows = turtling_game()
    d = C.digest_facts(ME, OPP, [], rows, rows, [], [], game_hp=[-120.0], games_wl=[{"won": 0, "lost": 2}])
    assert d["dominant"][0] == ["block_low", 30]                       # her dominant action
    assert d["dealt"] == 24 and d["taken"] == 30 * 12 + 8 * 10 + 6 * 6  # totals straight off the rows
    assert d["verdict"] == "losing"                                    # lost 2 / won 0, negative hp
    assert set(d["offense_used"]) == {"sweep", "c.mk"} and "block_low" not in d["offense_used"]
    assert d["already_tried"] == sorted({"block_low", "sweep", "c.mk"})  # the list the Coach must not repeat


def test_stage1_verdict_is_winning_when_she_wins_the_recent_games():
    rows = [row("sweep", "close", dealt=20, taken=0) for _ in range(10)]
    d = C.digest_facts(ME, OPP, [], rows, rows, [], [], game_hp=[80.0], games_wl=[{"won": 2, "lost": 0}])
    assert d["verdict"] == "winning" and C.coach_mode(d) == "consolidate"


def test_stage1_summarize_game_attaches_scout_prose_and_traces(tmp_path):
    """The driver's Stage-1 step calls Qwen once and attaches its prose as ``notes`` (the facts stand on their own if
    it fails). Seen RED: ``summarize_game`` did not exist, and no "scout" event was written to the trace."""
    driver = load_driver()
    rows = turtling_game()
    d = driver.summarize_game(OPP, [], rows, rows, [], [], [-120.0], [{"won": 0, "lost": 2}],
                              lambda msgs, task: "She turtled and lost.", ME)
    assert d["notes"] == "She turtled and lost." and d["verdict"] == "losing"

    # a down Scout leaves the facts intact (no prose, an error noted) - the integration-test discipline: facts are
    # queryable, the model only narrates them
    def boom(msgs, task):
        raise RuntimeError("scout down")
    d2 = driver.summarize_game(OPP, [], rows, rows, [], [], [-120.0], [{"won": 0, "lost": 2}], boom, ME)
    assert d2["notes"] == "" and "scout down" in d2["scout_error"] and d2["verdict"] == "losing"


# --------------------------------------------------------------- STAGE 2: the coach, two modes
def test_stage2_escalate_forbids_a_block_answer_keeps_offense():
    """ESCALATE (losing by turtling): a new DEFENSIVE "use more"/"always" answer is dropped mechanically; an offense
    answer survives. Seen RED: ``C.coach_filter`` did not exist; and with mode != escalate nothing is dropped (the
    consolidate twin below keeps the block), so the drop is specific to escalating."""
    block = {"view": "answer", "kind": "always", "move": "block_low", "range": "close", "when": "attacking"}
    offense = {"view": "answer", "kind": "use_more", "move": "sweep", "range": "close", "when": "attacking"}
    avoid = {"view": "stop", "kind": "avoid", "move": "back", "range": None, "when": "attacking"}

    kept, dropped = C.coach_filter([block, offense, avoid], "escalate")
    moves = [c["move"] for c in kept]
    assert "block_low" not in moves and all(not m.startswith("block_") for m in moves)   # no defensive answer
    assert "sweep" in moves and "back" in moves                        # offense kept; an avoid-defense kept
    assert dropped and "block_low" in dropped[0]

    kept2, dropped2 = C.coach_filter([block, offense], "consolidate")   # the twin: consolidate keeps the block
    assert [c["move"] for c in kept2] == ["block_low", "sweep"] and dropped2 == []


def test_stage2_strategize_prompt_switches_on_the_trend():
    """The strategize system prompt forbids defense only when escalating, and names the already-tried list; the
    consolidate prompt talks about promoting what works. Seen RED: ``C.strategize_messages`` did not exist."""
    losing = C.digest_facts(ME, OPP, [], turtling_game(), turtling_game(), [], [], [-120.0], [{"won": 0, "lost": 2}])
    esc = C.strategize_messages(ME, OPP, [], losing, C.lesson_moves(["sweep", "c.mk", "block_low"]))[0]["content"]
    assert "FORBIDDEN" in esc and "block_low" in esc and "ALREADY TRIED" in esc

    winning = C.digest_facts(ME, OPP, [], turtling_game(), turtling_game(), [], [], [80.0], [{"won": 2, "lost": 0}])
    con = C.strategize_messages(ME, OPP, [], winning, C.lesson_moves(["sweep"]))[0]["content"]
    assert "Consolidate" in con and "FORBIDDEN" not in con


def test_stage2_answer_grammar_is_the_same_json_parse_claims_reads():
    """The Coach's JSON schema is byte-for-byte the single prompt's, so parse_claims is unchanged. Seen RED: a schema
    drift (e.g. renaming "answer") makes parse_claims return no claims."""
    d = C.digest_facts(ME, OPP, [], turtling_game(), turtling_game(), [], [], [-120.0], [{"won": 0, "lost": 2}])
    system = C.strategize_messages(ME, OPP, [], d, C.lesson_moves(["sweep"]))[0]["content"]
    body = system[system.index('{"answer"'):]
    assert set(json.loads(body.replace("|", "_"))) == {"answer", "stop"}
    claims, problems = C.parse_claims({"answer": {"kind": "use_more", "move": "sweep", "range": "close",
                                                  "when": "attacking"}, "stop": None})
    assert claims and claims[0]["view"] == "answer" and not problems


def test_stage2_escalate_prompt_asks_for_common_grounded_offense():
    """BROADEN: the escalate prompt tells the Coach to propose offense for COMMON situations, that FIRES OFTEN, a
    GROUNDED anti-air (not a jump attack). Seen RED: the old ESCALATE text asked only for narrow offense ("an anti-air
    when he jumps") and named none of these strings, so each assert fails against the pre-change prompt."""
    losing = C.digest_facts(ME, OPP, [], turtling_game(), turtling_game(), [], [], [-120.0], [{"won": 0, "lost": 2}])
    esc = C.strategize_messages(ME, OPP, [], losing, C.lesson_moves(["sweep", "c.mk", "block_low"]))[0]["content"]
    assert "COMMON situations" in esc and "FIRES OFTEN" in esc            # broad, common conditions - not rare/narrow
    assert "GROUNDED" in esc and "jump attack" in esc                    # grounded anti-air; no jump attack on his cue
    assert "range-only" in esc                                           # prefer range-only / common-state rules


def test_stage2_coach_filter_drops_stance_invalid_jump_attacks():
    """STANCE guard: a jump attack ("j."/"jf.") keyed on HIS state is stance-invalid (she can't be airborne on cue) and
    is DROPPED with a logged reason; a GROUNDED anti-air (shoryuken_hp when he jumps), a grounded special
    (spinning_bird_kick when he stands) and a range-only poke (c.mk at mid) all pass. Seen RED: before this change
    coach_filter had no stance check, so j.hp/jf.hk were KEPT (the assert that they are absent fails)."""
    jump_air = {"view": "answer", "kind": "use_more", "move": "j.hp", "range": None, "when": "jumping"}
    jump_fwd = {"view": "answer", "kind": "use_more", "move": "jf.hk", "range": None, "when": "jumping"}
    grounded_aa = {"view": "answer", "kind": "use_more", "move": "shoryuken_hp", "range": "close", "when": "jumping"}
    grounded_sp = {"view": "answer", "kind": "use_more", "move": "spinning_bird_kick", "range": "mid",
                   "when": "standing"}
    poke = {"view": "answer", "kind": "use_more", "move": "c.mk", "range": "mid", "when": "standing"}

    kept, dropped = C.coach_filter([jump_air, jump_fwd, grounded_aa, grounded_sp, poke], "escalate")
    moves = [c["move"] for c in kept]
    assert "j.hp" not in moves and "jf.hk" not in moves                  # stance-invalid jump attacks dropped
    assert moves == ["shoryuken_hp", "spinning_bird_kick", "c.mk"]       # grounded offense + range-only poke kept
    assert len(dropped) == 2 and all("stance" in r and "jump attack" in r for r in dropped)  # reason logged for trace

    # the stance guard fires in consolidate mode too (a jump attack is invalid regardless of the trend)
    kept2, dropped2 = C.coach_filter([jump_air, poke], "consolidate")
    assert [c["move"] for c in kept2] == ["c.mk"] and len(dropped2) == 1


# --------------------------------------------------------------- the churn fix (lessons.py, independent of the split)
def good_rows():
    """Up close when he attacks: block_low nets -4 (better than her ~-10 average), sweep -18."""
    import random
    r = random.Random(0)

    def many(move, mean, n, kind="attack"):
        return [row(move, "close", dealt=max(mean + r.randint(-3, 3), 0),
                    taken=max(-(mean + r.randint(-3, 3)), 0), kind=kind) for _ in range(n)]
    return many("block_low", -4, 40, "defense") + many("sweep", -18, 40) + many("c.mk", -10, 40)


def test_churn_promotes_a_working_rule_that_keeps_winning():
    """A registered rule that survives PROMOTE_GAMES, still tracks good outcomes AND is winning becomes "sticky" so a
    per-decision CI flip no longer drops it. Seen RED (twin): the SAME review WITHOUT ``games_wl`` leaves it
    "registered" (the old code never promotes)."""
    rows = good_rows()
    reg, out = L.propose([], [{"kind": "always", "move": "block_low", "range": "close", "when": "attacking"}], rows, 0)
    assert out[0]["state"] == "registered"
    wl = [{"won": 2, "lost": 1}] * 4
    promoted = L.review(reg, rows, 3, game_hp=[10, 10, 10, 10], games_wl=wl)
    assert promoted[0]["state"] == "sticky" and "promoted" in promoted[0]["why"]
    assert promoted[0]["line"] in L.in_play(promoted)                  # a sticky line stays in play like a verified one
    # red twin: the old review (no games_wl) keeps it registered
    assert L.review(reg, rows, 3, game_hp=[10, 10, 10, 10])[0]["state"] == "registered"


def test_churn_retires_a_rule_correlated_with_losing():
    """A registered rule whose games since it began are lost is retired on the round outcomes, even while its
    per-decision evidence still holds. Seen RED (twin): the SAME review WITHOUT ``games_wl`` keeps it registered -
    exactly the low-churn turtling the single prompt produced."""
    rows = good_rows()
    reg, out = L.propose([], [{"kind": "avoid", "move": "sweep", "range": "close", "when": "attacking"}], rows, 0)
    assert out[0]["state"] == "registered"                             # clearly worse -> a valid avoid rule
    wl = [{"won": 0, "lost": 2}] * 4
    retired = L.review(reg, rows, 3, game_hp=[-100, -100, -100, -100], games_wl=wl)
    assert retired[0]["state"] == "retired" and "losing" in retired[0]["why"]
    # red twin: without the round outcomes the per-decision yardstick still backs it -> stays registered
    assert L.review(reg, rows, 3, game_hp=[-100, -100, -100, -100])[0]["state"] == "registered"


# --------------------------------------------------------------- end to end through the driver (mock Qwen)
class FakeEmu:
    def new_round(self):
        return self


def fake_play(emu, cat_advisor, move_advisor, me, opp, state, state_id, delay, lines, out, reader=None):
    os.makedirs(out, exist_ok=True)
    m = make_moment(dx=36, doing="attacking", my_bar=150 / 176.0, his_bar=150 / 176.0)
    d = two_stage_decide(cat_advisor, move_advisor, me, m, lines)
    rec = dict(d, i=0, k=12, k_prev=8, situation=["close", "attacking", "half", "half"],
               moment={"my_life": 150, "his_life": 150, "doing": "attacking", "his_air": False,
                       "side": "left", "dx": 36, "fireball": False})
    with open(os.path.join(out, "decisions.jsonl"), "w") as f:
        f.write(json.dumps(rec) + "\n")
    return {"decisions": 1}


def fake_score(round_dir):
    return {"result": "loss", "dealt": 20, "taken": 40, "hp": -20, "my_life_end": 100, "opp_life_end": 130}


class EscalatingBlockCoach:
    """A losing session: the Scout gets prose; the Coach (naively) answers with a BLOCK. The escalate guard must drop
    it, so no block_* answer ever reaches the short memory. Records every task it saw."""

    def __init__(self):
        self.tasks = []

    def __call__(self, messages, task):
        self.tasks.append(task)
        if task.startswith("scout"):
            return "She is losing by blocking."
        return json.dumps({"answer": {"kind": "always", "move": "block_high", "range": "close",
                                      "when": "attacking", "why": "turtle"}, "stop": None})


def _events(out, kind):
    with open(os.path.join(out, "trace.jsonl")) as f:
        return [json.loads(l) for l in f if json.loads(l).get("event") == kind]


def test_two_stage_run_drops_the_block_answer_and_traces_the_scout(tmp_path):
    """End to end: in a losing session the escalate guard means the Coach's block answer never enters play, and the
    run writes a per-game "scout" event. Seen RED: without the guard the block would be admitted (a block_* line in
    play); without the Scout step there is no scout event."""
    driver = load_driver()
    coach = EscalatingBlockCoach()
    out = str(tmp_path / "run")
    verdict = driver.run_loop(OPP, FollowerLaya(), FollowerLaya(), coach, games=3, rounds=1, seed_lines=[], out=out,
                              play_round_fn=fake_play, state=b"x", state_id={"path": "p", "sha256": "0"},
                              emu=FakeEmu(), score_fn=fake_score, seed_rng=1, qwen_mode="two")
    assert not any(l.startswith("block_") for l in verdict["in_play_end"]), verdict["in_play_end"]
    assert any(t.startswith("scout") for t in coach.tasks) and any(t.startswith("coach") for t in coach.tasks)
    scouts = _events(out, "scout")
    assert len(scouts) == 3 and scouts[0]["verdict"] == "losing"       # a scout digest per game
    qwen = _events(out, "qwen")
    assert qwen[0]["mode"] == "escalate" and any("forbidden" in p.lower() for p in qwen[0]["problems"])


def test_qwen_mode_one_still_uses_the_old_single_prompt(tmp_path):
    """--qwen-mode one is the A/B fallback: the OLD single prompt, one call per game, no scout event. Seen RED: if the
    flag were ignored and two-stage always ran, a "scout" event would appear and the single-call claim wiring
    (task "loop_*") would not drive the move."""
    driver = load_driver()
    seen = {"tasks": []}

    def mock(messages, task):
        seen["tasks"].append(task)
        if len(seen["tasks"]) == 1:
            return json.dumps({"answer": {"kind": "always", "move": "cl.hp", "range": "close",
                                          "when": "attacking", "why": "it lands"}, "stop": None})
        return json.dumps({"answer": None, "stop": None})

    out = str(tmp_path / "run_one")
    verdict = driver.run_loop(OPP, FollowerLaya(), FollowerLaya(), mock, games=2, rounds=1, seed_lines=[], out=out,
                              play_round_fn=fake_play, state=b"x", state_id={"path": "p", "sha256": "0"},
                              emu=FakeEmu(), score_fn=fake_score, seed_rng=1, qwen_mode="one")
    assert all(t.startswith("loop_") for t in seen["tasks"])           # the single prompt, never scout/coach
    assert _events(out, "scout") == []                                 # no Stage-1 event
    assert "always cl.hp up close when he attacks" in verdict["in_play_end"]
