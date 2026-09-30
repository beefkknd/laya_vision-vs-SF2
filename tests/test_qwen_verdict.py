"""scripts/qwen_lessons.py verdict: the chance baseline draws over POPULATED cells only (a claim on an empty cell can
never hold), and the verdict says how much of Qwen's "hold rate" was already on the table it was shown (registered at
proposal) and how often it repeated itself (refused as a duplicate). 2026-09-29 review, finding 2."""
import json
import os

from sf2.system2 import lessons as L
from sf2.system2.move_coach import MIN_TRIES
from tests.test_character_prompt import world
from tests.test_lock import qwen_lessons

Q = qwen_lessons()


def populated(rows, c):
    ev = L.condition_evidence(rows, c)
    return ev["tries"] >= MIN_TRIES and ev["others"] >= MIN_TRIES


def test_random_claims_fall_on_populated_cells_only():
    rows = world()
    claims = Q.random_claims(rows, 300)
    assert claims and all(populated(rows, c) for c in claims)
    assert {c["kind"] for c in claims} == set(L.KINDS)                     # a random kind
    assert {c["move"] for c in claims} == {"sweep", "c.mk"}                # hp and forward: 10 tries, too few


def test_random_claims_for_a_character_prompt_name_what_he_is_doing():
    rows = world()
    claims = Q.random_claims(rows, 200, need_when=True)
    assert claims and all(c["when"] and populated(rows, c) for c in claims)


def test_no_populated_cell_no_baseline():
    assert Q.random_claims(world()[:15], 50) == []


def jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


def claim(move, kind="use_more", rng=None, when=None):
    return {"kind": kind, "move": move, "range": rng, "when": when, "view": "attack"}


def test_verdict_reports_registered_at_proposal_and_duplicates(tmp_path):
    root = str(tmp_path)
    rows = world()
    outcome = [{"claim": claim("c.mk"), "state": "registered", "why": "clearly better"},
               {"claim": claim("sweep", "avoid"), "state": "rejected", "why": "not clearly worse"},
               {"claim": claim("hp"), "state": "testing", "why": "to be tried in play"},
               {"claim": claim("c.mk"), "state": "refused", "why": "already registered"},
               {"claim": claim("hp"), "state": "refused", "why": "already testing"},
               {"claim": claim("sweep"), "state": "refused", "why": "already rejected: clearly worse"},
               {"claim": claim("lp"), "state": "refused", "why": "already 2 claims in test"},    # not a repeat
               {"claim": claim("zz"), "state": "refused", "why": "zz is not one of her moves"}]
    jsonl(os.path.join(root, "loop", "ledger.jsonl"), [{"outcome": outcome, "violations": [], "registry": []}])
    jsonl(os.path.join(root, "loop", "actions.jsonl"), rows)
    for arm in ("loop", "none"):
        jsonl(os.path.join(root, arm, "rounds.jsonl"), [{"dealt": 5, "taken": 0, "result": "win"}])
    v = Q.verdict(root, "ken", history=False)
    assert v["proposed"] == 8 and v["valid"] == 3
    assert v["registered_at_proposal"] == 1 and abs(v["registered_at_proposal_share"] - 1 / 3) < 1e-9
    assert v["refused_duplicate"] == 3 and v["refused_duplicate_share"] == 3 / 8
    assert v["random_cells"] > 0 and v["random_hold_rate"] is not None
    assert "random_hold_rate_uniform" in v


def test_verdict_without_a_ledger_has_no_shares(tmp_path):
    v = Q.verdict(str(tmp_path), "ken", history=False)
    assert v["registered_at_proposal_share"] is None and v["refused_duplicate_share"] is None
