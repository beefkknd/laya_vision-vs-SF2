"""Option-D session driver paths, test-first (measurer + proposer faked, no games).

Covers: SOLVED-at-start freeze (win objective); SCREEN-reject skip; SCREEN -> CONFIRM -> PROMOTE then
re-check SOLVED and freeze; SCREEN -> CONTINUE -> REJECT (directional-unconfirmed).
"""
from sf2.system2.outcome_loop import Playbook
from sf2.system2.seq_loop import SeqSeeds, run_opponent
from sf2.system2.sequential import Block, SeqCfg

BOOK = Playbook("book", ("use more throw up close",))
WIN = Playbook("winner", ("use more throw up close", "use more s.mk at mid range when he attacks"))
MEH = Playbook("meh", ("use more throw up close", "use more s.mp at mid range when he stands"))

SEEDS = SeqSeeds(solved=(90, 91), pool=tuple((i, i + 1) for i in range(0, 40, 2)), terminal=(98, 99))

STRONG = Block(inc_hp=(0,) * 12, cand_hp=(40,) * 12, inc_wins=4, cand_wins=11, fire_rate=0.7, follows=1.0)
UNDERPOWERED = Block(inc_hp=(0,) * 12, cand_hp=tuple(29.5 + v for v in (140, -140, 120, -120, 100, -100, 80, -80, 60, -60, 40, -40)),
                     inc_wins=5, cand_wins=6, fire_rate=0.6, follows=1.0)
LOWFIRE = Block(inc_hp=(0,) * 12, cand_hp=(40,) * 12, inc_wins=4, cand_wins=11, fire_rate=0.2, follows=1.0)


def _wins(fixed):
    return lambda pb, seeds: (fixed, 12)


def test_solved_at_start_freezes_without_proposing():
    proposed = []

    def propose(opp, inc, k):
        proposed.append(k)
        return [WIN]

    r = run_opponent("guile", BOOK, propose, lambda i, c, s: STRONG, _wins(9), SEEDS, n_rounds=3)
    assert r.status == "solved" and r.final.id == "book"
    assert proposed == [], "must not propose when already solved"
    assert any(row["stage"] == "solved_check" for row in r.rows)


def test_screen_reject_skips_candidate():
    r = run_opponent("guile", BOOK, lambda o, i, k: [MEH] if k == 0 else [],
                     lambda i, c, s: LOWFIRE, _wins(5), SEEDS, n_rounds=1)
    assert r.status == "unsolved" and r.final.id == "book"
    scr = [row for row in r.rows if row["stage"] == "screen"]
    assert scr and scr[0]["screened"] is False
    assert not any(row["stage"] == "confirm" for row in r.rows)


def test_screen_then_confirm_promote_then_solved():
    # strong candidate: screens, confirms on the first block, promotes; then incumbent is solved -> freeze
    calls = {"n": 0}

    def measure(i, c, s):
        calls["n"] += 1
        return STRONG

    # incumbent not solved at first (5 wins), but solved after promotion (9 wins)
    seq = iter([5, 9])

    def wins(pb, s):
        return (next(seq), 12)

    r = run_opponent("guile", BOOK, lambda o, i, k: [WIN], measure, wins, SEEDS, n_rounds=3)
    assert r.final.id == "winner"
    assert r.status == "solved"
    assert any(row["stage"] == "promote" for row in r.rows)


def test_screen_then_continue_then_reject_after_max_looks():
    # dev screens through (strong), but each fresh CONFIRM block is only the +29 directional effect
    # (z~1.06); pooled over 3 looks it still can't clear z>=2 -> reject (directional-unconfirmed).
    calls = {"n": 0}

    def measure(i, c, s):
        calls["n"] += 1
        return STRONG if calls["n"] == 1 else UNDERPOWERED  # 1st call = dev screen, rest = confirm

    r = run_opponent("guile", BOOK, lambda o, i, k: [WIN] if k == 0 else [],
                     measure, _wins(5), SEEDS, n_rounds=1, cfg=SeqCfg(max_looks=3))
    assert r.final.id == "book", "underpowered directional effect must NOT promote"
    confirms = [row for row in r.rows if row["stage"] == "confirm"]
    assert len(confirms) == 3, "should take exactly max_looks confirm blocks"
    assert confirms[-1]["verdict"] == "reject"
    assert r.looks_used == 3


def test_out_of_seeds_is_handled():
    tiny = SeqSeeds(solved=(90, 91), pool=((0, 1),), terminal=(98, 99))  # only 1 block -> dev ok, no confirm seeds
    r = run_opponent("guile", BOOK, lambda o, i, k: [WIN], lambda i, c, s: STRONG, _wins(5), tiny, n_rounds=1)
    assert r.final.id == "book"
    assert any(row["stage"] == "out_of_seeds" for row in r.rows)
