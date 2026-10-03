"""Option D: screen -> powered sequential confirm, with the owner's WIN-based stop. Test-first.

The objective is WINNING, not HP-margin maximization: once the incumbent reliably beats the opponent
it is SOLVED and the lesson set FREEZES (no more churn) until the opponent ranks up - chasing pure
knockout margin would overfit the CPU's fixed patterns ("the game shrinks"). While still climbing,
HP-margin is the learning signal (continuous, high-information; win-rate at n=12 is too coarse):
  SCREEN (dev, biased by selection - NOT evidence): fires >= floor AND one-sided lower bound > 0.
  CONFIRM (fresh seeds, POOLED across sequential blocks, one-sided): promote at z >= 2.0 with a
    win-rate veto; futility-reject if the pooled gain is too small; give up after max_looks.
Reviewers (Fable + gpt6) converged on this; fixtures include the real Guile r0/r1 shapes.
"""
import pytest

from sf2.system2.sequential import Block, SeqCfg, confirm, screen, solved


def _blk(inc_hp, cand_hp, inc_wins=None, cand_wins=None, fire=0.6, follows=1.0):
    iw = inc_wins if inc_wins is not None else sum(h > 0 for h in inc_hp)
    cw = cand_wins if cand_wins is not None else sum(h > 0 for h in cand_hp)
    return Block(inc_hp=tuple(inc_hp), cand_hp=tuple(cand_hp), inc_wins=iw, cand_wins=cw,
                 fire_rate=fire, follows=follows)


# ---------------- SCREEN ----------------
def test_screen_passes_clear_positive_above_fire_floor():
    dev = _blk([0] * 12, [40] * 11 + [20], fire=0.55)   # strong positive, low variance, fires 55%
    ok, why = screen(dev)
    assert ok, why


def test_screen_rejects_below_fire_floor():
    dev = _blk([0] * 12, [40] * 12, fire=0.30)
    ok, why = screen(dev)
    assert not ok and "fire" in why.lower()


def test_screen_rejects_when_one_sided_lower_bound_not_above_zero():
    # big mean but huge variance -> one-sided lower bound <= 0 -> screened out
    dev = _blk([0] * 12, [120, -120, 100, -90, 80, -70, 60, -50, 40, -30, 20, -10], fire=0.6)
    ok, why = screen(dev)
    assert not ok


# ---------------- CONFIRM (sequential, pooled, one-sided) ----------------
def test_confirm_promote_when_pooled_z_clears_and_wins_not_vetoed():
    b = _blk([0] * 12, [40] * 12, fire=0.6)             # very strong, cand wins all
    dec = confirm([b])
    assert dec.verdict == "promote", dec.reason


def test_confirm_continue_when_directional_but_underpowered():
    # Guile r0 HELD shape: delta ~ +29.5, se ~ +28 -> z ~ 1.05 -> neither promote nor futile -> continue
    cand = [29.5 + v for v in (140, -140, 120, -120, 100, -100, 80, -80, 60, -60, 40, -40)]
    dec = confirm([_blk([0] * 12, cand, fire=0.6)])
    assert dec.verdict == "continue", (dec.verdict, dec.z, dec.reason)


def test_confirm_futility_reject_when_pooled_gain_tiny():
    cand = [2, -1, 3, -2, 1, 0, 2, -1, 1, 0, 1, -1]     # mean ~ +0.4, clearly not worth it
    dec = confirm([_blk([0] * 12, cand, fire=0.6)])
    assert dec.verdict == "reject", dec.reason


def test_confirm_pools_blocks_and_promotes_once_powered():
    # one block is underpowered (continue); a second fresh block pools to clear z>=2
    blk = lambda: _blk([0] * 12, [30 + v for v in (100, -100, 90, -90, 70, -70, 50, -50, 30, -30, 10, -10)], fire=0.6)
    one = confirm([blk()])
    assert one.verdict == "continue", (one.z, one.reason)
    two = confirm([blk(), blk()])            # pooled n=24 shrinks the SE by ~sqrt(2) -> z clears 2
    assert two.verdict == "promote", (two.z, two.reason)


def test_confirm_gives_up_after_max_looks():
    cfg = SeqCfg(max_looks=2)
    cand = [29 + v for v in (60, -55, 50, -45, 40, -35, 30, -25, 20, -15, 10, -5)]
    blocks = [_blk([0] * 12, cand, fire=0.6), _blk([0] * 12, cand, fire=0.6)]
    # construct so z stays < 2 across 2 looks by keeping variance high relative to n
    dec = confirm(blocks, cfg)
    assert dec.verdict in ("reject", "promote")  # must TERMINATE at max_looks, never 'continue'
    assert dec.verdict == "reject" or dec.z >= cfg.confirm_z


def test_confirm_win_veto_blocks_a_margin_only_gain():
    # candidate piles HP in a few blowout wins but LOSES more rounds than the incumbent -> vetoed
    cand = [200, 200, 200, -5, -5, -5, -5, -5, -5, -5, -5, -5]   # big mean, only 3 wins
    inc = [1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1]               # 8 wins
    dec = confirm([_blk(inc, cand, inc_wins=8, cand_wins=3, fire=0.6)])
    assert dec.verdict != "promote", dec.reason


# ---------------- SOLVED / FREEZE (owner's win objective) ----------------
def test_solved_when_incumbent_reliably_wins():
    assert solved(inc_wins=9, n=12) is True       # 75% round win-rate -> opponent beaten, freeze
    assert solved(inc_wins=6, n=12) is False      # 50% -> not solved, keep learning


def test_solved_threshold_configurable():
    assert solved(inc_wins=8, n=12, cfg=SeqCfg(win_target=0.60)) is True   # 67% >= 60%
    assert solved(inc_wins=8, n=12, cfg=SeqCfg(win_target=0.80)) is False  # 67% < 80%


def test_block_validation():
    with pytest.raises(ValueError):
        confirm([])                                           # no blocks
    with pytest.raises(ValueError):
        screen(_blk([0], [1], fire=0.6))                      # need >= 2 samples/arm for a variance
