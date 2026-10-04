"""BUG (found in the chun4 live test, Fable-grounded): per-decision credit is contaminated by DELAYED HITS.

When she lands an attack at decision t, he enters hit-stun and his health BAR drains over the next frames --
read in decision t+1's window. So a NON-damaging move she plays at t+1 (block/movement) gets credited with the
previous attack's damage. Measured in chun4: block_high in (mid,jumping) netted +19.7.

Fix (post-review, kind-based): a move whose kind is not "attack" (block/movement) cannot deal damage, so any
`dealt` in its window is delayed -> reattribute it to the nearest EARLIER ATTACK. An ATTACK keeps its own window
(so a real combo follow-up is credited, not zeroed). Keying on HER move (not the opponent's hit label) also avoids
crediting an unrecognised-sprite 'block' frame. Damage is conserved.
"""
from sf2.system2 import screen_evidence as SE


def _dec(action, his_life, doing="standing", rng="mid"):
    return {"action": action, "situation": [rng, doing, "full", "full"],
            "moment": {"my_life": 100, "his_life": his_life, "doing": doing, "his_air": doing == "jumping",
                       "his_label": "stand", "dx": 40, "side": "left"}}


def _dealt(rows):
    return {r["action"]: r["dealt"] for r in rows}


def test_delayed_hit_on_a_block_is_reattributed_to_the_attack():
    # lightning_legs connects (t0); she BLOCKS at t1 while his bar drains 100->60; the 40 belongs to the legs.
    rows, summary = SE.round_evidence(0, "chunli", "ryu", [
        _dec("lightning_legs", 100), _dec("block_high", 100), _dec("s.mk", 60)], replay=None)
    d = _dealt(rows)
    assert d["block_high"] == 0, "the block must NOT be credited with the delayed hit"
    assert d["lightning_legs"] == 40, "the attack that landed it gets the credit"
    assert summary["dealt"] == 40                                   # conserved


def test_an_attack_keeps_its_own_window_so_combos_are_credited():
    # cl.hp (20) then c.mk links for 30 while he is in hit-stun -- c.mk is an ATTACK, so it KEEPS its 30 (not zeroed)
    rows, _ = SE.round_evidence(0, "chunli", "ryu", [
        _dec("cl.hp", 100), _dec("c.mk", 80), _dec("block_high", 50)], replay=None)
    d = _dealt(rows)
    assert d["cl.hp"] == 20 and d["c.mk"] == 30                     # both attacks keep their damage


def test_delayed_hit_skips_an_intervening_block_to_find_the_attack():
    # legs -> block -> block(drain 40) -> s.mk: the 40 skips BOTH blocks back to the legs (the unknown-sink fix)
    rows, _ = SE.round_evidence(0, "chunli", "ryu", [
        _dec("lightning_legs", 100), _dec("block_high", 100), _dec("walk_back", 100), _dec("s.mk", 60)], replay=None)
    d = _dealt(rows)
    assert d["lightning_legs"] == 40 and d["block_high"] == 0 and d["walk_back"] == 0


def test_no_earlier_attack_leaves_the_delayed_hit_in_place():
    rows, summary = SE.round_evidence(0, "chunli", "ryu", [
        _dec("block_high", 100), _dec("s.mk", 50)], replay=None)
    assert _dealt(rows)["block_high"] == 50 and summary["dealt"] == 50   # conserved, not lost


# --------------------------------------------------------------------------- --no-score KO / last-window fix
def test_read_end_bars_takes_the_last_drawn_bars(tmp_path):
    rd = str(tmp_path)
    with open(__import__("os").path.join(rd, "reads.jsonl"), "w") as f:
        f.write('{"k": 10, "my_life": 90, "his_life": 80}\n')
        f.write('{"k": 20, "my_life": null, "his_life": null}\n')   # a read that saw no bars -> skipped
        f.write('{"k": 30, "my_life": 40, "his_life": 0}\n')        # the KO: his bar empty at round over
    assert SE.read_end_bars(rd) == {"my_life": 40, "his_life": 0}


def test_end_bars_credits_the_ko_and_fixes_the_result():
    # last decision BEFORE the final blow: she is behind (my 5, his 10). She then KOs (his bar -> 0).
    decs = [_dec("s.mk", 100), dict(_dec("s.mk", 10), moment={"my_life": 5, "his_life": 10, "doing": "standing",
                                                               "his_air": False, "his_label": "stand", "dx": 40})]
    # WITHOUT the end bars: last window is 0 and the result reads the pre-final-blow bars -> wrong "loss"
    rows0, s0 = SE.round_evidence(0, "chunli", "ryu", decs, replay=None)
    assert rows0[-1]["dealt"] == 0 and s0["result"] == "loss"
    # WITH the screen end bars (his_life 0): the KO (10 hp) is credited and the result is the real "win"
    rows1, s1 = SE.round_evidence(0, "chunli", "ryu", decs, replay=None, end_bars={"my_life": 5, "his_life": 0})
    assert rows1[-1]["dealt"] == 10 and s1["result"] == "win"
