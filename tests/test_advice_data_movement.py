"""Invariant: the shared advice dataset (scripts/build_advice_data.py, advice_v5) TEACHES text-laya to FOLLOW movement
advice. advice_v4 did not - walk_* was the correct answer in ~0.3% of move rows and offered in ~0.9% - so the move
model never learned to walk and ignored any approach/retreat rule in play (the turtle root cause, 2026-10-03). This
test pins the "move" follow-case: movement must be a real, offered, followed answer, WITHOUT regressing the block
default (condition_off -> block) or leaking a metadata shortcut. Seen RED on the v4 data (0.3%)."""
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(HERE, "scripts") not in sys.path:
    sys.path.insert(0, os.path.join(HERE, "scripts"))
import build_advice_data as B  # noqa: E402

SPLITS = B.build(per_char=200, seed=0)          # deterministic
TRAIN = SPLITS["train"]
ALL = [r for rs in SPLITS.values() for r in rs]


def _is_move(a):
    return "walk_" in a or "jump_" in a


def test_movement_is_a_followed_answer():
    """Movement is the correct move-round answer at a healthy rate (v4: 0.3%)."""
    mv = [r for r in TRAIN if r["round"] == "move"]
    moved = sum(1 for r in mv if any(_is_move(a) for a in r["answers"]))
    walked = sum(1 for r in mv if any("walk_" in a for a in r["answers"]))
    assert moved / len(mv) >= 0.08, "movement answer-rate too low: %.3f" % (moved / len(mv))
    assert walked / len(mv) >= 0.05, "walk answer-rate too low: %.3f" % (walked / len(mv))


def test_movement_is_offered_and_move_is_a_category_answer():
    mv = [r for r in TRAIN if r["round"] == "move"]
    offered = sum(1 for r in mv if any(_is_move(k) for k in r["question"]["criteria"]))
    assert offered / len(mv) >= 0.08, "movement offered too rarely: %.3f" % (offered / len(mv))
    cat = [r for r in TRAIN if r["round"] == "cat"]
    move_ans = sum(1 for r in cat if "move" in r["answers"])
    assert move_ans / len(cat) >= 0.08, "'move' category answer-rate too low: %.3f" % (move_ans / len(cat))


def test_move_case_is_actually_followed_not_defaulted():
    """A generated move-case advice line APPLIES, so the label must be soft/hard (followed), never default (block)."""
    mc = [r for r in ALL if r["case"] == "move"]
    assert mc, "no move cases generated"
    follow = sum(1 for r in mc if r["rule"] in ("soft", "hard"))
    assert follow == len(mc), "move case leaked to default: %d/%d followed" % (follow, len(mc))


def test_block_default_preserved():
    """The spam fix: a condition-OFF offense line still falls through to the block default."""
    co = [r for r in ALL if r["case"] == "condition_off"]
    assert co
    block = sum(1 for r in co if r["rule"] == "default")
    assert block == len(co), "condition_off no longer -> block: %d/%d" % (block, len(co))


def test_no_metadata_shortcut_from_movement():
    """Adding movement must not let the move round be read off metadata alone (meta <= chance + 0.05)."""
    _cc, _cm, move_chance, move_meta = B.shortcut_scores(SPLITS)
    assert move_meta <= move_chance + 0.05, "move-round metadata shortcut: chance %.3f meta %.3f" % (move_chance, move_meta)
