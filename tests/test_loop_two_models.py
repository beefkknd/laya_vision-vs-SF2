"""T5 (docs/plan_two_finetune_textlaya.md): loop_runner.two_stage_decide drives TWO trained checkpoints - round 1
(category) by ``cat_advisor``, round 2 (move) by ``move_advisor`` - and the model owns the default (no code forces
block). Block is only ever OFFERED: the block category on the round-1 menu, block_high always in the round-2 options.

Seen RED (demonstrated against the edited runner, each reverted after):
  * routing: if both rounds were asked of ``cat_advisor`` (the pre-split single-advisor flow), ``move`` records 0
    calls, so ``len(move.calls) == 1`` fails. Confirmed by pointing round 2 at ``cat_advisor``.
  * block-always-offered: with the old ``options = moves_in_stance(...) or [DEFAULT_MOVE]`` (block_high offered only
    when the stance list is empty), a non-block round-1 category leaves block_high OUT of the round-2 options, so
    ``DEFAULT_MOVE in move.calls[0]["options"]`` fails. Confirmed by dropping the always-append line.
  * no code floor: if any branch rewrote the pick to block when no rule applies, ``d["action"] == "walk_forward"``
    fails. There is no such branch; the model's max stands.

Mocks only - no MLX, no network, no emulator.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from looptools import FollowerLaya, make_moment                      # noqa: E402

from sf2.system1.action_menu import CATEGORY_ORDER, DEFAULT_MOVE     # noqa: E402
from sf2.system1.loop_runner import two_stage_decide                 # noqa: E402


class RecordingAdvisor:
    """A mock checkpoint: returns 1.0 on ``pick`` when offered (else the first option, so it is always valid), and
    records every (text, options, instructions) it was asked - so a test can prove which round reached it."""

    def __init__(self, pick: str):
        self.pick = pick
        self.calls = []

    def ask(self, text, question):
        opts = list(question["criteria"])
        self.calls.append({"text": text, "options": opts, "instructions": question["instructions"]})
        choice = self.pick if self.pick in opts else opts[0]
        return {o: (1.0 if o == choice else 0.0) for o in opts}


def test_round1_uses_cat_model_and_round2_uses_move_model():
    cat, move = RecordingAdvisor("kick"), RecordingAdvisor("cl.mk")
    m = make_moment(dx=36, doing="standing")                         # up close
    d = two_stage_decide(cat, move, "chunli", m, [])
    # each model asked exactly once, for its own round
    assert len(cat.calls) == 1 and len(move.calls) == 1
    # round 1 is the CATEGORY question (its options are the 7 categories); round 2 is the MOVE question (not categories)
    assert cat.calls[0]["options"] == CATEGORY_ORDER
    assert move.calls[0]["options"] != CATEGORY_ORDER and "cl.mk" in move.calls[0]["options"]
    # each model's pick stands, routed to its round
    assert d["category"] == "kick" and d["action"] == "cl.mk"


def test_no_rule_models_picks_stand_and_block_is_offered_in_both_rounds():
    # empty advice: no rule applies. The models pick freely; the code must NOT force block.
    cat, move = RecordingAdvisor("move"), RecordingAdvisor("walk_forward")   # a NON-block category + a NON-block move
    m = make_moment(dx=36, doing="standing")
    d = two_stage_decide(cat, move, "chunli", m, [])
    # block is OFFERED in both rounds (the safety nuance) but never forced
    assert "block" in cat.calls[0]["options"]                        # block category always on the round-1 menu
    assert DEFAULT_MOVE in move.calls[0]["options"]                  # block_high always added to the round-2 options
    # the models' picks STAND - no code floor rewrote them to block
    assert d["category"] == "move" and d["action"] == "walk_forward" and d["action"] != DEFAULT_MOVE


def test_special_then_move_flows_through_both_models():
    # advice names a special up close: round 1 -> the "special" category, round 2 -> the special move
    m = make_moment(dx=36, doing="standing")
    d = two_stage_decide(FollowerLaya(), FollowerLaya(), "chunli", m, ["always use lightning_legs up close"])
    assert d["category"] == "special" and d["action"] == "lightning_legs"
    assert d["rule"] == "hard" and d["follows_rule"]
