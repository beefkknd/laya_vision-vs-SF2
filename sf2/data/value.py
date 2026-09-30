"""Value labels for laya-vision (docs/plan_laya_vision_value.md): per move, how the exchange ends for me - the damage
I deal minus the damage I take from this decision to my next one - in five ordered buckets. The outcome question
(hit / whiff / ...) stays; the value question is asked next to it, on live rows only (a still dummy never punishes).
Also the note versions: v2 adds the opponent's general state opp_attacking (no move names: laya stays general)."""
from typing import Dict, Mapping

# net = dealt - taken (life points; a round starts at 176). A hit in SF2 takes 15-40, so a big swing is 30+.
BIG = 30
VALUE_BUCKETS = {         # ordered worst -> best; general words about the health bars, the same for every opponent
    "big_loss": "I lose a big chunk of my bar",
    "loss": "my bar drops more than theirs",
    "even": "neither bar changes, or both drop the same",
    "gain": "their bar drops more than mine",
    "big_gain": "they lose a big chunk of their bar",
}
# mean net inside each bucket over 62,868 no-advice decisions (rollouts/ab/*/*_none, 2026-09-30): the expected net of
# a predicted distribution weights these
BUCKET_NET = {"big_loss": -41.1, "loss": -22.0, "even": 0.0, "gain": 21.1, "big_gain": 47.0}

NOTE_VERSIONS = (1, 2)    # 1: runs/all8 and older (no version key in the checkpoint); 2: + opp_attacking


def value_bucket(net: int) -> str:
    if not isinstance(net, int) or isinstance(net, bool):
        raise TypeError("net must be an int of life points, got %r" % (net,))
    if net <= -BIG:
        return "big_loss"
    if net < 0:
        return "loss"
    if net == 0:
        return "even"
    return "gain" if net < BIG else "big_gain"


def value_question(action: str) -> Dict:
    """The laya-vision score question for one move. Byte-identical at train and play time: ask it through this
    function only. The label is list(VALUE_BUCKETS).index(bucket). laya renders a score option as "level i: <item>"
    of a list (a dict would show only its keys), so each item carries the bucket's name and words; its answers come
    back keyed "0".."4" (``value_probs``)."""
    return {"type": "score", "instructions": "If you do %s now, how does the exchange end for me?" % action,
            "criteria": ["%s: %s" % kv for kv in VALUE_BUCKETS.items()]}


def value_probs(answer_probs: Mapping[str, float]) -> Dict[str, float]:
    """laya's answer to ``value_question`` ({"0": p, ..., "4": p}) -> {bucket: p}."""
    names = list(VALUE_BUCKETS)
    if sorted(answer_probs) != [str(i) for i in range(len(names))]:
        raise ValueError("not a value answer: %r" % sorted(answer_probs))
    return {names[int(k)]: float(p) for k, p in answer_probs.items()}


def expected_net(probs: Mapping[str, float]) -> float:
    """Expected net (life points) of a predicted bucket distribution."""
    return float(sum(p * BUCKET_NET[b] for b, p in probs.items()))


def note_version(cfg: Mapping) -> int:
    """The note a checkpoint was trained on (its vlm_agent_config.json "note_version"; absent = 1)."""
    v = cfg.get("note_version", 1)
    if v not in NOTE_VERSIONS:
        raise ValueError("unknown note_version %r (known: %s)" % (v, NOTE_VERSIONS))
    return v
