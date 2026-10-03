"""Effective-coverage primitive for the outcome-driven loop. PURE: list of decision records -> floats.

A decision record (one frame) carries:
  - prompt_lines: the rules ROUTED IN (applicable) that frame; non-empty => a rule fired
  - follows_rule: did text-laya's chosen move match the rule (bool)

fire_rate = frac of frames that fired; follows = among FIRED frames, frac that were followed.
A frame where no rule applied is a non-fire and does not count toward follows (follows is undefined
there). This is the fire x follows part of the plan's "effective coverage = fire x follows x hp-delta".
"""


def coverage(decisions):
    """Return (fire_rate, follows) over a list of per-frame decision records."""
    n = len(decisions)
    if n == 0:
        return 0.0, 0.0
    fired = [d for d in decisions if d.get("prompt_lines")]
    nf = len(fired)
    fire_rate = nf / n
    follows = (sum(1 for d in fired if d.get("follows_rule")) / nf) if nf else 0.0
    return fire_rate, follows
