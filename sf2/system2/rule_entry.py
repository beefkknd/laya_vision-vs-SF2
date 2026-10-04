"""Build the registry entries text-laya plays from raw advice-grammar lines.

A candidate/incumbent Playbook is a list of advice lines ('use more throw up close'); to run it they
become VERIFIED registry entries - the --carry file play_loop_screen loads. Shared by the outcome loop
and scripts/measure_rule.py so the carry shape lives in one place. An unfollowable line (no move, or a
bare movement token like 'forward') raises - never silently fed to text-laya (component_boundaries).
"""
from sf2.system1.advice import parse

# text-laya polarity -> the registry claim kind
POL2KIND = {"soft": "use_more", "hard": "always", "neg": "avoid"}


def claim_of(line, moves):
    """A raw advice line -> a claim dict, via the same parser System 1 uses at play time."""
    lsn = parse(line, moves)
    if lsn is None or lsn.move is None or lsn.polarity == "none":
        raise ValueError("advice line not followable (no move / unfollowable token): %r -> %r" % (line, lsn))
    return {"kind": POL2KIND[lsn.polarity], "move": lsn.move, "range": lsn.where, "when": lsn.when, "view": None}


def entry(claim):
    """A claim dict -> a verified registry entry (the record shape play_loop_screen --carry loads)."""
    from sf2.system2 import lessons as L
    return {"claim": {k: claim.get(k) for k in ("kind", "move", "range", "when", "view")},
            "line": L.render(claim), "state": "verified", "why": "measured candidate", "since": -1,
            "evidence": {}, "qwen_why": ""}


def carry_entries(rules, moves):
    """A list of advice lines -> a list of verified registry entries, in order."""
    return [entry(claim_of(r, moves)) for r in rules]


# A blank playbook keeps BLOCK as the no-rule default (defend is already implicit), so a blank start only
# needs ONE attacking foothold so she is not a pure turtle from frame 1 (owner 2026-10-03).
DEFAULT_ATTACK = "use more s.mk when he stands"   # range-agnostic: fires close, mid OR far (not a mid-only turtle)


def default_kit(moves):
    """The minimal starting playbook for a BLANK book: one grounded attacking rule. Defend/block is
    already the no-rule default, so no defensive rule is seeded."""
    return carry_entries([DEFAULT_ATTACK], moves)


def candidate_rules(incumbent_rules, claims, moves):
    """Apply the Coach's kept claims to the incumbent playbook -> the candidate's rule lines. Each claim
    (already stance-filtered by coach_filter) renders to a line and is appended unless the incumbent
    already carries the same rule - compared in CANONICAL form (book lines use aliases like 'throw',
    render uses 'throw_F+hp'), so an equivalent rule is not duplicated. Incumbent lines are kept verbatim;
    order preserved. Pure (renders/parses via the shared helpers)."""
    from sf2.system2 import lessons as L
    canon = {L.render(claim_of(r, moves)) for r in incumbent_rules}
    out = list(incumbent_rules)
    for c in claims:
        line = L.render(c)
        if line not in canon:
            out.append(line)
            canon.add(line)
    return tuple(out)
