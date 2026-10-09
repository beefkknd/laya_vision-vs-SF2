"""PURE runtime lookup: opponent catalog sprite-key -> (LIMB, ZONE).

``attack_class(opponent, facts.sprite)`` enriches the value table's split label with a defender-relevant read of
the opponent's CURRENT pose:
    LIMB in {hand, leg, none}      hand = punch move, leg = kick move, none = startup / ambiguous
    ZONE in {high, mid, low, none} high = block standing, low = block crouching, mid = either, none = ambiguous

No I/O beyond loading the immutable table JSON once, on first use. Nothing is mutated.

The live wiring (value_table / screen_words) is FLAG-GATED by ``SF2_LIMB_KEY`` (``limb_key_on``): default OFF keeps
the value-table key byte-identical to the coarse ``his_label`` path; ON splits attack cells by ``fine_class``.

STATUS (owner 2026-10-09): the fine limb*zone split is OFF BY DEFAULT and stays off for all go-forward training.
It was built to test whether a defender-relevant attack-pose key beats one coarse ``attack`` label; the study
(docs/attack_key_coarse_vs_fine.md) found it does NOT pay for its complexity -- the best ACTION flips across
limb*zone in ~1% of decisions, and even the one cell where the fine value looked large (close|jumping|hand_high,
double_lariat +30 vs Ryu) the COARSE cell already picks the same action. So range|posture|behaviour-label carries
the decision; limb*zone does not. The code is retained (gated, experimental) only so the analysis stays
reproducible -- do NOT set SF2_LIMB_KEY in a production/forward run. The behavioural ``_maybe_split`` (value_table,
his_label) is a DIFFERENT mechanism and is kept on.
"""
import json
import os
import sys
from typing import Dict, Optional, Tuple

_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "opp_limb_table.json")
HAND, LEG = "hand", "leg"
HIGH, MID, LOW = "high", "mid", "low"
NONE = "none"
_UNKNOWN: Tuple[str, str] = (NONE, NONE)

_CACHE: Dict[str, Dict[str, Tuple[str, str]]] = {}
_WARNED = False


def _table() -> Dict[str, Dict[str, Tuple[str, str]]]:
    """Load the immutable lookup once; shared read-only view (callers must not mutate it)."""
    if not _CACHE:
        with open(_TABLE_PATH) as f:
            data = json.load(f)
        for char, by_key in data["table"].items():
            _CACHE[char] = {k: (v[0], v[1]) for k, v in by_key.items()}
    return _CACHE


def attack_class(opponent: str, sprite_key: str) -> Tuple[str, str]:
    """(limb, zone) for an opponent catalog sprite key.

    ``sprite_key`` is the live ``FighterFacts.sprite`` value, a catalog key "<char>/<hash>" (a bare "<hash>" is
    also accepted and resolved under ``opponent``). Any unknown opponent or key -> ("none", "none").
    """
    if not sprite_key:
        return _UNKNOWN
    per_opp = _table().get(opponent)
    if per_opp is None:
        return _UNKNOWN
    if sprite_key in per_opp:
        return per_opp[sprite_key]
    full = "%s/%s" % (opponent, sprite_key) if "/" not in sprite_key else sprite_key
    return per_opp.get(full, _UNKNOWN)


def limb_of(opponent: str, sprite_key: str) -> str:
    """Convenience: just the LIMB of ``attack_class``."""
    return attack_class(opponent, sprite_key)[0]


def zone_of(opponent: str, sprite_key: str) -> str:
    """Convenience: just the ZONE of ``attack_class``."""
    return attack_class(opponent, sprite_key)[1]


def fine_class(opponent: str, sprite_key: str) -> Optional[str]:
    """The fine split label "<limb>_<zone>" (e.g. "leg_high", "hand_mid") for an ATTACK frame, else None.

    None whenever either dimension is "none" (startup / ambiguous / non-attack) -- the caller then falls back to the
    coarse ``his_label``. This is the ONLY place the collapse rule lives.
    """
    limb, zone = attack_class(opponent, sprite_key)
    if limb == NONE or zone == NONE:
        return None
    return "%s_%s" % (limb, zone)


def limb_key_on() -> bool:
    """The master flag. ``SF2_LIMB_KEY`` unset / "" / "0" / "false" -> OFF (default). Anything else -> ON.

    One helper, read everywhere the enrichment branches (screen_words.moment, value_table.decider / credit), so the
    flag is never scattered. When ON it warns ONCE on stderr: the fine split is experimental and off by default
    (see this module's STATUS note / docs/attack_key_coarse_vs_fine.md), so a run can never enter it silently.
    """
    on = os.environ.get("SF2_LIMB_KEY", "").strip().lower() not in ("", "0", "false", "no", "off")
    if on:
        global _WARNED
        if not _WARNED:
            _WARNED = True
            print("[opp_limb] SF2_LIMB_KEY is ON: EXPERIMENTAL fine limb*zone attack split enabled "
                  "(off by default; low-ROI per docs/attack_key_coarse_vs_fine.md). Unset it for coarse `attack`.",
                  file=sys.stderr)
    return on


def limb_cells():
    """The TARGETED allowlist of base cells (``range|doing|fireball``) permitted to split by the fine (limb,zone)
    label, from ``SF2_LIMB_CELLS`` (comma-separated, e.g. "close|attacking|0"). Unset -> None = every attack cell
    may go fine (the original global behaviour). Set -> only the listed base cells go fine; all others stay coarse.
    This is how the detector's split_allowlist is enforced: split ONLY the cells proven contradictory."""
    raw = os.environ.get("SF2_LIMB_CELLS", "").strip()
    return frozenset(c.strip() for c in raw.split(",") if c.strip()) if raw else None
