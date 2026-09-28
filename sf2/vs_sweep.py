"""Stage-1 data: a still opponent. Each fighter is placed at a gap, the opponent holds one posture, the fighter
presses one of 20 actions, and RAM says what happened: hit, whiff, blocked (or none for movement).

    20 actions x 3 distance ranges (sf2.ram.dist_bin: close < 55 <= mid < 120 <= far) x 10 gaps x 3 postures

Split by gap, so held-out examples are at distances the model never saw: per range 7 gaps train, 3 test.
Train is collected on ONE side only (the fighter on the left, facing right) and mirrored for the other facing;
test is real frames on both sides (the check that mirroring works).

Mirroring (sf2.frames.mirror_frame / ``mirror_record``): every model frame has its HUD blanked (sf2/frames.py),
so the mirror is a plain left-right flip of the whole frame; physical left/right buttons are swapped and the
side-dependent note fields (side, signed dx) flip. Action names are relative (forward = toward the opponent), so
they stay as they are.
"""
from typing import Dict, List, Sequence, Tuple

from .ram import CLOSE, MID
from .vs import GROUND_Y, Step
from .vs_moves import BLOCK_REACTS, GUARD, HIT, SPECIAL, THROWN, JUMP, ATTACK

LEAD = 8        # idle frames before the action; the images are frames LEAD - 4 ("a moment ago") and LEAD ("now")
PREV_GAP = 4
_T = 2          # frames a button is held, then released

MOVEMENT: Dict[str, Tuple[Step, ...]] = {
    "idle": (((), 4),), "forward": ((("F",), 4),), "back": ((("B",), 4),), "jump": ((("U",), 4),),
    "jump_forward": ((("U", "F"), 4),), "jump_back": ((("U", "B"), 4),), "crouch": ((("D",), 4),),
}
NORMALS: Dict[str, Tuple[Step, ...]] = {b: (((b,), _T), ((), _T)) for b in ("lp", "mp", "hp", "lk", "mk", "hk")}


def _crouch(b: str) -> Tuple[Step, ...]:
    return ((("D", b), _T), (("D",), _T))


SPECIALS: Dict[str, Dict[str, Tuple[Step, ...]]] = {
    # ROM-verified timings (laya_two_system sf2/actions.py; tests/test_rom_moves.py there)
    "ryu": {"c.lk": _crouch("lk"), "c.mk": _crouch("mk"), "sweep": _crouch("hk"),
            "throw": ((("F", "hp"), _T), (("F",), _T)),
            "hadoken": ((("D",), 2), (("D", "F"), 2), (("F", "hp"), 2), ((), 2)),
            "shoryuken": ((("F",), 2), (("D",), 2), (("D", "F", "hp"), 2), ((), 2)),
            "tatsumaki": ((("D",), 2), (("D", "B"), 2), (("B", "hk"), 2), ((), 2))},
    "chunli": {"c.lk": _crouch("lk"), "c.mk": _crouch("mk"), "sweep": _crouch("hk"), "c.hp": _crouch("hp"),
               "throw": ((("F", "hp"), _T), (("F",), _T)),
               "lightning_legs": ((("lk",), 1), ((), 1)) * 12,
               "spinning_bird_kick": ((("D",), 64), (("U", "hk"), 2), ((), 2))},
}
SPECIALS["ken"] = SPECIALS["ryu"]
_CHARGE = 64    # frames a charge is held (61 needed on the ROM for Guile, + a margin)
_MASH = ((("lp",), 1), ((), 1)) * 12      # ~5+ presses of one punch: Hundred Hand Slap, Electricity
_BASE = {"c.lk": _crouch("lk"), "c.mk": _crouch("mk"), "sweep": _crouch("hk"), "c.hp": _crouch("hp"),
         "throw": ((("F", "hp"), _T), (("F",), _T))}


def _back_charge(button: str) -> Tuple[Step, ...]:
    # down-back charges "back" too and does not walk away, so the gap being measured stays put
    return ((("D", "B"), _CHARGE), (("F", button), 2), ((), 2))


SPECIALS["guile"] = dict(_BASE, sonic_boom=_back_charge("hp"),
                         flash_kick=((("D",), _CHARGE), (("U", "hk"), 2), ((), 2)))
SPECIALS["honda"] = dict(_BASE, hundred_hand_slap=_MASH, sumo_headbutt=_back_charge("hp"))
SPECIALS["blanka"] = dict(_BASE, electricity=_MASH, rolling_attack=_back_charge("hp"))
SPECIALS["zangief"] = dict(_BASE, spinning_piledriver=((("F",), 2), (("D", "F"), 2), (("D",), 2), (("D", "B"), 2),
                                                       (("B",), 2), (("U", "lp"), 2), ((), 2)),  # ROM-verified
                           clothesline=((("lp", "mp", "hp"), 2), ((), 2)))
SPECIALS["dhalsim"] = dict(_BASE, yoga_fire=((("D",), 2), (("D", "F"), 2), (("F", "hp"), 2), ((), 2)),
                           yoga_flame=((("B",), 2), (("D", "B"), 2), (("D",), 2), (("D", "F"), 2), (("F", "hp"), 2),
                                       ((), 2)))
# Specials that must show state 0C to count as done. Not here: the clothesline (a 0A attack), the pile driver (out
# of grab range Zangief advances and grabs, still 0C, but checked by outcome), and the mash moves (Hundred Hand
# Slap, Electricity, Lightning Legs): up close the first jab of the mash hits and its hit stun stops the special from starting,
# which is what the ROM does; ``special_state`` records whether it started.
SPECIAL_STATE = {"hadoken", "shoryuken", "tatsumaki", "spinning_bird_kick", "sonic_boom",
                 "flash_kick", "sumo_headbutt", "rolling_attack", "yoga_fire", "yoga_flame"}


def static_actions(char: str) -> Dict[str, Tuple[Step, ...]]:
    """The 20 actions tried against the still opponent (their outcome is what they do to him)."""
    out = dict(MOVEMENT, **NORMALS, **SPECIALS[char])
    assert len(out) == 20, (char, len(out))
    return out


def actions(char: str) -> Dict[str, Tuple[Step, ...]]:
    """Everything the character can do: the 20 static actions and the two blocks (their outcome is what they save the
    character from; sf2/vs_defense.py)."""
    from .vs_defense import BLOCKS, block_steps
    return dict(static_actions(char), **{b: block_steps(b) for b in BLOCKS})


POSTURES: Dict[str, Tuple[str, ...]] = {"stand": (), "crouch": ("D",), "crouch_block": ("D", "B")}
# Stage 1 trains on what is visible: a crouch-blocking dummy looks exactly like a crouching one until an attack
# comes, so its "blocked" rows share inputs with "hit" rows (650 conflicting pairs, 79% of the failing combinations
# in the first eval). They stay collected (shards) for the moving-opponent stage, where the guard pose shows.
STAGE1_POSTURES = ("stand", "crouch")
RANGES = ("close", "mid", "far")
# Target gaps (world px) per range, 10 each; index 2, 5, 8 are held out for test. The narrowest the fighters get
# is ~20 px (they push each other), the widest ~206 (the camera).
GAPS: Dict[str, List[int]] = {
    "close": [22, 25, 28, 31, 34, 37, 40, 44, 48, 52],
    "mid": [58, 64, 70, 76, 82, 88, 95, 102, 109, 116],
    "far": [124, 132, 140, 148, 156, 164, 172, 180, 190, 200],
}
TEST_INDEX = (2, 5, 8)


def range_of(gap: int) -> str:
    return "close" if gap < CLOSE else "mid" if gap < MID else "far"


def split_of(index: int) -> str:
    return "test" if index in TEST_INDEX else "train"


def outcome(rows: Sequence[Dict[str, int]], action: str) -> Dict[str, object]:
    """rows: a_ = the fighter who acted, d_ = the still opponent, from the action's first frame on.
    hit: hit stun with a hit reaction, or thrown. blocked: block stun and no hit (chip damage still counts as
    blocked). whiff: an attack came out and touched nothing (a proximity guard pose is still a whiff; it is kept in
    ``guard_pose``). none: a movement action."""
    thrown = any(r["d_state"] == THROWN for r in rows)
    hit = thrown or any(r["d_state"] == HIT and r["d_react"] not in BLOCK_REACTS for r in rows)
    # blocked = block stun (contact). The guard pose alone (08) is not: holding back, the dummy takes it whenever
    # an attack is out nearby, even one that falls short (proximity guard).
    blocked = not hit and any(r["d_state"] == HIT and r["d_react"] in BLOCK_REACTS for r in rows)
    attacked = any(r["a_state"] in (ATTACK, SPECIAL) or (r["a_state"] == JUMP and r["a_sub"] == 0x06) for r in rows)
    if action in MOVEMENT:
        label = "none"
    else:
        label = "hit" if hit else "blocked" if blocked else "whiff"
    busy = [i for i, r in enumerate(rows) if r["a_state"] not in (0, 2) or r["a_y"] != GROUND_Y]
    lives = [r["d_life"] for r in rows]
    special = any(r["a_state"] == SPECIAL for r in rows)
    return {"outcome": label, "thrown": thrown, "special_state": special, "attacked": attacked, "guard_pose": any(r["d_state"] == GUARD for r in rows), "executed": (special if action in SPECIAL_STATE else attacked) or action in MOVEMENT,
            "damage": max(0, lives[0] - min(lives)), "busy_frames": (busy[-1] + 1) if busy else 0,
            "travel": (1 if rows[0]["a_x"] < rows[0]["d_x"] else -1) * (rows[-1]["a_x"] - rows[0]["a_x"])}


OUTCOMES = ["hit", "whiff", "blocked", "none", "got_hit"]      # new classes go at the end: labels keep their numbers
OUTCOME_CRITERIA = {
    "hit": "it connects: the opponent is hit or thrown",
    "whiff": "it comes out but touches nothing (falls short or goes over)",
    "blocked": "a guard stops the attack (block stun, at most chip damage)",
    "none": "nothing is hit either way",
    "got_hit": "the opponent's attack hits me (I lose health)",
}


def outcome_question(action: str) -> Dict:
    """The laya-vision choice question for one action. Byte-identical at train and play time: ask it through this
    function only. The label is OUTCOMES.index(outcome)."""
    return {"type": "choice", "instructions": "If you do %s now, what happens?" % action,
            "criteria": dict(OUTCOME_CRITERIA)}


def note(me: str, opp: str, r: Dict[str, int], side: str) -> str:
    """The RAM note, as sf2.ram.text_state plus the side fields mirroring must swap (side, signed dx)."""
    dx = r["d_x"] - r["a_x"]
    return ("me=%s opp=%s dist=%s side=%s dx=%+d my_hp=100 opp_hp=100 last=idle airborne=0 opp_airborne=0 "
            "opp_crouch=%d" % (me, opp, range_of(abs(dx)), side, dx, int(r["d_state"] == 0x02)))


_SWAP = {"left": "right", "right": "left"}


def mirror_record(rec: Dict) -> Dict:
    """The same example seen from the other side: side and dx flip, physical left/right buttons swap."""
    out = dict(rec)
    out["side"] = _SWAP[rec["side"]]
    if "collected_side" in rec:
        out["collected_side"] = _SWAP[rec["collected_side"]]
    out["facing"] = _SWAP[rec["facing"]]
    out["dx"] = -rec["dx"]
    out["buttons"] = [[_SWAP.get(b, b) for b in f] for f in rec["buttons"]]
    out["state_text"] = (rec["state_text"].replace("side=%s" % rec["side"], "side=%s" % out["side"])
                         .replace("dx=%+d" % rec["dx"], "dx=%+d" % out["dx"]))
    out["mirrored"] = True
    out["id"] = rec["id"] + "-m"
    return out
