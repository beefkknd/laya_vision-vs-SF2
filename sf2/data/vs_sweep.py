"""Stage-1 data: a still opponent. Each fighter is placed at a gap, the opponent holds one posture, the fighter
presses one of 20 actions, and RAM says what happened: hit, whiff, blocked (or none for movement).

    20 actions x 3 distance ranges (close < 55 <= mid < 120 <= far, sf2.emu.ram) x 10 gaps x 3 postures

Split by gap, so held-out examples are at distances the model never saw: per range 7 gaps train, 3 test.
Train is collected on ONE side only (the fighter on the left, facing right) and mirrored for the other facing;
test is real frames on both sides (the check that mirroring works).

Mirroring (sf2.data.frames.mirror_frame / ``mirror_record``): every model frame has its HUD blanked (sf2/data/frames.py),
so the mirror is a plain left-right flip of the whole frame; physical left/right buttons are swapped and the
side-dependent note fields (side, signed dx) flip. Action names are relative (forward = toward the opponent), so
they stay as they are.
"""
import re
from typing import Dict, List, Sequence, Tuple

from ..vocab import FULL_LIFE, bar, range_of
from ..emu.vs import GROUND_Y, Step
from .vs_moves import BLOCK_REACTS, GUARD, HIT, SPECIAL, THROWN, JUMP, ATTACK
# The RAM-free action vocabulary (movement, normals, specials, the 20 static actions) lives in sf2.data.actions_free
# so the screen-only play path can import it with no RAM; re-exported here, where the RAM half (notes, outcome
# labels, the two blocks whose steps are in sf2.data.vs_defense) is built on top of it.
from .actions_free import MOVEMENT, NORMALS, SPECIALS, _crouch, static_actions  # noqa: F401

LEAD = 8        # idle frames before the action; the images are frames LEAD - 4 ("a moment ago") and LEAD ("now")
PREV_GAP = 4

# Specials that must show state 0C to count as done. Not here: the clothesline (a 0A attack), the pile driver (out
# of grab range Zangief advances and grabs, still 0C, but checked by outcome), and the mash moves (Hundred Hand
# Slap, Electricity, Lightning Legs): up close the first jab of the mash hits and its hit stun stops the special from starting,
# which is what the ROM does; ``special_state`` records whether it started.
SPECIAL_STATE = {"hadoken", "shoryuken", "tatsumaki", "spinning_bird_kick", "sonic_boom",
                 "flash_kick", "sumo_headbutt", "rolling_attack", "yoga_fire", "yoga_flame"}


def actions(char: str) -> Dict[str, Tuple[Step, ...]]:
    """Everything the character can do: the 20 static actions and the two blocks (their outcome is what they save the
    character from; sf2/data/vs_defense.py)."""
    from .vs_defense import BLOCKS, block_steps
    return dict(static_actions(char), **{b: block_steps(b) for b in BLOCKS})


POSTURES: Dict[str, Tuple[str, ...]] = {"stand": (), "crouch": ("D",), "crouch_block": ("D", "B")}
# Stage 1 trains on what is visible: a crouch-blocking dummy looks exactly like a crouching one until an attack
# comes, so its "blocked" rows share inputs with "hit" rows (650 conflicting pairs, 79% of the failing combinations
# in the first eval). They stay collected (shards) for the moving-opponent stage, where the guard pose shows.
STAGE1_POSTURES = ("stand", "crouch")
# Target gaps (world px) per range, 10 each; index 2, 5, 8 are held out for test. The narrowest the fighters get
# is ~20 px (they push each other), the widest ~206 (the camera).
GAPS: Dict[str, List[int]] = {
    "close": [22, 25, 28, 31, 34, 37, 40, 44, 48, 52],
    "mid": [58, 64, 70, 76, 82, 88, 95, 102, 109, 116],
    "far": [124, 132, 140, 148, 156, 164, 172, 180, 190, 200],
}
TEST_INDEX = (2, 5, 8)


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
OUTCOME_CRITERIA = {        # in general words about the health bars, the same for every opponent
    "hit": "it connects: the opponent's bar drops",
    "whiff": "it misses: no bar changes",
    "blocked": "a guard stops the attack: the bars hold",
    "none": "nothing touches either fighter",
    "got_hit": "the opponent's attack connects: my bar drops",
}


def outcome_question(action: str) -> Dict:
    """The laya-vision choice question for one action. Byte-identical at train and play time: ask it through this
    function only. The label is OUTCOMES.index(outcome)."""
    return {"type": "choice", "instructions": "If you do %s now, what happens?" % action,
            "criteria": dict(OUTCOME_CRITERIA)}


def note(me: str, opp: str, r: Dict[str, int], side: str, version: int = 1) -> str:
    """The RAM note laya reads (``r``: a_ = me, d_ = the opponent). General and opponent-agnostic: no opponent name
    (knowledge of an opponent is System 2's short memory), the health bars as general levels, no constant fields.
    ``opp`` is accepted for the callers' sake and deliberately not written. ``version`` 2 (sf2.data.value) adds
    opp_attacking: he is in an attack or a special now (a general state, never which move)."""
    if version not in (1, 2):
        raise ValueError("unknown note version %r" % (version,))
    dx = r["d_x"] - r["a_x"]
    text = ("me=%s dist=%s side=%s dx=%+d my_bar=%s opp_bar=%s opp_airborne=%d opp_crouch=%d" % (
        me, range_of(abs(dx)), side, dx, bar(r.get("a_life", FULL_LIFE)), bar(r.get("d_life", FULL_LIFE)),
        int(r.get("d_y", GROUND_Y) != GROUND_Y), int(r["d_state"] == 0x02)))
    return text + (" opp_attacking=%d" % int(r["d_state"] in (0x0A, 0x0C)) if version == 2 else "")


def current_note(rec: Dict) -> str:
    """A stored row's note in today's format (rows written before a format change keep their raw text in the shards):
    no opponent name, bars as general levels, no constant fields; opp_airborne from what the row records."""
    old = rec["state_text"]
    fields = dict(re.findall(r"(\w+)=(\S+)", old))
    air = rec.get("opp_air")
    if air is None:
        air = rec.get("probe") == "jump_in" and rec.get("kind") == "defense"
    my = rec.get("my_life", FULL_LIFE)
    his = rec.get("opp_life", FULL_LIFE)
    return ("me=%s dist=%s side=%s dx=%s my_bar=%s opp_bar=%s opp_airborne=%d opp_crouch=%s" % (
        fields["me"], fields["dist"], fields["side"], fields["dx"], bar(my), bar(his), int(bool(air)),
        fields.get("opp_crouch", "0")))


def without_opp(text: str) -> str:
    """A note written before the opponent's name was dropped, as laya reads it now."""
    return re.sub(r" opp=[a-z]+(?= dist=)", "", text)


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
