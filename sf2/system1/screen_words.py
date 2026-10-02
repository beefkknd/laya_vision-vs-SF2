"""Screen facts -> the T0 words (docs/laya_text_only_plan.md, build step 3: arm S0 = T0 with every fact from the screen).

The T0 path (sf2/system1/system1.py) turns one RAM row into: "can I act", the note the lookup table ranks
(sf2.data.vs_sweep.note, version 2 -> sf2.data.value_oracle.rank), text laya's situation (system1.situation) and the
facing the buttons are mirrored by. Here the same words come from ScreenFacts (sf2.screen.facts) only. Nothing in this
module reads RAM or imports a RAM row helper (tests/test_screen_play.py scans it); the words are pinned to the T0
functions on equivalent RAM rows by the same test.

The mapping table (one row per 7-answer sprite label, sf2.data.eye_v2.ACT2; the RAM state each label stands for in the
catalog: out/sprite_catalog cross-tab 2026-10-02):

    label           RAM state(s)          ground: opp_attacking / doing      in the air: opp_attacking / doing
    stand           00 stand, 02 crouch   0 / standing (crouching on a       0 / jumping
                                          crouch sprite: opp_crouch=1)
    walk            00                    0 / standing                       0 / jumping
    jump            04 (take-off rows)    0 / standing                       0 / jumping
    block           08 guard (and the     0 / standing                       0 / jumping
                    reader's unknown)
    attack          0A (ground), 04 (air) 1 / attacking                      0 / jumping   (a jump attack is state 04)
    special attack  0C, 0A                1 / attacking                      1 / jumping
    hit             0E hit_stun           0 / stunned                        0 / jumping

Crouch: the 7 answers fold crouch into "stand" (eye_v2.ACT2_OF); the sprite's own catalog label (its majority RAM
state, a fixed file labelled at collection time) says whether it is a crouch sprite.
Can I act (T0: p1_state in (stand, crouch) and on the ground): my label is stand or walk and I am not in the air.
Unknown sprite (the reader's ``unknown`` flag): the reader's default "block" (owner 2026-10-02, corrected) is used
as is - an unknown he gets T0's words for a guarding opponent (not attacking, not crouching, "standing"); an unknown
me is not stand / walk, so I cannot act and the loop waits 4 frames, as T0 does. The reader's UnknownLog saves the crop.
Health: the drawn bar fraction -> life points (fraction * 176, rounded) -> sf2.vocab.bar, as T0 does with RAM life.
"""
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional, Tuple

from ..screen.assets import CATALOG
from ..screen.facts import FighterFacts, ScreenFacts
from ..vocab import FULL_LIFE, bar, range_of

CAN_ACT = ("stand", "walk")
CROUCH_STATE = "02"
# label -> (opp_attacking on the ground, opp_attacking in the air, doing on the ground)
LABEL_WORDS: Dict[str, Tuple[int, int, str]] = {
    "stand": (0, 0, "standing"),
    "walk": (0, 0, "standing"),
    "jump": (0, 0, "standing"),
    "block": (0, 0, "standing"),
    "attack": (1, 0, "attacking"),
    "special attack": (1, 1, "attacking"),
    "hit": (0, 0, "stunned"),
}
AIR_DOING = "jumping"
DEFAULT_LABEL = "block"        # = sf2.screen.reader.DEFAULT_ACTION: no label from the reader (unknown)
# x when a fighter is not found in this frame and never was before (screen x of the round-start places)
START_X = {"left": 88, "right": 168}


@lru_cache(maxsize=2)
def crouch_sprites(catalog: str = CATALOG) -> frozenset:
    """Catalog keys ("<char>/<key>") whose majority RAM state is crouch (02)."""
    with open(catalog + "/catalog.json") as f:
        sprites = json.load(f)["sprites"]
    out = set()
    for ck, e in sprites.items():
        st = {k: v for k, v in e.get("labels", {}).get("state", {}).items() if k != "unknown"}
        if st and max(sorted(st), key=st.get) == CROUCH_STATE:
            out.add(ck)
    return frozenset(out)


@dataclass(frozen=True)
class Moment:
    """The decision facts T0 reads from RAM, here from the screen."""
    my_x: int
    his_x: int
    my_facing: Optional[str]       # drawn facing of my sprite ("left" / "right"); None when not found
    my_label: str
    my_air: bool
    his_label: str
    his_air: bool
    his_crouch: bool
    my_life: int                   # drawn bar * FULL_LIFE
    his_life: int
    filled: Tuple[str, ...] = ()   # facts not on the screen this frame, filled from the last moment / defaults

    @property
    def can_act(self) -> bool:
        return self.my_label in CAN_ACT and not self.my_air

    @property
    def side(self) -> str:
        """My side, as T0: "left" when my x is smaller, else "right"."""
        return "left" if self.my_x < self.his_x else "right"

    @property
    def dx(self) -> int:
        return self.his_x - self.my_x

    @property
    def his_attacking(self) -> int:
        ground, air, _ = LABEL_WORDS.get(self.his_label, LABEL_WORDS[DEFAULT_LABEL])
        return air if self.his_air else ground

    @property
    def doing(self) -> str:
        """sf2.system1.advice.opp_doing's word."""
        if self.his_air:
            return AIR_DOING
        if self.his_label == "stand" and self.his_crouch:
            return "crouching"
        return LABEL_WORDS.get(self.his_label, LABEL_WORDS[DEFAULT_LABEL])[2]

    def facing_right(self, movement: bool) -> bool:
        """Which way the stick is mirrored, as T0: walks by the x order, other moves by my sprite's drawn facing (RAM's
        facing byte on the T0 path); the x order when my sprite was not found."""
        if movement or self.my_facing is None:
            return self.my_x < self.his_x
        return self.my_facing == "right"


def life_of(fraction: Optional[float]) -> int:
    return FULL_LIFE if fraction is None else int(round(fraction * FULL_LIFE))


def players(facts: ScreenFacts) -> Tuple[FighterFacts, FighterFacts]:
    """(me = player 1, him = player 2)."""
    a, b = facts.left, facts.right
    if a.player == 2 or b.player == 1:
        a, b = b, a
    return a, b


def label(f: FighterFacts) -> str:
    """The fighter's 7-answer label as the reader gives it (its default "block" for an unknown sprite)."""
    return f.action or DEFAULT_LABEL


def moment(facts: ScreenFacts, last: Optional[Moment] = None) -> Moment:
    """The decision facts of one frame. A fighter not found keeps its last x (or its start place) and is listed in
    ``filled``; its label is the reader's default."""
    me, him = players(facts)
    filled = []

    def x_of(f: FighterFacts, prev: Optional[int], default: int, name: str) -> int:
        if f.found and f.x is not None:
            return int(f.x)
        filled.append(name)
        return prev if prev is not None else default

    my_x = x_of(me, last.my_x if last else None, START_X["left"], "my_x")
    his_x = x_of(him, last.his_x if last else None, START_X["right"], "his_x")
    hud = facts.hud.health
    my_frac = me.health if me.health is not None else hud[0]
    his_frac = him.health if him.health is not None else hud[1]
    for name, v in (("my_life", my_frac), ("his_life", his_frac)):
        if v is None:
            filled.append(name)
    return Moment(my_x, his_x, me.facing if me.found else None, label(me), bool(me.in_air),
                  label(him), bool(him.in_air), bool(not him.unknown and him.sprite in crouch_sprites()),
                  life_of(my_frac), life_of(his_frac), tuple(filled))


def note(me: str, m: Moment) -> str:
    """sf2.data.vs_sweep.note(..., version=2) in the same words (the lookup table's key)."""
    return ("me=%s dist=%s side=%s dx=%+d my_bar=%s opp_bar=%s opp_airborne=%d opp_crouch=%d opp_attacking=%d" % (
        me, range_of(abs(m.dx)), m.side, m.dx, bar(m.my_life), bar(m.his_life), int(m.his_air),
        int(m.his_crouch and m.his_label == "stand" and not m.his_air), m.his_attacking))


def situation(m: Moment) -> Tuple[str, str, str, str]:
    """sf2.system1.system1.situation in the same words: (range, what he is doing, my bar, his bar)."""
    return range_of(abs(m.dx)), m.doing, bar(m.my_life), bar(m.his_life)
