"""ScreenFacts: the fixed record the screen reader returns for one frame (docs/laya_text_only_plan.md, "The rule: no RAM
in play"). Immutable; built only from the screen image (sf2.screen.reader)."""
from dataclasses import dataclass, field
from typing import Optional, Tuple

SIDES = ("left", "right")
ROUND_STATES = ("fighting", "over")


@dataclass(frozen=True)
class FighterFacts:
    side: str                       # "left" / "right": where the fighter is on the screen in this frame
    character: str                  # the round's locked character for this fighter
    found: bool                     # a sprite of that character was located at all
    x: Optional[int]                # screen x of the fighter's anchor (RAM x - camera x), px
    y: Optional[int]                # screen y of the sprite's bottom (feet), px
    in_air: Optional[bool]
    facing: Optional[str]           # "left" / "right": the drawn sprite's orientation
    sprite: Optional[str]           # catalog key ("<char>/<key>") of the best match
    action: Optional[str]           # the catalog's majority 7-answer label of that sprite; None when unknown
    confidence: float               # match score of the best sprite, 0..1
    unknown: bool                   # no sprite matched well enough (or the matched sprite has no label)
    box: Optional[Tuple[int, int, int, int]] = None     # x0, y0, x1, y1 (exclusive) of the matched sprite
    health: Optional[float] = None  # this fighter's HUD bar (player 1's bar is the left one), 0..1
    player: Optional[int] = None    # 1 / 2: the side the character started the round on (HUD bar, name)


@dataclass(frozen=True)
class ProjectileFacts:
    x: int                          # screen x of the projectile's centre
    y: int
    owner_side: Optional[str]       # "left" / "right" when the sprite belongs to one of the two fighters only
    sprite: str
    confidence: float


@dataclass(frozen=True)
class HudFacts:
    health: Tuple[Optional[float], Optional[float]]     # left bar (player 1's), right bar (player 2's): 0..1
    timer: Optional[int]                                # the clock read from the digits, 0-99; None if unread
    bar_empty: Tuple[bool, bool]


@dataclass(frozen=True)
class ScreenFacts:
    left: FighterFacts
    right: FighterFacts
    projectiles: Tuple[ProjectileFacts, ...]
    hud: HudFacts
    round_state: str                # "fighting" / "over"
    gap: Optional[int] = None       # right.x - left.x when both found
    notes: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def projectile_present(self) -> bool:
        return bool(self.projectiles)
