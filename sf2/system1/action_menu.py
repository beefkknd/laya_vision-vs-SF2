"""The two-stage action menu for text laya (owner 2026-10-02): round 1 picks a CATEGORY, round 2 the MOVE within it,
the situation (stance / range) having already pruned which moves are available. Default when nothing applies: block
(hardcoded in the loop). This module is PURE DATA (move NAMES only, no RAM, no move mechanics) so the real play path
can import it; the mechanics for each name live in sf2.data.vs_moves (Chun-Li's move set).

Round-1 categories (7); round-2 is the listed moves filtered by the current stance/range at play time.
"""

# category -> every move name in it (names must match sf2.data.vs_moves.chunli()).
CATEGORIES = {
    "move":    ["walk_forward", "walk_back", "crouch", "jump_up", "jump_forward", "jump_back"],
    "punch":   ["s.lp", "cl.lp", "c.lp", "j.lp", "jf.lp",
                "s.mp", "cl.mp", "c.mp", "j.mp", "jf.mp",
                "s.hp", "cl.hp", "c.hp", "j.hp", "jf.hp"],
    "kick":    ["s.lk", "cl.lk", "c.lk", "j.lk", "jf.lk",
                "s.mk", "cl.mk", "c.mk", "j.mk", "jf.mk",
                "s.hk", "cl.hk", "c.hk", "j.hk", "jf.hk"],
    "block":   ["block_high", "block_low"],
    "throw":   ["throw_F+hp", "throw_F+mp"],
    "special": ["lightning_legs", "spinning_bird_kick"],
    "combo":   ["jf.hk_s.mp_s.hp", "jf.mk_legs"],
}

CATEGORY_ORDER = ["move", "punch", "kick", "block", "throw", "special", "combo"]

DEFAULT_MOVE = "block_high"  # hardcoded fallback when round 2 yields nothing usable (owner: "hard code to block")


def category_of(move: str) -> str:
    """The round-1 category a move name belongs to; raises if the name is unknown."""
    for cat, moves in CATEGORIES.items():
        if move in moves:
            return cat
    raise ValueError("move %r is in no category" % move)
