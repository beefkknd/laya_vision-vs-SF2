"""RAM-free action vocabulary: the 20 still-opponent actions each character can press (movement, normals, specials),
as relative-token step scripts, with NO RAM (nothing here imports sf2.emu, sf2.data.vs_defense or any RAM-row
builder). The screen-only play runner imports this to know what it can do; sf2.data.vs_sweep re-imports it and keeps
the RAM half (the notes, the outcome labels, the two blocks whose steps live in sf2.data.vs_defense).

These are the action names the live play path uses (sf2.system1.system1.choices goes through
sf2.data.vs_sweep.actions). The step shapes are the same ``Step`` tuples as sf2.emu.vs, pinned by the ROM-verified
macros in laya_two_system.
"""
from typing import Dict, Tuple

Step = Tuple  # (tokens, n) | ("until", cond_name, tokens, max_frames) -- same shape as sf2.emu.vs.Step

_T = 2          # frames a button is held, then released
_CHARGE = 64    # frames a charge is held (61 needed on the ROM for Guile, + a margin)

MOVEMENT: Dict[str, Tuple[Step, ...]] = {
    "idle": (((), 4),), "forward": ((("F",), 4),), "back": ((("B",), 4),), "jump": ((("U",), 4),),
    "jump_forward": ((("U", "F"), 4),), "jump_back": ((("U", "B"), 4),), "crouch": ((("D",), 4),),
}
NORMALS: Dict[str, Tuple[Step, ...]] = {b: (((b,), _T), ((), _T)) for b in ("lp", "mp", "hp", "lk", "mk", "hk")}


def _crouch(b: str) -> Tuple[Step, ...]:
    return ((("D", b), _T), (("D",), _T))


def _back_charge(button: str) -> Tuple[Step, ...]:
    # down-back charges "back" too and does not walk away, so the gap being measured stays put
    return ((("D", "B"), _CHARGE), (("F", button), 2), ((), 2))


_MASH = ((("lp",), 1), ((), 1)) * 12      # ~5+ presses of one punch: Hundred Hand Slap, Electricity
_BASE = {"c.lk": _crouch("lk"), "c.mk": _crouch("mk"), "sweep": _crouch("hk"), "c.hp": _crouch("hp"),
         "throw": ((("F", "hp"), _T), (("F",), _T))}

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


def static_actions(char: str) -> Dict[str, Tuple[Step, ...]]:
    """The 20 actions tried against the still opponent (their outcome is what they do to him)."""
    out = dict(MOVEMENT, **NORMALS, **SPECIALS[char])
    assert len(out) == 20, (char, len(out))
    return out
