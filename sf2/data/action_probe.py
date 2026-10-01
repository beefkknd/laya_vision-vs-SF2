"""The action-ID probe (docs/prereg_movement_data.md, last section): which RAM byte names the move a fighter is doing.
Pure logic over recorded struct bytes (scripts/probe_action_ram.py, scripts/probe_action_moves.py); no emulator.

Struct rows are uint8 arrays [rows, 0x200] of one fighter: player 1 = WRAM 0x0C00-0x0DFF, player 2 = 0x0E00-0x0FFF
(offsets below are relative to the fighter's base, the same for both: verified, see the probe report).

Finding (2026-10-01): offset 0x3E (player 1 0x0C3E, player 2 0x0E3E) is the ROM's per-character ATTACK ID: 0 when
no attack box is out, else a small number per move (Chun-Li: close / far standing normals 1-12, crouching 19-22, jump
normals 23-27 and 32-33, Spinning Bird Kick 45/46/49, Lightning Legs 50-53). It is set a few frames into the move (the
startup frames read 0) and cleared before the recovery ends; multi-hit moves step through several IDs (Spinning Bird
Kick 46-49-45, Tatsumaki 32-33). So the move of an attack episode is its FIRST non-zero attack ID, and the episode is the
run of rows with one ``state`` (offset 0x03). A few attacks put out no box (Hadoken's fireball is its own object,
throws, attacks cut off in their startup): their run has no ID and is classed by MOVE_CLASS / the projectile slot.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

P1_BASE, P2_BASE, STRIDE = 0x0C00, 0x0E00, 0x200
STATE, ATTACK_ID, MOVE_CLASS = 0x03, 0x3E, 0xBA
# Set from the move's 2nd row and held through its state run (0xFF when none): coarse for normals (Chun-Li's
# lp / c.lk 0x10, mp-hk 0x03, c.mk 0x05, sweep / c.hp 0x16) but one value per special (Chun-Li Spinning Bird Kick 0x09,
# Lightning Legs 0x0C; Ken Tatsumaki 0x09, Shoryuken 0x0A). Names a special that never put a box out.
SPECIAL_CLASS = 0x49
ATTACK_STATES = (0x0A, 0x0C)            # normal attack / special (player-input specials; the CPU's are 0x0A)
JUMP_STATE = 0x04
THROW_CLASSES = (0x06, 0x0C)            # MOVE_CLASS of a throw (Ken / Ryu / Zangief 0x06, Chun-Li 0x0C)
CPU_SPECIAL_CLASS = 0x08                # MOVE_CLASS of a CPU special (Hadoken, Sonic Boom, ...)


def address(player: int, offset: int) -> int:
    if player not in (1, 2) or not 0 <= offset < STRIDE:
        raise ValueError("player %r offset %r" % (player, offset))
    return P1_BASE + STRIDE * (player - 1) + offset


@dataclass(frozen=True)
class Episode:
    start: int                 # first row
    end: int                   # one past the last row
    state: int
    ids: Tuple[int, ...]       # the non-zero attack IDs in order, consecutive repeats collapsed

    @property
    def attack_id(self) -> int:
        """The move: the first non-zero attack ID (0: no attack box in the episode)."""
        return self.ids[0] if self.ids else 0

    def __len__(self) -> int:
        return self.end - self.start


def runs(values: Sequence[int]) -> List[Tuple[int, int]]:
    """[start, end) of each run of equal values."""
    out, n = [], len(values)
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[j] == values[i]:
            j += 1
        out.append((i, j))
        i = j
    return out


def collapse(seq: Iterable[int]) -> Tuple[int, ...]:
    out: List[int] = []
    for v in seq:
        if v and (not out or out[-1] != v):
            out.append(int(v))
    return tuple(out)


def episodes(struct: np.ndarray, min_len: int = 1) -> List[Episode]:
    """One fighter's rows -> one Episode per run of equal ``state``."""
    if struct.ndim != 2 or struct.shape[1] <= MOVE_CLASS:
        raise ValueError("struct rows must be [rows, >= 0x%X] bytes, got %r" % (MOVE_CLASS + 1, struct.shape))
    st = [int(x) for x in struct[:, STATE]]
    return [Episode(a, b, st[a], collapse(struct[a:b, ATTACK_ID])) for a, b in runs(st) if b - a >= min_len]


def _mode(values) -> Optional[int]:
    c = Counter(int(x) for x in values).most_common(1)
    return c[0][0] if c else None


def no_box_kind(struct: np.ndarray, ep: Episode, own_shot: Optional[Sequence[int]] = None) -> str:
    """What an attack episode without an attack ID is: "throw", "projectile" (its own shot slot active),
    "special_<SPECIAL_CLASS hex>" (a player special, state 0x0C, hit or stopped before its box came out), "cut" (none
    of these: an attack stopped in its startup)."""
    if _mode(struct[ep.start + 1:ep.end, MOVE_CLASS]) in THROW_CLASSES:
        return "throw"
    if own_shot is not None and any(own_shot[ep.start:ep.end]):
        return "projectile"
    sc = _mode(struct[ep.start + 1:ep.end, SPECIAL_CLASS])
    if ep.state == 0x0C and sc is not None and sc != 0xFF:
        return "special_%02x" % sc
    return "cut"


def purity(eps: Sequence[Episode]) -> Dict[str, float]:
    """Over attack-state episodes: the share with an ID, and of those the share whose IDs are one value (the rest step
    through several, e.g. a multi-hit special)."""
    att = [e for e in eps if e.state in ATTACK_STATES]
    with_id = [e for e in att if e.ids]
    single = [e for e in with_id if len(set(e.ids)) == 1]
    return {"episodes": len(att), "with_id": len(with_id) / len(att) if att else 0.0,
            "single_id": len(single) / len(with_id) if with_id else 0.0}


def id_counts(eps: Sequence[Episode], states: Sequence[int] = ATTACK_STATES + (JUMP_STATE,)) -> Counter:
    return Counter(e.attack_id for e in eps if e.state in states and e.attack_id)


def move_table(takes: Sequence[Dict]) -> Tuple[Dict[str, Dict[str, List[int]]], Dict[int, List[str]]]:
    """Pressed takes ({"action", "gap", "struct"}) -> (move -> {range: first IDs of its attack episodes}, ID -> the
    "<move>@<range>" that produced it). Range: "close" (< 100 px) or "far"."""
    by_move: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))
    by_id: Dict[int, List[str]] = defaultdict(list)
    for t in takes:
        rng = "close" if t["gap"] < 100 else "far"
        firsts = [e.attack_id for e in episodes(t["struct"]) if e.attack_id]
        for i in firsts:
            if i not in by_move[t["action"]][rng]:
                by_move[t["action"]][rng].append(i)
            tag = "%s@%s" % (t["action"], rng)
            if tag not in by_id[i]:
                by_id[i].append(tag)
    return {m: dict(v) for m, v in by_move.items()}, dict(by_id)
