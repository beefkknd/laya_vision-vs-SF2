"""The fighters' pose (image) tables in ROM - pure (no emulator). Collection time only (scripts/probe_pose_table.py,
scripts/collect_full_sprites.py); nothing here runs in play.

What the probe found (2026-10-02, Street Fighter II (USA), LoROM, map mode 0x20, 2 MiB):
  * At round start the game copies each fighter's data from ROM into a per-player WRAM buffer: player 1 at 7E:6000,
    player 2 at 7E:C000 (the struct's 0x?C20 word holds that base). Layout, offsets from the buffer base:
      +0x0000  0x880 bytes, ROM 09:800D (the same block for every character)
      +0x0880  ANIM:  u16 offset table (anim i at ANIM + off[i]) then the animation scripts
      +0x1100  IMAGE: u16 offset table (pose i at IMAGE + off[i]) then the pose records (OAM piece lists)
  * An animation script is a run of 4-byte records [duration, flags, pose index, box id]; flags & 0x80 ends the
    script and is followed by an s16 jump relative to the jump word itself (end record + 4; the stance loops this way).
  * Fighter struct (player 1 0x0C00, player 2 0x0E00): 0x?C1A u16 = the current animation record (WRAM address),
    0x?C17 its countdown, 0x?C1E u16 = the current POSE record (WRAM address) = IMAGE + off[pose index].
    The pose pointer is what decides the drawn sprite (purity, scripts/probe_pose_table.py).
"""
import struct
from collections import Counter, defaultdict
from typing import Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

BUFFER = {1: 0x6000, 2: 0xC000}            # per-player WRAM buffer (bank 7E)
ANIM_OFF, IMAGE_OFF = 0x0880, 0x1100       # tables inside the buffer
BUFFER_LEN = 0x6000                        # player 1's buffer ends where player 2's starts
POSE_PTR, ANIM_PTR, ANIM_TIMER, BUFFER_PTR = 0x1E, 0x1A, 0x17, 0x20   # fighter-struct offsets (u16, u16, u8, u16)
RECORD = 4
END_FLAG = 0x80


def lorom_offset(bank: int, addr: int) -> int:
    """File offset of LoROM bank:addr (addr 0x8000-0xFFFF)."""
    if not 0x8000 <= addr <= 0xFFFF or not 0 <= bank <= 0xFF:
        raise ValueError("not a LoROM ROM address: %02x:%04x" % (bank, addr))
    return (bank & 0x7F) * 0x8000 + (addr - 0x8000)


def lorom_address(offset: int) -> Tuple[int, int]:
    if offset < 0:
        raise ValueError("negative offset")
    return offset // 0x8000, 0x8000 + offset % 0x8000


def u16(buf: bytes, i: int) -> int:
    return struct.unpack_from("<H", buf, i)[0]


def offset_table(buf: bytes, base: int, limit: int) -> List[int]:
    """A u16 offset table at ``base`` (offsets relative to ``base``): the entries up to the first record (the smallest
    non-zero offset marks the table's end). 0 = an empty slot (kept, so indices hold). Every other offset must land
    inside ``base .. limit``."""
    first = u16(buf, base)
    if first < 2 or first % 2:
        raise ValueError("no offset table at %#x (first offset %#x)" % (base, first))
    offs, end = [], first
    while 2 * len(offs) < end:
        o = u16(buf, base + 2 * len(offs))
        offs.append(o)
        if o:
            end = min(end, o)
    bad = [o for o in offs if o and not 2 * len(offs) <= o < limit - base]
    if bad:
        raise ValueError("offset table at %#x: %d offsets outside the table (%s..)" % (base, len(bad), hex(bad[0])))
    return offs


def image_table(buf: bytes, player: int = 1) -> List[Optional[int]]:
    """WRAM addresses of every pose record of the fighter whose buffer ``buf`` (a whole 128 KiB WRAM dump) holds."""
    base = BUFFER[player] + IMAGE_OFF
    return [base + o if o else None for o in offset_table(buf, base, BUFFER[player] + BUFFER_LEN)]


def anim_table(buf: bytes, player: int = 1) -> List[Optional[int]]:
    base = BUFFER[player] + ANIM_OFF
    return [base + o if o else None for o in offset_table(buf, base, BUFFER[player] + IMAGE_OFF)]


def walk_anim(buf: bytes, start: int, limit: int, max_records: int = 256) -> List[Tuple[int, int, int, int]]:
    """(address, duration, flags, pose index) of each record of one script, up to and including its end record."""
    out, a = [], start
    for _ in range(max_records):
        if a + RECORD > limit:
            break
        dur, flags, pose = buf[a], buf[a + 1], buf[a + 2]
        out.append((a, dur, flags, pose))
        if flags & END_FLAG:
            break
        a += RECORD
    return out


def poses_used(buf: bytes, player: int = 1) -> Counter:
    """pose index -> how many animation records name it (over every script of the anim table)."""
    c: Counter = Counter()
    limit = BUFFER[player] + IMAGE_OFF
    for a in anim_table(buf, player):
        if a is None:
            continue
        for _, _, _, pose in walk_anim(buf, a, limit):
            c[pose] += 1
    return c


def find_in_rom(rom: bytes, chunk: bytes) -> List[int]:
    """Every file offset where ``chunk`` occurs (at most 8)."""
    out, i = [], rom.find(chunk)
    while i >= 0 and len(out) < 8:
        out.append(i)
        i = rom.find(chunk, i + 1)
    return out


def rom_segments(buf: bytes, rom: bytes, start: int, end: int, step: int = 64) -> List[Dict]:
    """Map WRAM start..end onto ROM in ``step`` blocks: runs of blocks found at one consistent ROM offset.
    [{wram: [a0, a1], rom: [bank, addr] | None}]; ambiguous blocks continue the current run when it fits."""
    segs: List[Dict] = []
    for a in range(start, end, step):
        hits = find_in_rom(rom, buf[a:a + step])
        cur = segs[-1] if segs else None
        if cur and cur["_off"] is not None and cur["_off"] + (a - cur["wram"][0]) in hits:
            cur["wram"][1] = a + step
            continue
        off = hits[0] if hits else None
        if cur and off is None and cur["_off"] is None:
            cur["wram"][1] = a + step
            continue
        segs.append({"wram": [a, a + step], "_off": off})
    return [{"wram": s["wram"], "rom": None if s["_off"] is None else list(lorom_address(s["_off"])),
             "rom_offset": s["_off"]} for s in segs]


def purity(values: Sequence[Hashable], keys: Sequence[Hashable]) -> Dict:
    """How well ``value -> key`` holds: the share of samples whose key is the majority key of their value, plus
    the reverse (key -> value) and the counts."""
    if len(values) != len(keys) or not values:
        raise ValueError("purity needs equal, non-empty sequences")
    fwd: Dict[Hashable, Counter] = defaultdict(Counter)
    rev: Dict[Hashable, Counter] = defaultdict(Counter)
    for v, k in zip(values, keys):
        fwd[v][k] += 1
        rev[k][v] += 1
    n = len(values)
    return {"n": n, "values": len(fwd), "keys": len(rev),
            "purity": sum(c.most_common(1)[0][1] for c in fwd.values()) / n,
            "reverse": sum(c.most_common(1)[0][1] for c in rev.values()) / n,
            "majority": {v: c.most_common(1)[0][0] for v, c in fwd.items()}}


def pose_samples(raw: Dict[int, Sequence[Sequence[int]]], frames: Iterable[Dict], chars: Dict[int, str],
                 lag: int = 1, offset: int = POSE_PTR) -> Tuple[List[Tuple[str, int]], List[str]]:
    """((char, pose pointer relative to the player's buffer), sprite key) per fighter frame record.
    ``raw[p][row]`` = the 128 bytes 0x?C00-0x?C7F of player p at that stream row; the sprite of record k is the
    one drawn for row k - lag (sf2.data.perception.LAG = 1)."""
    vals, keys = [], []
    for fr in frames:
        p = {"p1": 1, "p2": 2}.get(fr["group"])
        t = fr["k"] - lag
        if p is None or t < 0:
            continue
        r = raw[p][t]
        ptr = r[offset] | (r[offset + 1] << 8)
        base = r[BUFFER_PTR] | (r[BUFFER_PTR + 1] << 8)
        vals.append((chars[p], ptr - base))
        keys.append(fr["key"])
    return vals, keys


def pose_scripts(buf: bytes, player: int = 1) -> Dict[int, List[Tuple[int, int, int]]]:
    """pose index -> [(script start, record naming it, records before it)] over every script, shortest prefix
    first (the first record naming the pose in each script)."""
    out: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    limit = BUFFER[player] + IMAGE_OFF
    for a in sorted({a for a in anim_table(buf, player) if a is not None}):
        named = set()
        for n, (rec, _, _, pose) in enumerate(walk_anim(buf, a, limit)):
            if pose not in named:
                named.add(pose)
                out[pose].append((a, rec, n))
    return {p: sorted(v, key=lambda t: (t[2], t[0])) for p, v in out.items()}
