"""VS BATTLE: Python plays both controllers. Power-on to round 1 of a 2-player fight, range setups, and a recorder
that runs one scripted exchange (attacker + defender inputs) and returns every frame of both fighters' RAM.

Menu route (SF2 World Warrior USA, from reset, observed on the ROM 2026-09-27): title, START, cursor down to
V.S. BATTLE, START; player select with 1P's cursor on Ryu and 2P's on Ken; both jab; HANDICAP / STAGE SELECT
(defaults: equal handicap, the stage of player 1), START; round 1 at clock 99 with x 208 (1P) vs 304 (2P).

Controller 2 needs ``--snes.port2.type=SnesController`` on the Mesen command line (sf2/headless.py adds it).
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from .ram import Var

# Both fighters' structs: player 1 at 0x0C00 / 0x0D00, player 2 at 0x0E00 / 0x0F00 (the +0x200 stride of the RAM
# map in ram_maps/sf2_snes.txt). "special" and "facing" are only verified for player 1; player 2's are the same
# offsets and are recorded as data, not relied on by any check.
_FIELDS = [("hp", 0x0D12, 2), ("life", 0x0C35, 1), ("x", 0x0D18, 2), ("y", 0x0D10, 2), ("state", 0x0C03, 1),
           ("sub", 0x0C04, 1), ("react", 0x0C4A, 1), ("dizzy", 0x0C89, 1), ("special", 0x0D80, 1),
           ("facing", 0x0CF4, 1), ("char", 0x0CD1, 1)]
VARS: List[Var] = [Var("p%d_%s" % (p, n), a + 0x200 * (p - 1), s, False) for p in (1, 2) for n, a, s in _FIELDS]
VARS += [Var("timer", 0x1AC8, 1, False), Var("result", 0x1ACF, 1, False),
         Var("shot1", 0x1000, 1, False), Var("shot1_x", 0x1007, 2, False),
         Var("shot2", 0x1050, 1, False), Var("shot2_x", 0x1057, 2, False)]
NAMES = [v.name for v in VARS]

IDS = {"ryu": 0, "honda": 1, "blanka": 2, "guile": 3, "ken": 4, "chunli": 5, "zangief": 6, "dhalsim": 7}
CURSOR_START = {1: IDS["ryu"], 2: IDS["ken"]}
GROUND_Y = 192
START_X = (208, 304)
FULL_HP = 176
MENU: List[Tuple[str, int]] = [("-", 900), ("start", 2), ("-", 60), ("down", 2), ("-", 10), ("start", 2), ("-", 150)]
PICK_WAIT, HANDICAP_WAIT = 150, 150   # frames after the jabs before START on the handicap screen, and after it
MAX_WAIT = 3000


def _frames(steps: Sequence[Tuple[str, int]]) -> List[List[str]]:
    return [[] if b == "-" else b.split("+") for b, n in steps for _ in range(n)]


def rows_of(obs) -> List[Dict[str, int]]:
    return [dict(zip(NAMES, r)) for r in obs.rams]


def _walk_cursor(bridge, player: int, target: int) -> None:
    """Move one player's select cursor to ``target`` (grid: top row 0-3, bottom 4-7), checked against RAM."""
    key = "p%d_char" % player
    stuck = False
    for _ in range(12):
        cur = rows_of(bridge.run([[]]))[-1][key]
        if cur == target:
            return
        # Across first: a cursor cannot step onto the other player's cursor (1P on Ryu can't go down onto 2P's
        # Ken), so the vertical move comes last, and after a press that did not move, the other axis is tried.
        across = cur % 4 != target % 4
        vertical = (cur < 4) != (target < 4)
        if vertical and (not across or stuck):
            press = "down" if cur < 4 else "up"
        else:
            press = "right" if cur % 4 < target % 4 else "left"
        pad = [[press]] * 2 + [[]] * 10
        after = rows_of(bridge.run(pad if player == 1 else [[]] * 12, p2=None if player == 1 else pad))[-1][key]
        stuck = after == cur
    raise RuntimeError("player %d cursor never reached %d" % (player, target))


def _controllable(bridge) -> bool:
    """True when holding toward each other moves both fighters further than idling does (state restored)."""
    at = bridge.save_state()
    idle = rows_of(bridge.run([[]] * 3, p2=[[]] * 3))[-1]
    bridge.load_state(at)
    walk = rows_of(bridge.run([["right"]] * 3, p2=[["left"]] * 3))[-1]
    bridge.load_state(at)
    return walk["p1_x"] != idle["p1_x"] and walk["p2_x"] != idle["p2_x"]


def boot_vs(bridge, p1: str, p2: str) -> bytes:
    """Reset, go to VS BATTLE, pick ``p1`` / ``p2``, and return the savestate of the first frame both answer the
    stick in round 1. Replaces the bridge's VARS with VARS."""
    bridge.set_vars(VARS)
    bridge.reset()
    bridge.run(_frames(MENU))
    # a cursor cannot step onto the other's: when 1P's pick is where 2P's cursor starts (Ken), 2P moves first
    order = (2, 1) if IDS[p1] == CURSOR_START[2] else (1, 2)
    for player in order:
        _walk_cursor(bridge, player, IDS[p1 if player == 1 else p2])
    jab = [["y"]] * 2 + [[]] * PICK_WAIT
    bridge.run(jab, p2=jab)
    bridge.run(_frames([("start", 2), ("-", HANDICAP_WAIT)]))
    for _ in range(0, MAX_WAIT, 10):
        r = rows_of(bridge.run([[]] * 10))[-1]
        laid_out = (r["timer"] == 0x99 and (r["p1_x"], r["p2_x"]) == START_X and r["p1_hp"] == r["p2_hp"] == FULL_HP)
        if laid_out:
            break
    else:
        raise RuntimeError("no round 1 after the handicap screen")
    for _ in range(MAX_WAIT):
        if _controllable(bridge):
            return bridge.save_state()
        bridge.run([[]])
    raise RuntimeError("fighters never became controllable")


FINE = 12   # px from the target gap where gap_state switches from both walking to one fighter stepping
STALL = 16  # frames without the gap changing that end the walk (both against the walls / the camera limit)


def gap_state(bridge, start: bytes, gap: int) -> Tuple[bytes, int]:
    """From ``start`` (player 1 left, player 2 right) both walk toward, or away from, each other until the gap in
    world x reaches ``gap``, then settle 20 idle frames. Returns the savestate and the gap actually reached (walls
    and the camera's ~212 px limit cap it)."""
    bridge.load_state(start)
    r = rows_of(bridge.run([[]]))[-1]
    closer = abs(r["p2_x"] - r["p1_x"]) > gap
    p1, p2 = (["right"], ["left"]) if closer else (["left"], ["right"])
    stalled, solo = 0, 1        # frames the gap has not changed (a walk takes a frame or two to start; walls)
    for _ in range(600):
        now = abs(r["p2_x"] - r["p1_x"])
        if (now <= gap if closer else now >= gap) or stalled > STALL:
            break
        # both walk while far off; the last FINE px one fighter at a time (player 1, or player 2 once player 1 is
        # stuck on a wall), a frame per step, for exact gaps
        both = abs(now - gap) > FINE
        at = bridge.save_state()
        pads = ([p1], [p2]) if both else (([p1], [[]]) if solo == 1 else ([[]], [p2]))
        r = rows_of(bridge.run(pads[0], p2=pads[1]))[-1]
        after = abs(r["p2_x"] - r["p1_x"])
        if not both and abs(after - gap) > abs(now - gap) and (after <= gap if closer else after >= gap):
            bridge.load_state(at)     # this step overshoots further than staying: keep the closer frame
            r = rows_of(bridge.run([[]]))[-1]
            break
        stalled = stalled + 1 if after == now else 0
        if not both and stalled == STALL // 2:
            solo = 2
    r = rows_of(bridge.run([[]] * 20, p2=[[]] * 20))[-1]
    return bridge.save_state(), abs(r["p2_x"] - r["p1_x"])


# ---------------------------------------------------------------------------------------------------- recording
Step = Tuple  # (tokens, n) fixed frames | ("until", cond_name, tokens, max_frames)
Cond = Callable[[Dict[str, int], Dict[str, int]], bool]  # (row now, row at move start) with a_/d_ keys


@dataclass
class Take:
    rows: List[Dict[str, int]]           # RAM before frame 0..n, raw p1_/p2_ names
    p1: List[List[str]]                  # buttons held, per frame
    p2: List[List[str]]
    images: Dict[int, object]            # frame -> RGB array


def view(row: Dict[str, int], attacker: int) -> Dict[str, int]:
    """p1_/p2_ keys -> a_ (attacker) / d_ (defender)."""
    a, d = "p%d_" % attacker, "p%d_" % (3 - attacker)
    out = dict(row, attacker=attacker)
    for k, v in row.items():
        if k.startswith(a):
            out["a_" + k[3:]] = v
        elif k.startswith(d):
            out["d_" + k[3:]] = v
    return out


def physical(tokens: Sequence[str], facing_right: bool, pad: Dict[str, str]) -> List[str]:
    fwd, back = ("right", "left") if facing_right else ("left", "right")
    table = {"U": "up", "D": "down", "F": fwd, "B": back}
    return [table.get(t) or pad[t] for t in tokens]


SETTLE = 10  # neutral frames in a row (both standing or crouching on the ground, no projectile) that end a take


def settled(r: Dict[str, int]) -> bool:
    return (r["p1_state"] in (0, 2) and r["p2_state"] in (0, 2) and r["p1_y"] == GROUND_Y == r["p2_y"]
            and r["shot1"] == 0 == r["shot2"])


def record(bridge, attacker: int, a_steps: Sequence[Step], d_steps: Sequence[Step], tail: int,
           conds: Dict[str, Cond], pad: Dict[str, str], every: int = 2, shots: Optional[Set[int]] = None,
           d_hold: Tuple[str, ...] = (), track: Tuple[int, ...] = ()) -> Take:
    """Play one exchange from the loaded state. Each side's steps run in lockstep, one frame per bridge RUN (so an
    "until" step can react to RAM); directions are relative to where the opponent is at the start. After both
    sides' inputs, idle until SETTLE neutral frames in a row or ``tail`` frames. Screenshots every ``every`` frames
    (0: none), or exactly at the frame indices in ``shots``. ``d_hold``: tokens the defender holds once its steps
    are used up (a posture such as crouch-block that must last the whole take). ``track``: players whose F / B are
    resolved every frame from where the other fighter is NOW (a block holds away from him even after he crosses over),
    instead of once at the start."""
    snap = (lambda k: k in shots) if shots is not None else (lambda k: bool(every) and k % every == 0)
    obs0 = bridge.run([], caps=[0] if snap(0) else [])
    first = rows_of(obs0)
    rows, p1_in, p2_in, images = [first[0]], [[]], [[]], {0: obs0.images[0]} if 0 in obs0.images else {}
    right = {attacker: first[0]["p%d_x" % attacker] < first[0]["p%d_x" % (3 - attacker)]}
    right[3 - attacker] = not right[attacker]
    start = view(first[0], attacker)
    queues = {attacker: _expand_fixed(a_steps), 3 - attacker: _expand_fixed(d_steps)}
    idle_left, calm = tail, 0
    while any(queues.values()) or (idle_left > 0 and calm < SETTLE):
        now = view(rows[-1], attacker)
        busy = any(queues.values())
        frame = {p: _next_tokens(q, now, start, conds) for p, q in queues.items()}
        if not queues[3 - attacker]:
            frame[3 - attacker] = frame[3 - attacker] or d_hold
        if not busy:
            idle_left -= 1
        for p in track:
            right[p] = rows[-1]["p%d_x" % p] < rows[-1]["p%d_x" % (3 - p)]
        b1 = physical(frame[1], right[1], pad)
        b2 = physical(frame[2], right[2], pad)
        k = len(rows)
        obs = bridge.run([b1], caps=[1] if snap(k) else [], p2=[b2])
        rows.append(dict(zip(NAMES, obs.rams[-1])))
        calm = calm + 1 if not busy and settled(rows[-1]) else 0
        p1_in.append(b1)
        p2_in.append(b2)
        if 1 in obs.images:
            images[k] = obs.images[1]
        if len(rows) > 2000:
            raise RuntimeError("exchange never ended")
    return Take(rows, p1_in, p2_in, images)


def _expand_fixed(steps: Sequence[Step]) -> List[list]:
    """Queue of per-frame items: tokens tuples, or mutable ["until", cond, tokens, frames_left]."""
    q: List[list] = []
    for s in steps:
        if s[0] == "until":
            q.append(["until", s[1], tuple(s[2]), s[3]])
        else:
            q.extend([["fixed", tuple(s[0])]] * s[1])
    return q


def _next_tokens(q: List[list], now, start, conds) -> Tuple[str, ...]:
    while q:
        head = q[0]
        if head[0] == "fixed":
            q.pop(0)
            return head[1]
        if conds[head[1]](now, start) or head[3] <= 0:
            q.pop(0)
            continue
        q[0] = ["until", head[1], head[2], head[3] - 1]
        return head[2]
    return ()
