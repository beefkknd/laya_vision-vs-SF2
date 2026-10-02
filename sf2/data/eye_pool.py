"""The eye's ONE frame pool (docs/eye_questions_v1.md, "Datasets v1"; scripts/build_eye_data.py): every image pair the
2P collection (rollouts/pairs2p, committed games) has on disk, with the facts every question is answered from, read
from the stored RAM at the DISPLAYED row t (lag 1: the images are the captures t - 3 and t + 1, i.e. rows t - 4, t).

An image pair = two images g<game>_k<k-4>.png and g<game>_k<k>.png of one game, both on disk (the movement samples,
the projectile samples and any other pair the saved images happen to form); t = k - 1.

Per pair (``facts``), pure, no files:
  fighters  per slot 1 / 2: character, side (smaller x at t = left; None on equal x), act (moving / attack / special
            from the movement rule with the pressed-word rule, sf2.data.pairs_labels.movement_pressed; None when the
            movement is unknown), mv10 (the grid movement), in_episode (rows t - 4 .. t one grid movement,
            pairs_labels.same_episode), air at t and at t - 4, pose (``pose_of``).
  shots     per slot: on / drawn at t and t - 4 (drawn = slot on and blink bit 0 of shot<s>_hide clear; None when the
            game has no blink bytes, games < 10), the kind of the flight the slot is in (``flight_kinds``).
  fire      q1 (v1.1, over BOTH shown frames t - 4 and t): "yes" = a fireball flight's slot on, drawn (blink bit
            clear) AND on the screen (``on_screen``; the camera is estimated from the fighters' x, ``camera_x``) in
            at least one frame; "no" = no fireball (nor unknown projectile) slot on in either frame (before spawn,
            after impact, a yoga flame: "no"); None = no blink bytes, an unknown projectile on, or a fireball on but
            drawn on the screen in neither frame (hidden by the blink in both, or off / at the edge of the screen).
  hard      the q1 "no" hard-negative tags (``hard_tags``).
  dist      q5: |x1 - x2| at t <= the calibrated poke band of all characters (lessons/perception_thresholds_v2.json
            poke_max["all"]) -> close, else far; None when an x is impossible.
"""
import collections
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

from . import movement_collect_io as MIO
from . import pairs_collect_io as IO
from . import pairs_labels as L
from . import pairs_moves as PM
from . import pairs_shots as S
from .perception import LAG, STAGE_X, UNKNOWN

GAP = 4
SCREEN_W, CAMERA_MIN, CAMERA_MAX, SCREEN_PAD = 256, 32, 224, 8
NEAR = 12                     # hard negatives: a fireball flight ended / starts within this many rows of t
ACT_ANSWERS = ("moving", "attack", "special")
IMG = re.compile(r"^g(\d{4})_k(\d{5})\.png$")


def act_of(mv10: str) -> Optional[str]:
    """The grid movement -> moving / attack / special (None for unknown)."""
    if mv10 == UNKNOWN:
        return None
    return mv10 if mv10 in ("attack", "special") else "moving"


def disk_pairs(names: Sequence[str]) -> Dict[int, List[int]]:
    """Image file names of one match -> {game: sorted displayed rows t} of every image pair on disk."""
    ks: Dict[int, set] = collections.defaultdict(set)
    for n in names:
        m = IMG.match(n)
        if m:
            ks[int(m.group(1))].add(int(m.group(2)))
    return {g: sorted(k - LAG for k in v if k - GAP in v) for g, v in sorted(ks.items())}


def flight_kinds(rows: Sequence[Dict[str, int]], words: Dict[int, List], chars: Dict[int, str]) -> Dict[int, List]:
    """Per slot, per row: None (slot off) or the kind of the flight the row is in: "fireball" (the slot's own player
    pressed its character's projectile word at the flight's first row), "yoga_flame" (Dhalsim pressed yoga flame), or
    "other"."""
    out: Dict[int, List] = {}
    for s in (1, 2):
        kinds: List = [None] * len(rows)
        a = None
        for u in range(len(rows) + 1):
            on = u < len(rows) and bool(rows[u]["shot%d" % s])
            if on and a is None:
                a = u
            if not on and a is not None:
                w = words[s][a]
                k = ("fireball" if S.is_projectile(chars[s], w) else
                     "yoga_flame" if chars[s] == "dhalsim" and w == "yoga_flame" else "other")
                for v in range(a, u):
                    kinds[v] = k
                a = None
        out[s] = kinds
    return out


def drawn(row: Dict[str, int], s: int) -> Optional[bool]:
    """Slot ``s`` on and drawn in this row; None when the game has no blink byte."""
    if not row["shot%d" % s]:
        return False
    if "shot%d_hide" % s not in row:
        return None
    return not row["shot%d_hide" % s] & 1


def pose_of(rows, t: int, s: int, char: str, word: Optional[str], mv10: str) -> str:
    """"projectile" while the fighter is in an attack state (0x0A / 0x0C) with its projectile word pressed (its
    throwing pose, with or without a projectile drawn), else its act answer ("unknown" when None)."""
    st = rows[t]["p%d_state" % s]
    if st in (L.ATTACK, L.SPECIAL) and S.is_projectile(char, word):
        return "projectile"
    return act_of(mv10) or UNKNOWN


def fighter(rows, t: int, s: int, char: str, words: List, classes: List) -> Dict:
    mv10 = L.grid_movement(rows, t, s, classes[t])
    return {"char": char, "side": S.side_of(rows[t], s), "mv10": mv10, "act": act_of(mv10),
            "in_episode": L.same_episode(rows, t, s, classes), "air": L.air(rows, t, s),
            "air_prev": L.air(rows, t - GAP, s), "pose": pose_of(rows, t, s, char, words[t], mv10),
            "pressed": words[t]}


def shot(rows, kinds: List, t: int, s: int) -> Dict:
    return {"on": bool(rows[t]["shot%d" % s]), "drawn": drawn(rows[t], s), "kind": kinds[t],
            "on_prev": bool(rows[t - GAP]["shot%d" % s]), "drawn_prev": drawn(rows[t - GAP], s),
            "kind_prev": kinds[t - GAP]}


def camera_x(row: Dict[str, int]) -> float:
    """The screen's left edge in world x, estimated from RAM (no scroll byte is recorded): centred on the fighters,
    clamped to the stage (fitted on 2,318 drawn hadoken pairs: 97.6% agree with the blue-pixel test of the frame)."""
    mid = (row["p1_x"] + row["p2_x"]) / 2.0
    return min(max(mid - SCREEN_W / 2, CAMERA_MIN), CAMERA_MAX)


def on_screen(row: Dict[str, int], s: int, pad: int = SCREEN_PAD) -> bool:
    """Slot ``s``'s projectile x inside the estimated screen, ``pad`` px in from both edges."""
    u = row["shot%d_x" % s] - camera_x(row)
    return pad <= u <= SCREEN_W - pad


def fire_frames(rows, kinds: Dict[int, List], t: int) -> List[str]:
    """The shown frames ("n-4" = row t - 4, "n" = row t) with a fireball drawn on the screen."""
    return [name for name, u in (("n-4", t - GAP), ("n", t))
            if any(kinds[s][u] == "fireball" and drawn(rows[u], s) and on_screen(rows[u], s) for s in (1, 2))]


def fire_why(rows, kinds: Dict[int, List], t: int) -> Tuple[Optional[str], str]:
    """(q1 answer, why), over BOTH shown frames (rows t - 4 and t; v1.1, owner review: a fireball blinking off in
    frame n is still a fireball coming): yes = a fireball drawn on the screen in at least one of them; no = no
    fireball or unknown projectile slot active in either (a yoga flame is no fireball); None when a frame has no
    blink byte, an unknown projectile is active, or a fireball is active but drawn on the screen in neither frame
    (blink-hidden in both: "hidden_both"; off the screen or at its edge: "off_screen")."""
    us = (t - GAP, t)
    if any("shot%d_hide" % s not in rows[u] for s in (1, 2) for u in us):
        return None, "no_blink_byte"
    if fire_frames(rows, kinds, t):
        return "yes", "drawn_on_screen"
    active = [(kinds[s][u], drawn(rows[u], s)) for s in (1, 2) for u in us if kinds[s][u] is not None]
    if any(k == "other" for k, _ in active):
        return None, "unknown_active"
    fire = [d for k, d in active if k == "fireball"]
    if fire:
        return None, "off_screen" if any(fire) else "hidden_both"
    return "no", "none_active"


def fire_answer(rows, kinds: Dict[int, List], t: int) -> Optional[str]:
    return fire_why(rows, kinds, t)[0]


def hard_tags(rows, kinds: Dict[int, List], t: int, poses: Dict[int, str], near: int = NEAR) -> List[str]:
    """Why a q1 "no" pair (no fireball active at t - 4 nor t) is hard: before_spawn / after_impact (a fireball flight
    starts within t + 1 .. t + near / ended within t - near .. t - 5), pose (a fighter in its throwing pose, e.g.
    Guile's windup swoosh), yoga_flame (a yoga flame drawn in either frame)."""
    tags = []
    for s in (1, 2):
        k = kinds[s]
        if any(k[u] == "fireball" for u in range(t + 1, min(len(rows), t + near + 1))):
            tags.append("before_spawn")
        if any(k[u] == "fireball" for u in range(max(0, t - near), t - GAP)):
            tags.append("after_impact")
        if any(k[u] == "yoga_flame" and drawn(rows[u], s) for u in (t - GAP, t)):
            tags.append("yoga_flame")
    if "projectile" in poses.values():
        tags.append("pose")
    return sorted(set(tags))


def dist_answer(row: Dict[str, int], band: int) -> Optional[str]:
    if not all(0 <= row[k] <= STAGE_X for k in ("p1_x", "p2_x")):
        return None
    return "close" if abs(row["p1_x"] - row["p2_x"]) <= band else "far"


def facts(rows, t: int, chars: Dict[int, str], words: Dict[int, List], classes: Dict[int, List],
          kinds: Dict[int, List], band: int) -> Dict:
    """Everything the questions need about the pair shown at rows (t - 4, t)."""
    fs = {s: fighter(rows, t, s, chars[s], words[s], classes[s]) for s in (1, 2)}
    poses = {s: fs[s]["pose"] for s in (1, 2)}
    fire, why = fire_why(rows, kinds, t)
    return {"t": t, "fighters": fs, "shots": {s: shot(rows, kinds[s], t, s) for s in (1, 2)}, "fire": fire,
            "fire_why": why, "fire_frames": fire_frames(rows, kinds, t) if fire == "yes" else [],
            "hard": hard_tags(rows, kinds, t, poses) if fire == "no" else [],
            "dist": dist_answer(rows[t], band), "gap": abs(rows[t]["p1_x"] - rows[t]["p2_x"]),
            "hide": all("shot%d_hide" % s in rows[t] for s in (1, 2))}


def game_facts(rows, moves: List, chars: Dict[int, str], ts: Sequence[int], band: int) -> List[Dict]:
    words = PM.pressed_words(moves, len(rows))
    classes = {s: [PM.pressed_class(chars[s], w) for w in words[s]] for s in (1, 2)}
    kinds = flight_kinds(rows, words, chars)
    return [facts(rows, t, chars, words, classes, kinds, band) for t in ts if GAP <= t < len(rows)]


def match_facts(root: str, pair_name: str, band: int) -> List[Dict]:
    """Every image pair of one match dir (committed games only), with its facts and its identity."""
    base = os.path.join(root, pair_name)
    a, b = pair_name.split("_vs_")
    chars = {1: a, 2: b}
    logs = {g["game"]: g.get("moves", []) for g in IO.committed(base)}
    out = []
    for game, ts in disk_pairs(os.listdir(os.path.join(base, "images"))).items():
        if game not in logs:
            continue
        rows = MIO.read_ram(os.path.join(base, "ram", "g%04d.json.gz" % game))
        for f in game_facts(rows, logs[game], chars, ts, band):
            t = f["t"]
            out.append(dict(f, pair_name=pair_name, pair=[a, b], game=game, k_prev=t - GAP + LAG, k_now=t + LAG,
                            images=["frames/%s/g%04d_k%05d.png" % (pair_name, game, k)
                                    for k in (t - GAP + LAG, t + LAG)]))
    return out


def _one(args):
    return match_facts(*args)


def pool(root: str, band: int, workers: int = 32) -> List[Dict]:
    """The whole pool, one process per match dir (sorted, deterministic)."""
    names = sorted(n for n in os.listdir(root) if "_vs_" in n and os.path.isdir(os.path.join(root, n)))
    jobs = [(root, n, band) for n in names]
    if workers <= 1:
        parts = [_one(j) for j in jobs]
    else:
        from multiprocessing import get_context
        with get_context("spawn").Pool(min(workers, len(jobs))) as p:
            parts = p.map(_one, jobs)
    return [f for part in parts for f in part]
