"""The U arm's perception labels (docs/prereg_u_perception.md, questions 1-7): pure functions of the RAM rows that
play_system1 --ram-log stores around each decision (sf2.system1.game_log.ram_entry). Training labels only: at play
time laya-vision sees the screen, never these rows. "me" is player 1 (System 1's fighter), "him" player 2 (the CPU).

The moment: the image captured at frame n shows the RAM of frame n - LAG (measured on the his_moves probe, the same
headless raw-capture path as play_system1: the HUD clock digits change one frame after the RAM clock in 98.9% of
12,166 image pairs (90-92% at lags 0 or 2), and his fireball sprite is on screen at none of 7 spawn frames and at all
7 one frame later). So every label is read at t = n - lag ("now") and t - 4 ("prev"); the lookahead counts from t.

Exact rules (thresholds ``th``: lessons/perception_thresholds_v1.json, scripts/calibrate_perception.py):
 1 range   gap = |his x - my x| at t: throw if <= throw_max[me], poke if <= poke_max[me], mid if < mid_max, else far
           (a character without its own band uses the pooled "all"). trend: gap(t) - gap(t - 4) < -eps closing,
           > eps opening, else steady.
 2 phase   being hit: his state 0x0E with a hit reaction (not 06 / 08), or thrown (0x14). blocking: state 0x08, or
           0x0E with a block reaction (06 / 08). attacking: state 0x0A / 0x0C; it is "recovering after a miss"
           (prereg amendment 2026-10-01) when (a) his attack episode (the consecutive attack rows around t) starts
           and ends inside the rows, (b) t lies in its second half (frame j of 0..L-1 with 2j >= L), and (c) I take
           no contact during the whole episode - on none of its rows do I ENTER hit / block stun or thrown
           (0x0E / 0x14) or lose life. Contact seen decides "attacking" even when the episode runs past the rows; no
           contact seen and a start or end outside the rows: unknown; the first half: attacking. Anything else
           (stand, crouch, jump, turn) is neutral.
 3 air     grounded: his y == GROUND_Y at t. Airborne: landing if he is on the ground at some frame in (t, t + k];
           else jumping at me if his x moved toward me over t - 4 -> t, else jumping away or straight up. Airborne
           with frame t + k missing and no landing seen: unknown. (A fighter knocked into the air is labelled by the
           same motion rule.)
 4 shot    his projectile is slot 2 (shot2 / shot2_x: player 2's slot, owner byte 1; Chun-Li's slot shot1 never turns
           on vs Ryu / Ken in the probe, Ryu's fireballs are shot2). None if shot2 is off at t, or if it moves away
           from me (t - 1 -> t; just spawned: it leaves his side toward me). Coming at me: near if
           |shot2_x - my x| < near_px, else far.
 5 can act free unless in state 0x0E / 0x14. dizzy: 0x0E with sub 0x08 and the dizzy flag set on the first row of
           that sub-0x08 run (the flag may clear before the stars end; a run starting before the rows counts only
           with the flag set on the first row there is). knocked down: the stun episode around t (consecutive
           0x0E / 0x14 rows) reaches sub 0x04 (the landing) somewhere. stunned: an episode that ends inside the rows
           without sub 0x04. An episode running past the rows without a sub 0x04 seen: unknown.
 6 corner  me (or him) if x is within corner_d (<=) of a wall (walls: the x the fighters pile up at, observed); me
           when both are.
 x         a position outside 0..STAGE_X (a negative x read as unsigned 16 bit, e.g. -167 -> 65369: one frame of
           Blanka in 2.28M collected, mid-attack in the corner) is impossible: every label that reads it (range,
           trend, air, projectile, corner) is unknown for that decision; the others stand. The RAM is recorded as
           read (unsigned), unchanged.
 7 bars    sf2.vocab.bar of hp at t: the DRAWN bar, which drains to life after a hit (prereg amendment 2026-10-01).
"""
from typing import Dict, List, Optional, Tuple

from ..emu.vs import GROUND_Y
from ..vocab import CHARACTERS, bar

Rows = List[Dict[str, int]]
UNKNOWN = "unknown"
LAG = 1
QUESTIONS: Dict[str, Tuple[str, ...]] = {
    "range": ("throw", "poke", "mid", "far"),
    "trend": ("closing", "steady", "opening"),
    "phase": ("neutral", "attacking", "recovering after a miss", "blocking", "being hit"),
    "air": ("grounded", "jumping at me", "jumping away or straight up", "landing"),
    "projectile": ("none", "far", "near"),
    "me_can_act": ("free", "stunned", "knocked down", "dizzy"),
    "him_can_act": ("free", "stunned", "knocked down", "dizzy"),
    "corner": ("me", "him", "neither"),
    "my_bar": ("full", "high", "half", "low"),
    "his_bar": ("full", "high", "half", "low"),
}
ATTACK = (0x0A, 0x0C)
GUARD, HIT, THROWN = 0x08, 0x0E, 0x14
STUN = (HIT, THROWN)
BLOCK_REACTS = (0x06, 0x08)
KNOCKDOWN_SUB, DIZZY_SUB = 0x04, 0x08
STAGE_X = 512             # world x: walls at 53 / 459 (symmetric about 256); projectiles seen 13-498


def _x_ok(r: Optional[Dict[str, int]], *keys: str) -> bool:
    return r is not None and all(0 <= r[k] <= STAGE_X for k in keys)


def decode(rec: Dict) -> Tuple[Rows, int]:
    """A ram.jsonl record -> (rows as dicts, index of the decision row)."""
    names = rec["names"]
    return [dict(zip(names, r)) for r in rec["rows"]], rec["n"]


def _row(rows: Rows, i: int) -> Optional[Dict[str, int]]:
    return rows[i] if 0 <= i < len(rows) else None


def _per_char(th: Dict, key: str, me: str) -> float:
    return th[key].get(me, th[key]["all"])


def _gap(r: Dict[str, int]) -> int:
    return abs(r["p2_x"] - r["p1_x"])


def _life(v: int) -> int:
    return v if v < 200 else 0


def _run(rows: Rows, t: int, pred) -> Tuple[Optional[int], Optional[int]]:
    """(first, last) index of the run of rows around t where ``pred`` holds; None for an end at the rows' edge (the
    run may go on outside them)."""
    s = t
    while s > 0 and pred(rows[s - 1]):
        s -= 1
    e = t
    while e < len(rows) - 1 and pred(rows[e + 1]):
        e += 1
    return (s if s > 0 else None), (e if e < len(rows) - 1 else None)


# ---- 1 ----------------------------------------------------------------------------------------------------------

def range_band(rows: Rows, t: int, th: Dict) -> str:
    if not _x_ok(rows[t], "p1_x", "p2_x"):
        return UNKNOWN
    me = CHARACTERS.get(rows[t]["p1_char"], "?")
    g = _gap(rows[t])
    if g <= _per_char(th, "throw_max", me):
        return "throw"
    if g <= _per_char(th, "poke_max", me):
        return "poke"
    return "mid" if g < th["mid_max"] else "far"


def trend(rows: Rows, t: int, th: Dict) -> str:
    p = _row(rows, t - 4)
    if not (_x_ok(p, "p1_x", "p2_x") and _x_ok(rows[t], "p1_x", "p2_x")):
        return UNKNOWN
    d = _gap(rows[t]) - _gap(p)
    return "closing" if d < -th["trend_eps"] else "opening" if d > th["trend_eps"] else "steady"


# ---- 2 ----------------------------------------------------------------------------------------------------------

def _contact_on_me(rows: Rows, s: int, e: int) -> bool:
    for i in range(s, e + 1):
        prev, r = rows[i - 1], rows[i]
        if r["p1_state"] in STUN and prev["p1_state"] not in STUN:
            return True
        if _life(r["p1_life"]) < _life(prev["p1_life"]):
            return True
    return False


def phase(rows: Rows, t: int, th: Dict) -> str:
    r = rows[t]
    st = r["p2_state"]
    if st == THROWN or (st == HIT and r["p2_react"] not in BLOCK_REACTS):
        return "being hit"
    if st in (GUARD, HIT):
        return "blocking"
    if st not in ATTACK:
        return "neutral"
    s, e = _run(rows, t, lambda x: x["p2_state"] in ATTACK)
    if _contact_on_me(rows, max(s or 1, 1), len(rows) - 1 if e is None else e):
        return "attacking"            # contact seen in the part of the episode the rows hold: decided
    if s is None or e is None:
        return UNKNOWN                # no contact seen, but it may have come before the rows or after them
    return "recovering after a miss" if 2 * (t - s) >= e - s + 1 else "attacking"     # second half: recovery


# ---- 3 ----------------------------------------------------------------------------------------------------------

def _toward(r: Dict[str, int], dx: int) -> bool:
    """A move of dx in world x goes toward me (player 1) from where he (player 2) is in row r."""
    return dx * (r["p1_x"] - r["p2_x"]) > 0


def air(rows: Rows, t: int, th: Dict) -> str:
    if rows[t]["p2_y"] == GROUND_Y:
        return "grounded"
    k = th["k"]
    if any(r["p2_y"] == GROUND_Y for r in rows[t + 1:t + k + 1]):
        return "landing"
    p = _row(rows, t - 4)
    if _row(rows, t + k) is None or not (_x_ok(p, "p2_x") and _x_ok(rows[t], "p1_x", "p2_x")):
        return UNKNOWN
    return "jumping at me" if _toward(rows[t], rows[t]["p2_x"] - p["p2_x"]) else "jumping away or straight up"


# ---- 4 ----------------------------------------------------------------------------------------------------------

def projectile(rows: Rows, t: int, th: Dict) -> str:
    r = rows[t]
    if not r["shot2"]:
        return "none"
    p = _row(rows, t - 1)
    if not _x_ok(r, "p1_x", "p2_x", "shot2_x") or (p is not None and p["shot2"] and not _x_ok(p, "shot2_x")):
        return UNKNOWN
    if p is not None and p["shot2"]:
        coming = (r["shot2_x"] - p["shot2_x"]) * (r["p1_x"] - r["shot2_x"]) >= 0     # held still on impact
    else:                                       # just out: it leaves his side, toward me
        coming = True
    if not coming:
        return "none"
    return "near" if abs(r["shot2_x"] - r["p1_x"]) < th["near_px"] else "far"


# ---- 5 ----------------------------------------------------------------------------------------------------------

def can_act(rows: Rows, t: int, who: int) -> str:
    st, sub, dz = "p%d_state" % who, "p%d_sub" % who, "p%d_dizzy" % who
    r = rows[t]
    if r[st] not in STUN:
        return "free"
    if r[st] == HIT and r[sub] == DIZZY_SUB:
        s, _ = _run(rows, t, lambda x: x[st] == HIT and x[sub] == DIZZY_SUB)
        if rows[s if s is not None else 0][dz]:
            return "dizzy"
    s, e = _run(rows, t, lambda x: x[st] in STUN)
    lo, hi = (s if s is not None else 0), (e if e is not None else len(rows) - 1)
    if any(x[sub] == KNOCKDOWN_SUB for x in rows[lo:hi + 1]):
        return "knocked down"
    return "stunned" if e is not None else UNKNOWN


# ---- 6, 7 -------------------------------------------------------------------------------------------------------

def corner(rows: Rows, t: int, th: Dict) -> str:
    lo, hi = th["walls"]
    if not _x_ok(rows[t], "p1_x", "p2_x"):
        return UNKNOWN
    dist = {p: min(rows[t]["p%d_x" % p] - lo, hi - rows[t]["p%d_x" % p]) for p in (1, 2)}
    if dist[1] <= th["corner_d"]:
        return "me"
    return "him" if dist[2] <= th["corner_d"] else "neither"


def labels(rows: Rows, n: int, th: Dict) -> Dict[str, str]:
    """Every question's answer for the image at frame n (``rows[n]`` is the decision row), each "unknown" where the
    rows do not decide it."""
    t = n - th.get("lag", LAG)
    if _row(rows, t) is None:
        return {q: UNKNOWN for q in QUESTIONS}
    r = rows[t]
    return {
        "range": range_band(rows, t, th), "trend": trend(rows, t, th), "phase": phase(rows, t, th),
        "air": air(rows, t, th), "projectile": projectile(rows, t, th),
        "me_can_act": can_act(rows, t, 1), "him_can_act": can_act(rows, t, 2), "corner": corner(rows, t, th),
        "my_bar": bar(r["p1_hp"]), "his_bar": bar(r["p2_hp"]),
    }


# ---- gate 2 helper: is the table's best move in a ranking's top 3 ------------------------------------------------

def table_best(table: Dict[Tuple, Dict[str, float]], cell: Tuple, moves: Optional[List[str]] = None) -> str:
    """The table's best move in ``cell`` exactly as the T arm plays it (System1._by_table via value_oracle.rank): the
    max over ``moves`` (default sf2.system1.system1.choices(me), whose order breaks ties), a move missing from the
    cell counting 0, values rounded to 3 places."""
    if moves is None:
        from ..system1.system1 import choices
        moves = choices(cell[0])
    row = table.get(tuple(cell), {})
    vals = {m: round(float(row.get(m, 0.0)), 3) for m in moves}
    return max(vals, key=vals.get)


def best_in_top3(table: Dict[Tuple, Dict[str, float]], cell: Tuple, ranking, moves: Optional[List[str]] = None
                 ) -> Optional[bool]:
    """Prereg gate 2 for one decision: is the table's best move (by RAM's cell) among the first 3 of ``ranking`` (a
    list, best first, or {move: score}, higher first; ties keep the given order)? None when the table's best is
    walking in (those decisions are reported separately: walking in is always offered)."""
    best = table_best(table, cell, moves)
    if best == FORWARD:
        return None
    order = list(ranking) if not isinstance(ranking, dict) else sorted(ranking, key=lambda m: -ranking[m])
    return best in order[:3]


# ---- 8: the soft target ------------------------------------------------------------------------------------------
Q8_ANSWERS = ("likely works", "may work", "likely fails")
Q8_MARGIN = 3.0          # hp over walking in for "likely works" (prereg)
FORWARD = "forward"


def _resampled_means(rng, clusters: List, picks) -> "np.ndarray":
    """Per resample r: the shrunk mean (sf2.data.value_oracle: n / (n + SHRINK)) of the decisions of the opponents
    picked in row r of ``picks``, each pick resampled with replacement on its own (an opponent picked twice gets two
    independent resamples); no decisions -> 0, as the table's empty cell."""
    import numpy as np

    from .value_oracle import SHRINK

    sums = np.zeros(picks.shape)
    sizes = np.array([len(c) for c in clusters])[picks]
    for o, vals in enumerate(clusters):
        mask = picks == o
        if len(vals) and mask.any():
            draws = rng.integers(0, len(vals), (int(mask.sum()), len(vals)))
            sums[mask] = np.asarray(vals)[draws].sum(axis=1)
    return sums.sum(axis=1) / np.maximum(sizes.sum(axis=1) + SHRINK * (sizes.sum(axis=1) > 0), 1)


def _q8_one(rng, move: List[List[float]], fwd: List[List[float]], resamples: int, margin: float) -> Dict[str, float]:
    picks = rng.integers(0, len(move), (resamples, len(move)))
    diff = _resampled_means(rng, move, picks) - _resampled_means(rng, fwd, picks)
    works = int((diff >= margin).sum())
    may = int(((diff > 0) & (diff < margin)).sum())
    return {"likely works": works / resamples, "may work": may / resamples,
            "likely fails": (resamples - works - may) / resamples}


def q8_targets(entries, resamples: int = 200, seed: int = 0, margin: float = Q8_MARGIN,
               moves: Optional[Dict[str, List[str]]] = None) -> Dict[Tuple, Dict[str, Dict]]:
    """Per (table cell, move): {"p": share of resamples per Q8_ANSWERS, "n": decisions, "n_by_opp": {opp: n},
    "mean": raw mean net, "forward_mean": raw mean net of walking in}. The decisions are the table's
    (sf2.data.value_oracle.build: explored, training-split games, not Chun-Li vs Guile). One resample: draw the cell's
    opponents with replacement (a cluster bootstrap: opponents who disagree make the answer spread), then each drawn
    opponent's decisions with replacement, for the move and for walking in alike; compare the shrunk means as the
    table does. ``moves``: per character, the moves to answer for (default: every move it tried anywhere); walking
    in is the baseline and gets no target of its own. Seeded per (cell, move): reproducible."""
    import zlib

    import numpy as np

    from .value_oracle import HELDOUT, _cell_of
    from .vs_sweep import TEST_INDEX

    if resamples <= 0:
        raise ValueError("resamples must be positive, got %r" % resamples)
    nets: Dict[Tuple, Dict[str, Dict[str, List[float]]]] = {}
    tried: Dict[str, set] = {}
    for e in entries:
        if not e.get("explored") or e["game"] % 10 in TEST_INDEX or (e["me"], e["opp"]) == HELDOUT:
            continue
        nets.setdefault(_cell_of(e), {}).setdefault(e["action"], {}).setdefault(e["opp"], []).append(
            float(e["dealt"] - e["taken"]))
        tried.setdefault(e["me"], set()).add(e["action"])
    out: Dict[Tuple, Dict[str, Dict]] = {}
    for c in sorted(nets):
        by_move = nets[c]
        opps = sorted({o for per in by_move.values() for o in per})
        fwd = [by_move.get(FORWARD, {}).get(o, []) for o in opps]
        names = (moves or {}).get(c[0], sorted(tried.get(c[0], ())))
        row: Dict[str, Dict] = {}
        for m in names:
            if m == FORWARD:
                continue
            mv = [by_move.get(m, {}).get(o, []) for o in opps]
            rng = np.random.default_rng([seed, zlib.crc32(repr((c, m)).encode())])
            flat, flat_f = [v for x in mv for v in x], [v for x in fwd for v in x]
            row[m] = {"p": _q8_one(rng, mv, fwd, resamples, margin), "n": len(flat),
                      "n_by_opp": {o: len(x) for o, x in zip(opps, mv) if x},
                      "mean": round(sum(flat) / len(flat), 3) if flat else None,
                      "forward_mean": round(sum(flat_f) / len(flat_f), 3) if flat_f else None}
        out[c] = row
    return out
