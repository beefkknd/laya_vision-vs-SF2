"""Measurements of one recorded exchange (rows with a_ = the fighter doing the move, d_ = the other; frame 0 is
before the first input). Frame counts are 60 Hz frames. Distances are world-x / y pixels."""
from typing import Dict, List, Optional

from .emu.vs import GROUND_Y
from .vs_moves import GUARD, HIT, THROWN, Rows, gap, life_drops, toward

NEUTRAL = (0x00, 0x02)


def _first(rows: Rows, pred) -> Optional[int]:
    return next((i for i, r in enumerate(rows) if pred(r)), None)


def _last(rows: Rows, pred) -> Optional[int]:
    return next((i for i in range(len(rows) - 1, -1, -1) if pred(rows[i])), None)


def _shot_speed(rows: Rows) -> Dict[str, Optional[float]]:
    for slot in ("shot1", "shot2"):
        xs = [r[slot + "_x"] for r in rows if r[slot]]
        if len(xs) > 4:
            steps = [abs(b - a) for a, b in zip(xs, xs[1:]) if abs(b - a) < 20]
            return {"shot_frames": len(xs), "shot_speed": round(sum(steps) / max(len(steps), 1), 2),
                    "shot_travel": abs(xs[-1] - xs[0])}
    return {"shot_frames": 0, "shot_speed": None, "shot_travel": 0}


def _startup(drops: List[int], contact: Optional[int]) -> Optional[int]:
    """Frames from the first input to the first touch: a life drop, or the defender entering hit / guard / thrown."""
    seen = ([drops[0]] if drops else []) + ([contact + 1] if contact is not None else [])
    return min(seen) if seen else None


def measure(rows: Rows) -> Dict[str, object]:
    """Startup (frames to the first hit / block contact), duration (frames until the attacker is neutral on the
    ground again), damage and hits, hit stun, knockdown, travel, height, air time, pushback, projectile speed."""
    busy = lambda r: r["a_state"] not in NEUTRAL or r["a_y"] != GROUND_Y  # noqa: E731
    drops = life_drops(rows)
    contact = _first(rows[1:], lambda r: r["d_state"] in (HIT, GUARD, THROWN))
    end = _last(rows, busy)
    air = [r for r in rows if r["a_y"] != GROUND_Y]
    out: Dict[str, object] = {
        "gap_start": gap(rows[0]),
        "gap_end": gap(rows[-1]),
        "startup": _startup(drops, contact),
        "duration": end if end is not None else 0,
        "hits": len(drops),
        "damage": sum(rows[i - 1]["d_life"] - rows[i]["d_life"] for i in drops),
        "hitstun": sum(1 for r in rows if r["d_state"] == HIT and r["d_react"] not in (0x06, 0x08)),
        "blockstun": sum(1 for r in rows if r["d_state"] == HIT and r["d_react"] in (0x06, 0x08)),
        "knockdown": any(r["d_state"] == HIT and r["d_sub"] == 0x04 for r in rows),
        "thrown": any(r["d_state"] == THROWN for r in rows),
        "travel": toward(rows),
        "max_forward": max((1 if rows[0]["a_x"] < rows[0]["d_x"] else -1) * (r["a_x"] - rows[0]["a_x"]) for r in rows),
        "height": GROUND_Y - min(r["a_y"] for r in rows),
        "airtime": len(air),
        "self_damage": rows[0]["a_life"] - min(r["a_life"] for r in rows),
        "frames": len(rows) - 1,
    }
    out.update(_shot_speed(rows))
    return out


def walk_speed(rows: Rows, frames: int) -> float:
    """px per frame over the frames the stick was held (the first few are the walk starting)."""
    xs = [r["a_x"] for r in rows[: frames + 1]]
    return round(abs(xs[-1] - xs[0]) / max(frames, 1), 3)


def reach(results: List[Dict[str, object]]) -> Dict[str, Optional[int]]:
    """From a gap sweep [{gap, connected}...]: the widest and narrowest gap that connected."""
    hit = [int(r["gap"]) for r in results if r["connected"]]
    return {"reach_max": max(hit) if hit else None, "reach_min": min(hit) if hit else None,
            "sweep_hits": len(hit), "sweep_tries": len(results)}
