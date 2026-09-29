"""What the CPU opponent was doing, by move, for the game log (sf2.system1.game_log): pure, no I/O.

An attack episode is a run of consecutive frames with p2_state == 0x0A. Its move, from RAM (verified on 57,780
recorded frames vs Ryu, Ken and Honda, checked by screenshots; y smaller = higher, the ground is y 192):
    throw        Chun-Li is thrown (p1_state 0x14), or held off the ground in state 0 (Honda's bear hug)
    fireball     his projectile slot (shot2) goes 0 -> 1 inside the episode (~14 frames in)
    uppercut     starts on the ground and rises >= 28 px
    hurricane    held level 12-17 px above the ground for >= 16 frames
    slap         p2_sub reaches 6 or 8 (Honda's hundred hand slap)
    jump_attack  starts airborne
    normal       anything else
The first that matches, in that order, wins. p2_special is always 0 for the CPU and state 0x0C never occurs: unused.

A special looks like a normal for its first 14-41 frames and spans several decisions (~16 frames each), so one
decision's window is not enough. ``OppMoveTracker`` (one per round) follows the episodes across decisions and holds
each action entry back until every episode its window overlaps has ended; ``flush`` at the round's end classifies an
unfinished episode from what it has and releases the rest. Entries come out in order, as new dicts with ``opp_move``
(the highest-precedence move of the episodes overlapping the window, "none" without one) and ``opp_shot`` (shot2 at
the decision) added after the existing keys.
"""
from typing import Dict, List, Optional, Sequence

MOVES = ("throw", "fireball", "uppercut", "hurricane", "slap", "jump_attack", "normal")    # precedence order
NONE = "none"
ATTACK = 0x0A
GROUND_Y = 192
THROWN = 0x14
UPPERCUT_RISE = 28
HURRICANE_HEIGHT = (12, 17)
HURRICANE_FRAMES = 16
SLAP_SUBS = (6, 8)


def attacking(r: Dict) -> bool:
    return r["p2_state"] == ATTACK


def _thrown(rows: Sequence[Dict]) -> bool:
    return any(r["p1_state"] == THROWN or (r["p1_state"] == 0 and r["p1_y"] != GROUND_Y) for r in rows)


def _fireball(rows: Sequence[Dict], pre_shot: int) -> bool:
    shots = [pre_shot] + [r["shot2"] for r in rows]
    return any(b and not a for a, b in zip(shots, shots[1:]))


def _uppercut(rows: Sequence[Dict]) -> bool:
    return rows[0]["p2_y"] == GROUND_Y and GROUND_Y - min(r["p2_y"] for r in rows) >= UPPERCUT_RISE


def _hurricane(rows: Sequence[Dict]) -> bool:
    lo, hi = HURRICANE_HEIGHT
    run, last = 0, None
    for r in rows:
        y = r["p2_y"]
        run = (run + 1 if y == last else 1) if lo <= GROUND_Y - y <= hi else 0
        last = y
        if run >= HURRICANE_FRAMES:
            return True
    return False


def classify(rows: Sequence[Dict], pre_shot: int = 0) -> str:
    """The move of one attack episode (its rows, in order); ``pre_shot``: shot2 on the frame before it."""
    if not rows:
        return "normal"
    checks = (
        ("throw", lambda: _thrown(rows)),
        ("fireball", lambda: _fireball(rows, pre_shot)),
        ("uppercut", lambda: _uppercut(rows)),
        ("hurricane", lambda: _hurricane(rows)),
        ("slap", lambda: any(r["p2_sub"] in SLAP_SUBS for r in rows)),
        ("jump_attack", lambda: rows[0]["p2_y"] != GROUND_Y),
    )
    return next((move for move, hit in checks if hit()), "normal")


def top_move(moves: Sequence[str]) -> str:
    """The highest-precedence move of several ("none" for no move)."""
    return min(moves, key=MOVES.index) if moves else NONE


class OppMoveTracker:
    """One per round. ``add`` each action entry as it is closed (with its decision row and the rows to the next
    decision); it returns the entries that are ready, stamped. ``flush`` at the round's end returns the rest."""

    def __init__(self) -> None:
        self.episodes: List[Dict] = []       # closed: {"first": row, "frames": n, "move": ...}
        self._open: Optional[Dict] = None     # the episode being followed: {"id", "rows", "pre_shot"}
        self._held: List[Dict] = []           # {"entry", "shot", "eps"} in order
        self._last: Optional[Dict] = None     # the last row fed, and its episode id
        self._last_ep: Optional[int] = None

    def add(self, entry: Dict, before: Dict, rows: Sequence[Dict]) -> List[Dict]:
        """``rows``: every row after the decision up to the next one (the entry's window; ``[before]`` if empty).
        ``before`` is fed only if it is not the row fed last (in play_round it is the previous window's last row)."""
        fed_before = self._last_ep if self._last is not None and (before is self._last or before == self._last) \
            else self._feed(before)
        ids = [self._feed(r) for r in rows] if rows else [fed_before]
        self._held.append({"entry": entry, "shot": bool(before["shot2"]),
                           "eps": frozenset(i for i in ids if i is not None)})
        return self._release()

    def flush(self) -> List[Dict]:
        """The round is over: classify an unfinished episode from what it has and release everything."""
        self._close()
        return self._release()

    def _feed(self, r: Dict) -> Optional[int]:
        if attacking(r):
            if self._open is None:
                pre = self._last["shot2"] if self._last is not None else 0
                self._open = {"id": len(self.episodes), "rows": [], "pre_shot": pre}
            self._open["rows"].append(r)
            ep = self._open["id"]
        else:
            self._close()
            ep = None
        self._last, self._last_ep = r, ep
        return ep

    def _close(self) -> None:
        if self._open is None:
            return
        rows = self._open["rows"]
        self.episodes.append({"first": rows[0], "frames": len(rows),
                              "move": classify(rows, self._open["pre_shot"])})
        self._open = None

    def _release(self) -> List[Dict]:
        ready: List[Dict] = []
        while self._held and all(i < len(self.episodes) for i in self._held[0]["eps"]):
            h = self._held.pop(0)
            move = top_move([self.episodes[i]["move"] for i in h["eps"]])
            ready.append(dict(h["entry"], opp_move=move, opp_shot=h["shot"]))
        return ready
