"""PURE data layer for the loop monitor TUI (``scripts/monitor_tui.py``).

READ-ONLY. Nothing here touches the play path: it only parses the files the loop runner has already
flushed under ``rollouts/loop_screen/<name>/``. It imports no play/runner/screen module. The optional
G5 rule grading imports ``sf2.eval.loop_report.make_g5_scorer`` (the OFFLINE table scorer, itself
outside play) and degrades to "ungraded" if that or the table is unavailable.

All model objects are frozen dataclasses: a parse returns a fresh ``DashboardModel`` and never mutates
its inputs (owner coding-style: immutability). The TUI renderer (``scripts/monitor_tui.py``) consumes
this model; this module knows nothing about ``rich``, so it is testable headless.

What is read where (the loop flushes all of these per event):
  run.json                -> metadata (me, opp, games, rounds)
  trace.jsonl             -> seed{lines}, round{result,dealt,taken}, qwen{added,removed,in_play_after},
                             and decision{dealt,taken,follows_rule} (used for round/session STATS only)
  g<GG>_r<R>/decisions.jsonl -> FULL per-decision records (facts->HP, cat_probs, move_probs, pressed):
                             the live pipeline + current HP come from the newest round's file.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HP_MAX = 176                       # full health bar in SF2 hp units; facts healths are 0..1 fractions
ROUND_DIR_RE = re.compile(r"^g(\d+)_r(\d+)$")
# a session groups many play ROUNDS (one opponent matchup each), laid out as round_<NN>_<opp>/ -- each itself a
# normal run dir (run.json, trace.jsonl, gNN_rM/). NB "round" here is a whole matchup, not an SF2 round.
SESSION_ROUND_RE = re.compile(r"^round_(\d+)_(.+)$")

GradeFn = Callable[[str], Dict]    # a line -> {verdict, value, forward, scorable, ...} scorer


# --------------------------------------------------------------------------- model dataclasses

@dataclass(frozen=True)
class DecisionView:
    """One decision, flattened for the pipeline display (event -> category -> action -> pressed)."""
    game: int
    round: int
    k: int
    rng: str                       # close / mid / far (situation[0])
    opp_doing: str                 # what the opponent is doing (moment.doing / his_label)
    fireball: bool                 # a projectile is coming
    dx: int                        # signed horizontal gap to the opponent
    category: str
    cat_prob: float                # System 1's probability for the chosen category
    action: str
    move_prob: float               # System 1's probability for the chosen move
    rule: str                      # which rule fired (default / soft / hard / ...)
    follows_rule: bool
    pressed: str                   # compact rendering of the button sequence


@dataclass(frozen=True)
class RoundStats:
    """Totals for the current round, plus a session-wide follows-rule rate."""
    game: int
    round: int
    dealt: int
    taken: int
    hits: int                      # decisions that dealt damage
    decisions: int
    follows_rule_pct: float        # over every decision seen so far this run (0..100)


@dataclass(frozen=True)
class RoundResult:
    game: int
    round: int
    result: str
    hp: int
    dealt: int
    taken: int


@dataclass(frozen=True)
class GradedRule:
    line: str
    verdict: Optional[str]         # good / ok / bad / not_scorable, or None when ungraded
    value: Optional[float] = None
    forward: Optional[float] = None


@dataclass(frozen=True)
class QwenView:
    """Qwen's churn for one game boundary."""
    game: int
    added: Tuple[GradedRule, ...]
    removed: Tuple[str, ...]
    in_play_after: Tuple[str, ...]


@dataclass(frozen=True)
class DashboardModel:
    # metadata
    me: str
    opp: str
    games: int
    rounds: int
    run_dir: str
    # live gameplay
    cur_game: int
    cur_round: int
    me_hp: int
    opp_hp: int
    me_hp_frac: float
    opp_hp_frac: float
    decisions: Tuple[DecisionView, ...]      # newest last, up to a cap
    stats: RoundStats
    # right pane
    in_play: Tuple[str, ...]                 # current short-memory rules
    qwen: Tuple[QwenView, ...]               # per game, newest last
    results: Tuple[RoundResult, ...]
    per_game_wl: Tuple[str, ...]             # one 'W'/'L'/'T' per game played, in game order
    # inferred loop state
    qwen_thinking: bool
    status: str                              # "running" / "thinking" / "done" / "empty"
    error: Optional[str] = None              # a parse note (never raised at the caller)


@dataclass(frozen=True)
class RoundSummary:
    """One play ROUND (one opponent matchup) within a session: its per-game W/L and the CUMULATIVE win rate
    across the whole session up to and including this round -- the 'round NN vs opp -> W-L, cum win%' line."""
    num: int
    opp: str
    run_dir: str
    per_game_wl: Tuple[str, ...]       # one W/L/T per game of this round, in game order
    wins: int
    losses: int
    ties: int
    cum_wins: int                      # cumulative wins across all rounds so far
    cum_played: int                    # cumulative games played across all rounds so far
    cum_pct: int                       # cumulative win percentage after this round


@dataclass(frozen=True)
class SessionModel:
    """A whole multi-round play session (``<session>/round_<NN>_<opp>/`` subdirs). ``per_game_wl`` is the history
    strip -- every game of every round in order, so the win-rate meter 'grows'. ``active`` is the latest round's
    single-run model (reused by the renderer for the live gameplay). Immutable; ``build_session_model`` never raises."""
    session_dir: str
    rounds_planned: int                        # from session.json {rounds}, else the count of round dirs present
    opp_order: Tuple[str, ...]                 # from session.json {opp_order}, else ()
    rounds: Tuple[RoundSummary, ...]           # one per round dir, in round order
    per_game_wl: Tuple[str, ...]               # cumulative, every game of every round, in order
    per_round_wl: Tuple[str, ...]              # cumulative, every ROUND (health bar) in order - the finest unit
    cum_wins: int
    cum_played: int
    cum_pct: int
    active_num: int                            # the live (latest) round's number, -1 if none
    active_opp: str
    active: Optional[DashboardModel]           # the latest round's single-run model, None if no round dirs
    status: str                                # "running" / "done" / "empty"
    error: Optional[str] = None


# --------------------------------------------------------------------------- pure helpers

def hp_bar(frac: float, width: int = 24) -> str:
    """Render a 0..1 fraction as a text bar of ``width`` cells. Clamps out-of-range input.

    >>> hp_bar(1.0, 10)
    '##########'
    >>> hp_bar(0.0, 10)
    '----------'
    >>> hp_bar(0.5, 10)
    '#####-----'
    """
    if width <= 0:
        return ""
    try:
        f = float(frac)
    except (TypeError, ValueError):
        f = 0.0
    f = 0.0 if f < 0 else (1.0 if f > 1 else f)
    filled = int(round(f * width))
    filled = 0 if filled < 0 else (width if filled > width else filled)
    return "#" * filled + "-" * (width - filled)


# shaded-bar glyphs: full cell, the eighth-block ramp (1/8..7/8) for a partial last cell, empty track
_FULL_BLOCK = "█"                               # █
_EIGHTHS = ("▏", "▎", "▍", "▌",  # ▏ ▎ ▍ ▌
            "▋", "▊", "▉")            # ▋ ▊ ▉  (1/8 .. 7/8)
_EMPTY_BLOCK = "░"                              # ░


def _bar_color(frac: float) -> str:
    """Health -> rich color: green when >0.6, yellow 0.3..0.6, red below 0.3."""
    if frac > 0.6:
        return "green"
    if frac >= 0.3:
        return "yellow"
    return "red"


def shaded_bar(frac: float, width: int = 24, color: Optional[str] = None) -> str:
    """Render a 0..1 fraction as a rich-markup shaded bar of ``width`` cells.

    The filled part is whole '█' cells plus, for sub-cell precision, one partial cell from the
    eighth-block ramp '▏▎▍▌▋▊▉'; the empty part is a dim '░' track. The filled run is colored by
    health (``_bar_color``) unless ``color`` overrides it (e.g. the win-rate meter forces green).
    Returns VALID, balanced rich markup. Clamps out-of-range / non-numeric input.

    >>> from rich.text import Text
    >>> _ = Text.from_markup(shaded_bar(0.42, 10))   # parses cleanly
    """
    if width <= 0:
        return ""
    try:
        f = float(frac)
    except (TypeError, ValueError):
        f = 0.0
    f = 0.0 if f < 0 else (1.0 if f > 1 else f)
    total = width * 8
    eighths = int(round(f * total))
    eighths = 0 if eighths < 0 else (total if eighths > total else eighths)
    full = eighths // 8
    rem = eighths % 8
    partial = _EIGHTHS[rem - 1] if rem else ""
    empty = width - full - (1 if rem else 0)
    filled = _FULL_BLOCK * full + partial
    track = _EMPTY_BLOCK * empty
    col = color or _bar_color(f)
    out = ""
    if filled:
        out += f"[{col}]{filled}[/{col}]"
    if track:
        out += f"[dim]{track}[/dim]"
    return out


def per_game_wl(results: Sequence["RoundResult"]) -> Tuple[str, ...]:
    """Reduce per-round results to one verdict per game, in game order.

    'W' when a game won more rounds than it lost, 'L' when it lost more, 'T' on a tie (equal, or
    no decisive rounds). Derived from the round results already in the model -- no play-path access.
    """
    wins: Dict[int, int] = {}
    losses: Dict[int, int] = {}
    for r in results:
        wins.setdefault(r.game, 0)
        losses.setdefault(r.game, 0)
        if r.result == "win":
            wins[r.game] += 1
        elif r.result == "loss":
            losses[r.game] += 1
    out: List[str] = []
    for g in sorted(wins.keys()):
        w, l = wins[g], losses[g]
        out.append("W" if w > l else ("L" if l > w else "T"))
    return tuple(out)


def win_rate(per_game: Sequence[str]) -> Tuple[int, int, int]:
    """(wins, games_played, win_pct) from a per-game W/L/T sequence. Empty -> (0, 0, 0)."""
    played = len(per_game)
    wins = sum(1 for v in per_game if v == "W")
    pct = int(round(100.0 * wins / played)) if played else 0
    return wins, played, pct


# --------------------------------------------------------------------------- trend series (pure)

def cum_winrate_series(per_game: Sequence[str]) -> Tuple[float, ...]:
    """Running win fraction AFTER each game, in order: the trend that should climb as the loop learns.
    ('W','L','W') -> (1.0, 0.5, 0.667). Ties count as non-wins (same as win_rate)."""
    out: List[float] = []
    wins = 0
    for i, v in enumerate(per_game, 1):
        if v == "W":
            wins += 1
        out.append(wins / i)
    return tuple(out)


def margin_series(results: Sequence["RoundResult"]) -> Tuple[int, ...]:
    """Per-game hp margin (dealt - taken), in game order: the offense/defense trend alongside win-rate."""
    agg: Dict[int, int] = {}
    for r in results:
        agg[r.game] = agg.get(r.game, 0) + (r.dealt - r.taken)
    return tuple(agg[g] for g in sorted(agg))


_SPARK = "▁▂▃▄▅▆▇█"


def spark(values: Sequence[float], lo: Optional[float] = None, hi: Optional[float] = None) -> str:
    """A one-line sparkline over `values` using the 8 block levels. lo/hi override the auto range."""
    if not values:
        return ""
    lo = min(values) if lo is None else lo
    hi = max(values) if hi is None else hi
    span = (hi - lo) or 1.0
    out = []
    for v in values:
        t = max(0.0, min(1.0, (v - lo) / span))
        out.append(_SPARK[int(round(t * (len(_SPARK) - 1)))])
    return "".join(out)


def vbars(values: Sequence[float], height: int, lo: float = 0.0, hi: float = 1.0) -> Tuple[str, ...]:
    """A multi-row vertical bar chart: `height` strings, top row first. Each value is one column; its
    bar rises with the value (full/partial top block). Normalised to [lo, hi]. Pure, so it is testable."""
    span = (hi - lo) or 1.0
    levels = [max(0.0, min(1.0, (v - lo) / span)) * height for v in values]
    rows: List[str] = []
    for r in range(height, 0, -1):            # top row = highest
        row = []
        for lv in levels:
            if lv >= r:
                row.append("█")
            elif lv > r - 1:
                row.append(_SPARK[int((lv - (r - 1)) * (len(_SPARK) - 1))])
            else:
                row.append(" ")
        rows.append("".join(row))
    return tuple(rows)


def _pressed_str(pressed: Sequence, max_tokens: int = 4) -> str:
    """Compact a button list into 'tok tok tok xN' -- the first few tokens plus the total count,
    so a long hold ('left' x24) stays on one short line instead of flooding the pipeline column.

    >>> _pressed_str(['b', '-', 'b', '-'])
    'b - b - x4'
    >>> _pressed_str(['left'] * 24)
    'left left left left x24'
    """
    if not pressed:
        return ""
    toks = [str(p) for p in pressed]
    head = " ".join(toks[:max_tokens])
    return f"{head} x{len(toks)}"


def parse_round_dir(name: str) -> Optional[Tuple[int, int]]:
    """'g03_r1' -> (3, 1); anything else -> None."""
    m = ROUND_DIR_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _read_jsonl(path: str) -> List[dict]:
    """Read a .jsonl file into a list of dicts, skipping blank/partial trailing lines (live file)."""
    out: List[dict] = []
    try:
        with open(path, "r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    # a half-flushed final line while the loop is still writing: ignore it
                    continue
    except OSError:
        return out
    return out


def _read_json(path: str) -> dict:
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}


def list_round_dirs(run_dir: str) -> List[Tuple[int, int, str]]:
    """Sorted (game, round, dirname) for every gNN_rM dir present. Handles the dir growing live."""
    out: List[Tuple[int, int, str]] = []
    try:
        names = os.listdir(run_dir)
    except OSError:
        return out
    for name in names:
        gr = parse_round_dir(name)
        if gr is None:
            continue
        if not os.path.isdir(os.path.join(run_dir, name)):
            continue
        out.append((gr[0], gr[1], name))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


# --------------------------------------------------------------------------- grading

def get_grader() -> Optional[GradeFn]:
    """Build the offline G5 rule scorer if it (and its table) are on the branch, else None.

    Imported lazily and defensively: the TUI must still run with the scorer absent.
    """
    try:
        from .loop_report import make_g5_scorer
        return make_g5_scorer()
    except Exception:
        return None


def _grade_line(line: str, grader: Optional[GradeFn]) -> GradedRule:
    if grader is None:
        return GradedRule(line=line, verdict=None)
    try:
        r = grader(line)
        return GradedRule(line=line, verdict=r.get("verdict"),
                          value=r.get("value"), forward=r.get("forward"))
    except Exception:
        return GradedRule(line=line, verdict=None)


# --------------------------------------------------------------------------- the parse

def _me_opp_health(facts: dict, me: str, opp: str) -> Tuple[float, float]:
    """Return (my_frac, opp_frac) from a decision's facts, matching by character then falling back
    to left=me / right=opp."""
    left = facts.get("left", {}) or {}
    right = facts.get("right", {}) or {}
    lh = left.get("health")
    rh = right.get("health")
    lh = lh if isinstance(lh, (int, float)) else 0.0
    rh = rh if isinstance(rh, (int, float)) else 0.0
    if left.get("character") == me or right.get("character") == opp:
        return float(lh), float(rh)
    if right.get("character") == me or left.get("character") == opp:
        return float(rh), float(lh)
    return float(lh), float(rh)


def _decision_views(records: Sequence[dict], game: int, rnd: int, cap: int) -> List[DecisionView]:
    views: List[DecisionView] = []
    for r in records[-cap:]:
        moment = r.get("moment", {}) or {}
        sit = r.get("situation") or []
        rng = sit[0] if sit else "?"
        cat = r.get("category", "?")
        act = r.get("action", "?")
        cat_prob = float((r.get("cat_probs") or {}).get(cat, 0.0) or 0.0)
        move_prob = float((r.get("move_probs") or {}).get(act, 0.0) or 0.0)
        views.append(DecisionView(
            game=game, round=rnd, k=int(r.get("k", 0) or 0),
            rng=str(rng),
            opp_doing=str(moment.get("doing") or moment.get("his_label") or "?"),
            fireball=bool(moment.get("fireball", False)),
            dx=int(moment.get("dx", 0) or 0),
            category=str(cat), cat_prob=cat_prob,
            action=str(act), move_prob=move_prob,
            rule=str(r.get("rule", "default")),
            follows_rule=bool(r.get("follows_rule", False)),
            pressed=_pressed_str(r.get("pressed") or []),
        ))
    return views


def _round_stats(trace: Sequence[dict], game: int, rnd: int) -> RoundStats:
    """Current-round dealt/taken/hits from trace decision events; follows-rule% over the whole run."""
    dealt = taken = hits = decisions = 0
    seen = follows = 0
    for e in trace:
        if e.get("event") != "decision":
            continue
        seen += 1
        if e.get("follows_rule"):
            follows += 1
        if e.get("game") == game and e.get("round") == rnd:
            decisions += 1
            d = int(e.get("dealt", 0) or 0)
            t = int(e.get("taken", 0) or 0)
            dealt += d
            taken += t
            if d > 0:
                hits += 1
    pct = (100.0 * follows / seen) if seen else 0.0
    return RoundStats(game=game, round=rnd, dealt=dealt, taken=taken,
                      hits=hits, decisions=decisions, follows_rule_pct=round(pct, 1))


def _infer_thinking(games: int, rounds: int, round_dirs: Sequence[Tuple[int, int, str]],
                    round_events: Sequence[dict]) -> bool:
    """Qwen reflects after EVERY round (per-round reflection). 'Thinking' = the newest round has ended
    (a round event for it exists) and the next round's decisions have not started -- inferred purely
    from file state, never from a runner signal. Not thinking once the whole run is done (final round of
    the final game). Works at every round boundary, including the game boundary."""
    if not round_dirs:
        return False
    g, r, _ = round_dirs[-1]            # newest (game, round)
    if g >= games - 1 and r >= rounds - 1:
        return False                    # final round of the final game ended -> done, not thinking
    ended = any(e.get("event") == "round" and e.get("game") == g and e.get("round") == r
                for e in round_events)
    return ended                        # between this round and the next: Qwen is reflecting


def build_model(run_dir: str, grader: Optional[GradeFn] = None,
                grade_qwen: bool = True, max_decisions: int = 6) -> DashboardModel:
    """Parse a loop run dir into a ``DashboardModel``. Never raises on malformed/partial files: a
    parse note is recorded in ``.error`` and the rest renders with best-effort defaults."""
    meta = _read_json(os.path.join(run_dir, "run.json"))
    me = str(meta.get("me", "?"))
    opp = str(meta.get("opp", "?"))
    games = int(meta.get("games", 0) or 0)
    rounds = int(meta.get("rounds", 0) or 0)

    trace = _read_jsonl(os.path.join(run_dir, "trace.jsonl"))
    seed_lines: Tuple[str, ...] = ()
    qwen_events: List[dict] = []
    round_events: List[dict] = []
    for e in trace:
        ev = e.get("event")
        if ev == "seed":
            seed_lines = tuple(e.get("lines", []) or [])
        elif ev == "qwen":
            qwen_events.append(e)
        elif ev == "round":
            round_events.append(e)

    round_dirs = list_round_dirs(run_dir)

    # current game/round + its live decisions + HP
    cur_game = cur_round = 0
    decisions: List[DecisionView] = []
    me_frac = opp_frac = 1.0
    if round_dirs:
        cur_game, cur_round, dirname = round_dirs[-1]
        recs = _read_jsonl(os.path.join(run_dir, dirname, "decisions.jsonl"))
        decisions = _decision_views(recs, cur_game, cur_round, max_decisions)
        if recs:
            me_frac, opp_frac = _me_opp_health(recs[-1].get("facts", {}) or {}, me, opp)

    stats = _round_stats(trace, cur_game, cur_round)

    # short memory: latest qwen in_play_after, else the seed
    if qwen_events:
        in_play = tuple(qwen_events[-1].get("in_play_after", []) or [])
    else:
        in_play = seed_lines

    # qwen churn per game (grade the added rules)
    if grade_qwen and grader is None:
        grader = get_grader()
    qwen_views: List[QwenView] = []
    for e in qwen_events:
        added = tuple(_grade_line(ln, grader) for ln in (e.get("added", []) or []))
        qwen_views.append(QwenView(
            game=int(e.get("game", 0) or 0),
            added=added,
            removed=tuple(e.get("removed", []) or []),
            in_play_after=tuple(e.get("in_play_after", []) or []),
        ))

    results = tuple(RoundResult(
        game=int(e.get("game", 0) or 0), round=int(e.get("round", 0) or 0),
        result=str(e.get("result", "?")), hp=int(e.get("hp", 0) or 0),
        dealt=int(e.get("dealt", 0) or 0), taken=int(e.get("taken", 0) or 0),
    ) for e in round_events)

    thinking = _infer_thinking(games, rounds, round_dirs, round_events)

    if not round_dirs and not trace:
        status = "empty"
    elif thinking:
        status = "thinking"
    elif round_dirs and round_dirs[-1][0] >= games - 1 and round_dirs[-1][1] >= rounds - 1 \
            and any(e.get("game") == games - 1 and e.get("round") == rounds - 1 for e in round_events):
        status = "done"
    else:
        status = "running"

    error = None if meta else "run.json missing or unreadable"

    me_hp = int(round(max(0.0, min(1.0, me_frac)) * HP_MAX))
    opp_hp = int(round(max(0.0, min(1.0, opp_frac)) * HP_MAX))

    return DashboardModel(
        me=me, opp=opp, games=games, rounds=rounds, run_dir=run_dir,
        cur_game=cur_game, cur_round=cur_round,
        me_hp=me_hp, opp_hp=opp_hp, me_hp_frac=me_frac, opp_hp_frac=opp_frac,
        decisions=tuple(decisions), stats=stats,
        in_play=in_play, qwen=tuple(qwen_views), results=results,
        per_game_wl=per_game_wl(results),
        qwen_thinking=thinking, status=status, error=error,
    )


def latest_run_dir(base: str = os.path.join("rollouts", "loop_screen")) -> Optional[str]:
    """The newest (by mtime) run dir under ``base``, or None."""
    try:
        subs = [os.path.join(base, d) for d in os.listdir(base)]
    except OSError:
        return None
    subs = [d for d in subs if os.path.isdir(d)]
    if not subs:
        return None
    return max(subs, key=lambda d: os.path.getmtime(d))


# --------------------------------------------------------------------------- session (multi-round) layer

def parse_session_round_dir(name: str) -> Optional[Tuple[int, str]]:
    """'round_00_honda' -> (0, 'honda'); anything else -> None."""
    m = SESSION_ROUND_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), m.group(2)


def list_session_round_dirs(session_dir: str) -> List[Tuple[int, str, str]]:
    """Sorted (num, opp, dirname) for every round_<NN>_<opp> subdir present. Handles a session growing live."""
    out: List[Tuple[int, str, str]] = []
    try:
        names = os.listdir(session_dir)
    except OSError:
        return out
    for name in names:
        pr = parse_session_round_dir(name)
        if pr is None:
            continue
        if not os.path.isdir(os.path.join(session_dir, name)):
            continue
        out.append((pr[0], pr[1], name))
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def build_session_model(session_dir: str, grader: Optional[GradeFn] = None,
                        grade_qwen: bool = True, max_decisions: int = 6) -> SessionModel:
    """Aggregate across all ``round_<NN>_<opp>/`` subdirs of a session dir into a ``SessionModel``: the cumulative
    per-game W/L history (every game of every round, in order), a per-round summary with the running win rate, and
    the latest round's single-run ``DashboardModel`` for the live view. Tolerates a missing ``session.json`` and
    partial/empty rounds; never raises (parse notes go in ``.error``)."""
    meta = _read_json(os.path.join(session_dir, "session.json"))
    opp_order = tuple(meta.get("opp_order", []) or [])

    if grade_qwen and grader is None:
        grader = get_grader()

    round_dirs = list_session_round_dirs(session_dir)
    rounds_planned = int(meta.get("rounds", len(round_dirs)) or 0)

    summaries: List[RoundSummary] = []
    cum_series: List[str] = []
    round_series: List[str] = []       # per ROUND (health bar) across the whole session - the finest trend unit
    active: Optional[DashboardModel] = None
    active_num, active_opp = -1, "?"
    for num, opp, dirname in round_dirs:
        rd = os.path.join(session_dir, dirname)
        dm = build_model(rd, grader=grader, grade_qwen=grade_qwen, max_decisions=max_decisions)
        wl = list(dm.per_game_wl)
        round_series += ["W" if r.result == "win" else ("L" if r.result == "loss" else "T") for r in dm.results]
        cum_series += wl
        cw, cp, cpct = win_rate(cum_series)
        summaries.append(RoundSummary(
            num=num, opp=opp, run_dir=rd, per_game_wl=tuple(wl),
            wins=sum(1 for v in wl if v == "W"),
            losses=sum(1 for v in wl if v == "L"),
            ties=sum(1 for v in wl if v == "T"),
            cum_wins=cw, cum_played=cp, cum_pct=cpct,
        ))
        active, active_num, active_opp = dm, num, opp     # last iteration -> the live round

    cum_wins, cum_played, cum_pct = win_rate(cum_series)

    if not round_dirs:
        status = "empty"
    elif active is not None and active.status == "done" and rounds_planned and len(round_dirs) >= rounds_planned:
        status = "done"
    else:
        status = "running"

    error = None
    if not round_dirs:
        error = "no round_<NN>_<opp> subdirs under %s" % session_dir

    return SessionModel(
        session_dir=session_dir, rounds_planned=rounds_planned, opp_order=opp_order,
        rounds=tuple(summaries), per_game_wl=tuple(cum_series), per_round_wl=tuple(round_series),
        cum_wins=cum_wins, cum_played=cum_played, cum_pct=cum_pct,
        active_num=active_num, active_opp=active_opp, active=active,
        status=status, error=error,
    )
