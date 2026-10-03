"""Per-GAME diagnostic report for a loop run (README G7): "did she improve, and if not, why".

ANALYSIS ONLY, never in the play path. It reads an existing loop-run directory (nothing runs: no Qwen, no
emulator) and, GAME BY GAME, lays out the trend and the three things that could explain a flat/dropping trend,
exactly the G8 pre-registration's diagnosis tree (``docs/prereg_loop_run.md``):

    1. Qwen writes bad rules      -> an in-play rule the table (G5) marks BAD that she still FOLLOWS.
    2. text laya ignores good rules -> a G5-GOOD rule whose follows_rule is LOW where it applied.
    3. bad screen facts           -> reader-vs-RAM agreement LOW that game (from the replay referee).

Per game it reports: hp/round + rounds won/lost (the headline trend), Qwen churn (added/removed, cumulative
in-play count), follows_rule overall and PER RULE (how often text laya followed each in-play rule where it
applied), each in-play rule's G5 verdict (good/ok/bad/not_scorable, via ``sf2.eval.rule_score`` -- "unavailable"
when that offline scorer is not on the branch), and reader-vs-RAM agreement (from ``score.json`` / per-round
``replay.json``; "n/a" when the run was played with --no-score).

Inputs it reads from the run dir: ``run.json``, ``trace.jsonl`` (seed / decision / round / qwen events),
``verdict.json`` (optional), ``score.json`` (optional), and per-round ``g*_r*/replay.json`` (optional).

Pure, file-reading helpers plus one pure ``diagnose_game``; the G5 scorer is injected so the diagnosis tree is
testable without the table. Immutable: every function returns fresh dicts/lists and never mutates its inputs.
"""
import json
import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

# ---- thresholds (named, not magic), matching the pre-registration's language ----
FOLLOW_HI = 0.5     # "she still follows it": followed on at least this share of the decisions where it applied
FOLLOW_LO = 0.5     # "follows_rule low": followed on less than this share where it applied
AGREE_LO = 0.8      # "reader-vs-RAM low": the game's agreement below this flags bad screen facts
AGREE_KEY = "table_cell"   # the headline agreement axis (reader's situation cell vs RAM's)

GOOD, OK, BAD, NOT_SCORABLE, UNAVAILABLE = "good", "ok", "bad", "not_scorable", "unavailable"

# Chun-Li's two-stage menu (sf2.system1.action_menu on the G5 branch); preferred live, vendored as a fallback so
# the report parses in-play lines even on a branch without that module.
try:                                                                       # pragma: no cover - import shim
    from ..system1.action_menu import CATEGORIES as _CATS
    MENU_MOVES: Tuple[str, ...] = tuple(m for ms in _CATS.values() for m in ms)
except Exception:                                                          # pragma: no cover
    MENU_MOVES = (
        "walk_forward", "walk_back", "crouch", "jump_up", "jump_forward", "jump_back",
        "s.lp", "cl.lp", "c.lp", "j.lp", "jf.lp", "s.mp", "cl.mp", "c.mp", "j.mp", "jf.mp",
        "s.hp", "cl.hp", "c.hp", "j.hp", "jf.hp", "s.lk", "cl.lk", "c.lk", "j.lk", "jf.lk",
        "s.mk", "cl.mk", "c.mk", "j.mk", "jf.mk", "s.hk", "cl.hk", "c.hk", "j.hk", "jf.hk",
        "block_high", "block_low", "throw_F+hp", "throw_F+mp", "lightning_legs", "spinning_bird_kick",
        "jf.hk_s.mp_s.hp", "jf.mk_legs",
    )

_GR = re.compile(r"g(\d+)_r(\d+)")


# ---------- reading the run ----------

def read_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _read_json(path: str) -> Optional[Dict]:
    return json.load(open(path)) if os.path.exists(path) else None


def load_run(run_dir: str) -> Dict:
    """Everything the report needs, read once. Missing optional files come back as None/[]."""
    return {
        "dir": run_dir,
        "run": _read_json(os.path.join(run_dir, "run.json")),
        "trace": read_jsonl(os.path.join(run_dir, "trace.jsonl")),
        "verdict": _read_json(os.path.join(run_dir, "verdict.json")),
        "score": _read_json(os.path.join(run_dir, "score.json")),
    }


def _game_of_rec(rec: str) -> Optional[int]:
    m = _GR.search(rec or "")
    return int(m.group(1)) if m else None


def _round_of_rec(rec: str) -> Optional[int]:
    m = _GR.search(rec or "")
    return int(m.group(2)) if m else None


def game_indices(run: Dict) -> List[int]:
    """Every game index the run touched, from the trace, the verdict and score.json together."""
    games = set()
    for ev in run["trace"]:
        if "game" in ev and isinstance(ev["game"], int):
            games.add(ev["game"])
    if run.get("verdict"):
        games.update(range(len(run["verdict"].get("games", []))))
    for r in (run.get("score") or {}).get("rounds", []):
        g = _game_of_rec(r.get("rec", ""))
        if g is not None:
            games.add(g)
    return sorted(games)


# ---------- per-game pieces ----------

def rounds_of_game(run: Dict, g: int) -> List[Dict]:
    """hp/result per round for game ``g``. Prefers trace ``round`` events (screen result); falls back to
    ``score.json`` rounds (the replay referee) when the trace has none (e.g. a run that only emitted a seed)."""
    rounds = [{"round": ev.get("round"), "result": ev.get("result"), "hp": ev.get("hp"),
               "dealt": ev.get("dealt"), "taken": ev.get("taken"), "source": ev.get("source", "screen")}
              for ev in run["trace"] if ev.get("event") == "round" and ev.get("game") == g]
    if rounds:
        return sorted(rounds, key=lambda r: (r["round"] is None, r["round"]))
    out = []
    for r in (run.get("score") or {}).get("rounds", []):
        if _game_of_rec(r.get("rec", "")) == g:
            out.append({"round": _round_of_rec(r.get("rec", "")), "result": r.get("result"), "hp": r.get("hp"),
                        "dealt": r.get("dealt"), "taken": r.get("taken"), "source": "replay"})
    return sorted(out, key=lambda r: (r["round"] is None, r["round"]))


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def won_lost_of_game(run: Dict, g: int, rounds: Sequence[Dict]) -> Dict:
    """Rounds won/lost: from verdict.json when present, else counted from round results."""
    v = run.get("verdict")
    if v and g < len(v.get("games", [])):
        gg = v["games"][g]
        return {"won": gg.get("won"), "lost": gg.get("lost"), "source": "verdict"}
    won = sum(1 for r in rounds if r.get("result") == "win")
    lost = sum(1 for r in rounds if r.get("result") == "loss")
    return {"won": won, "lost": lost, "source": "rounds"}


def decisions_of_game(run: Dict, g: int) -> List[Dict]:
    return [ev for ev in run["trace"] if ev.get("event") == "decision" and ev.get("game") == g]


def qwen_of_game(run: Dict, g: int) -> Optional[Dict]:
    for ev in run["trace"]:
        if ev.get("event") == "qwen" and ev.get("game") == g:
            return {"added": list(ev.get("added", [])), "removed": list(ev.get("removed", [])),
                    "in_play_after": list(ev.get("in_play_after", [])),
                    "in_play_count": len(ev.get("in_play_after", []))}
    return None


def in_play_lines_of_game(run: Dict, g: int, decisions: Sequence[Dict], qwen: Optional[Dict]) -> List[str]:
    """The in-play rules that governed game ``g``: the lines the decisions carried (preferred), else Qwen's
    after-list, else the seed lines."""
    for d in decisions:
        if d.get("lines"):
            return list(d["lines"])
    if qwen and qwen["in_play_after"]:
        return list(qwen["in_play_after"])
    for ev in run["trace"]:
        if ev.get("event") == "seed" and ev.get("lines"):
            return list(ev["lines"])
    return []


def follows_overall(decisions: Sequence[Dict]) -> Dict:
    n = len(decisions)
    followed = sum(1 for d in decisions if d.get("follows_rule"))
    return {"decisions": n, "followed": followed, "share": (followed / n if n else None)}


def _reader():
    """``advice.read`` / ``opp_doing`` if importable (both branches have them); else None (per-rule follows = n/a)."""
    try:
        from ..system1.advice import read, opp_doing  # noqa
        return read, opp_doing
    except Exception:                                  # pragma: no cover
        return None


def _situation(d: Dict) -> Tuple[Optional[str], Optional[str]]:
    sit = d.get("situation") or []
    rng = sit[0] if len(sit) > 0 else None
    doing = sit[1] if len(sit) > 1 else None
    return rng, doing


def follows_per_rule(decisions: Sequence[Dict], lines: Sequence[str],
                     moves: Sequence[str] = MENU_MOVES) -> Dict[str, Dict]:
    """How often text laya followed each in-play rule WHERE IT APPLIED. A rule applies to a decision when its
    range/condition match the decision's situation; "followed" = she did the move (for use_more/always) or did
    not do it (for avoid). A line that names no readable move, or needs the advice parser that is absent, is n/a.
    """
    rd = _reader()
    out: Dict[str, Dict] = {}
    for line in lines:
        if rd is None:
            out[line] = {"applies": None, "followed": None, "share": None, "note": "advice parser unavailable"}
            continue
        read, _ = rd
        les = read(line, list(moves))
        if les.move is None or les.polarity == "none":
            out[line] = {"applies": None, "followed": None, "share": None, "note": "line names no move"}
            continue
        applies = followed = 0
        for d in decisions:
            rng, doing = _situation(d)
            if rng is None or doing is None or not les.applies(rng, doing):
                continue
            applies += 1
            action = d.get("action")
            did = (action != les.move) if les.polarity == "neg" else (action == les.move)
            followed += int(did)
        out[line] = {"applies": applies, "followed": followed,
                     "share": (followed / applies if applies else None),
                     "move": les.move, "polarity": les.polarity}
    return out


def g5_of_rules(lines: Sequence[str], scorer: Optional[Callable[[str], Dict]]) -> Dict[str, Dict]:
    """Each in-play rule's G5 verdict. ``scorer`` is a line->{verdict,...} callable (``make_g5_scorer``); None
    (the scorer/table not on this branch) -> every rule "unavailable"."""
    out: Dict[str, Dict] = {}
    for line in lines:
        out[line] = {"verdict": UNAVAILABLE} if scorer is None else scorer(line)
    return out


def reader_vs_ram_of_game(run: Dict, g: int, key: str = AGREE_KEY) -> Dict:
    """Reader-vs-RAM agreement for game ``g``, averaged over its rounds' replay agreement. n/a with no replay
    (the run was played --no-score)."""
    rows = [r for r in (run.get("score") or {}).get("rounds", []) if _game_of_rec(r.get("rec", "")) == g]
    rows = [r for r in rows if r.get("agreement")]
    if not rows:
        return {"agreement": None, "rounds": 0, "note": "no replay (run played --no-score)"}
    headline = _mean([r["agreement"].get(key) for r in rows])
    keys = sorted({k for r in rows for k in r["agreement"]})
    per_axis = {k: _mean([r["agreement"].get(k) for r in rows]) for k in keys}
    overall = _mean([v for v in per_axis.values() if v is not None])
    return {"agreement": headline, "key": key, "overall": overall, "rounds": len(rows), "per_axis": per_axis}


# ---------- the diagnosis tree (pure) ----------

def diagnose_game(improved: Optional[bool], g5: Dict[str, Dict], follows: Dict[str, Dict],
                  agreement: Optional[float]) -> Dict:
    """Flag the likely cause when hp did NOT improve (``improved`` is False). Inputs are already-computed: the
    per-rule G5 verdicts, the per-rule follows shares, and the game's reader-vs-RAM agreement. Pure -- no files,
    no scorer -- so the tree is testable directly.

    (a) bad rules      : a rule G5 marks BAD that she still FOLLOWS (share >= FOLLOW_HI).
    (b) not following  : a rule G5 marks GOOD whose follows share is LOW (< FOLLOW_LO).
    (c) bad facts      : reader-vs-RAM agreement < AGREE_LO.
    """
    if improved is None:
        return {"improved": None, "flags": [], "causes": {}, "note": "first game: no prior game to improve on"}
    if improved:
        return {"improved": True, "flags": [], "causes": {}, "note": "hp improved over the previous game"}

    bad_followed, good_ignored = [], []
    for line, sc in g5.items():
        verdict = sc.get("verdict")
        share = (follows.get(line) or {}).get("share")
        if verdict == BAD and share is not None and share >= FOLLOW_HI:
            bad_followed.append({"line": line, "follows_share": share})
        if verdict == GOOD and share is not None and share < FOLLOW_LO:
            good_ignored.append({"line": line, "follows_share": share})

    flags, causes = [], {}
    if bad_followed:
        flags.append("a_bad_rules")
        causes["a_bad_rules"] = bad_followed
    if good_ignored:
        flags.append("b_not_following")
        causes["b_not_following"] = good_ignored
    if agreement is not None and agreement < AGREE_LO:
        flags.append("c_bad_facts")
        causes["c_bad_facts"] = {"agreement": agreement, "threshold": AGREE_LO}

    note = "no cause flagged" if not flags else "likely cause(s): " + ", ".join(flags)
    g5_seen = any(sc.get("verdict") in (GOOD, OK, BAD, NOT_SCORABLE) for sc in g5.values())
    if not g5_seen:
        note += " [G5 unavailable: (a)/(b) not evaluated]"
    if agreement is None:
        note += " [reader-vs-RAM n/a: (c) not evaluated]"
    return {"improved": False, "flags": flags, "causes": causes, "note": note}


# ---------- assembling the report ----------

def game_report(run: Dict, g: int, prev_mean_hp: Optional[float],
                scorer: Optional[Callable[[str], Dict]]) -> Dict:
    rounds = rounds_of_game(run, g)
    mean_hp = _mean([r["hp"] for r in rounds])
    improved = None if prev_mean_hp is None or mean_hp is None else (mean_hp > prev_mean_hp)
    decisions = decisions_of_game(run, g)
    qwen = qwen_of_game(run, g)
    lines = in_play_lines_of_game(run, g, decisions, qwen)
    g5 = g5_of_rules(lines, scorer)
    follows = follows_per_rule(decisions, lines)
    rr = reader_vs_ram_of_game(run, g)
    diag = diagnose_game(improved, g5, follows, rr["agreement"])
    return {
        "game": g,
        "rounds": rounds,
        "mean_hp_per_round": mean_hp,
        "improved": improved,
        "won_lost": won_lost_of_game(run, g, rounds),
        "qwen": qwen,
        "in_play_lines": lines,
        "follows_overall": follows_overall(decisions),
        "follows_per_rule": follows,
        "g5": g5,
        "g5_summary": _g5_summary(g5),
        "reader_vs_ram": rr,
        "diagnosis": diag,
    }


def _g5_summary(g5: Dict[str, Dict]) -> Dict[str, int]:
    out = {GOOD: 0, OK: 0, BAD: 0, NOT_SCORABLE: 0, UNAVAILABLE: 0}
    for sc in g5.values():
        out[sc.get("verdict", UNAVAILABLE)] = out.get(sc.get("verdict", UNAVAILABLE), 0) + 1
    return dict(out, rules=len(g5))


def build_report(run_dir: str, scorer: Optional[Callable[[str], Dict]] = None) -> Dict:
    """The full per-game report for a loop-run directory."""
    run = load_run(run_dir)
    games = []
    prev = None
    for g in game_indices(run):
        gr = game_report(run, g, prev, scorer)
        games.append(gr)
        if gr["mean_hp_per_round"] is not None:
            prev = gr["mean_hp_per_round"]
    return {
        "run_dir": run_dir,
        "opp": (run.get("run") or {}).get("opp"),
        "me": (run.get("run") or {}).get("me"),
        "qwen": (run.get("run") or {}).get("qwen"),
        "g5_available": scorer is not None,
        "n_games": len(games),
        "games": games,
    }


# ---------- the G5 scorer (imports the offline table scorer when it is on the branch) ----------

def make_g5_scorer(table_path: Optional[str] = None) -> Optional[Callable[[str], Dict]]:
    """A line -> {verdict, value, forward, reason, move} callable backed by ``sf2.eval.rule_score`` + the value
    table. Returns None (G5 unavailable) when the scorer or the table is not on this branch -- the report then
    reports every rule "unavailable" rather than guessing."""
    try:
        from ..data.value_oracle import load
        from .rule_score import rule_from_line, score_rule
    except Exception:
        return None
    path = table_path or os.path.join("lessons", "value_oracle_v1.json")
    if not os.path.exists(path):
        return None
    table = load(path)

    def scorer(line: str) -> Dict:
        rs = score_rule(rule_from_line(line, source="in_play", moves=MENU_MOVES), table)
        return {"verdict": rs.verdict, "value": rs.value, "forward": rs.forward,
                "reason": rs.reason, "move": rs.rule.move, "scorable": rs.scorable}

    return scorer


# ---------- readable text summary ----------

def _fmt_share(x: Optional[float]) -> str:
    return "n/a" if x is None else "%.0f%%" % (100 * x)


def _fmt_hp(x: Optional[float]) -> str:
    return "n/a" if x is None else "%+.1f" % x


def render_text(report: Dict) -> str:
    L: List[str] = []
    L.append("loop report: %s  (me=%s vs %s, qwen=%s, G5=%s)" % (
        report["run_dir"], report.get("me"), report.get("opp"), report.get("qwen"),
        "on" if report["g5_available"] else "UNAVAILABLE"))
    hps = [g["mean_hp_per_round"] for g in report["games"]]
    L.append("trend (mean hp/round by game): " + " -> ".join(_fmt_hp(h) for h in hps))
    for g in report["games"]:
        L.append("")
        wl = g["won_lost"]
        imp = {True: "improved", False: "did NOT improve", None: "baseline"}[g["improved"]]
        L.append("== game %d == mean hp/round %s (%s)  rounds won %s/lost %s"
                 % (g["game"], _fmt_hp(g["mean_hp_per_round"]), imp, wl["won"], wl["lost"]))
        for r in g["rounds"]:
            L.append("   round %s: %s hp=%s (dealt %s / taken %s) [%s]"
                     % (r["round"], r["result"], _fmt_hp(r["hp"]), r["dealt"], r["taken"], r["source"]))
        q = g["qwen"]
        if q:
            L.append("   qwen churn: +%d added %s  -%d removed %s  in-play now %d"
                     % (len(q["added"]), q["added"], len(q["removed"]), q["removed"], q["in_play_count"]))
        else:
            L.append("   qwen churn: n/a (no qwen event this game)")
        fo = g["follows_overall"]
        L.append("   follows_rule overall: %s (%s of %d decisions)"
                 % (_fmt_share(fo["share"]), fo["followed"], fo["decisions"]))
        for line in g["in_play_lines"]:
            fr = g["follows_per_rule"].get(line, {})
            sc = g["g5"].get(line, {})
            extra = fr.get("note") or ("applied %s, followed %s" % (fr.get("applies"), fr.get("followed")))
            L.append("     - [%s] follows=%s (%s)  :: %s"
                     % (sc.get("verdict", "?"), _fmt_share(fr.get("share")), extra, line))
        gs = g["g5_summary"]
        L.append("   G5: good=%d ok=%d bad=%d not_scorable=%d unavailable=%d"
                 % (gs[GOOD], gs[OK], gs[BAD], gs[NOT_SCORABLE], gs[UNAVAILABLE]))
        rr = g["reader_vs_ram"]
        L.append("   reader-vs-RAM (%s): %s over %d round(s)%s"
                 % (rr.get("key", AGREE_KEY), _fmt_share(rr["agreement"]), rr["rounds"],
                    "" if rr["agreement"] is not None else " [" + rr.get("note", "n/a") + "]"))
        d = g["diagnosis"]
        L.append("   DIAGNOSIS: %s" % d["note"])
        for flag in d["flags"]:
            L.append("      * %s: %s" % (flag, json.dumps(d["causes"][flag])))
    return "\n".join(L)
