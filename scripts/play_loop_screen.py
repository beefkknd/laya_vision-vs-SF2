"""The SCREEN-ONLY, Qwen-in-loop self-learning loop (README M1 + minimal G3), Chun-Li vs one opponent:

    seed (web/book rules -> short memory)
      -> play a game (screen only, text laya follows the short memory: sf2.system1.loop_runner, NO table / RAM)
      -> screen evidence (sf2.system2.screen_evidence: the lesson rows, from the screen record + the OFFLINE replay)
      -> System 2 (sf2.system2.lessons.review + Qwen + lessons.propose: admit / keep / retire advice)
      -> the updated short memory feeds the NEXT game.

This driver wires the clean play module (sf2.system1.loop_runner) to the System-2 pieces that legitimately read the
offline replay (RAM AFTER play) and run Qwen. It is NOT on the hard gate's play path: only sf2.system1.loop_runner is
(scripts/hard_gate.py --entry sf2/system1/loop_runner.py is clean). ``run_loop`` takes its heavy pieces INJECTED
(the advisor, a Qwen caller, a play function, a replay scorer) so tests drive the whole loop with a mock Qwen and a
follower text laya and never touch the network or the emulator; ``main`` wires the real ones.

    python scripts/play_loop_screen.py --opp honda --games 10 --book lessons/book.json      # needs Qwen ON

Trace (<out>/trace.jsonl): the seed rules; per decision (from the round records) facts / words / advice in force /
category+move / follows_rule; per round result + hp; per Qwen step the claims proposed, their verdicts and the
short-memory diff.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import _path  # noqa: F401
from sf2.config import PORTS
from sf2.system1.advice import char_menu_moves, followable
from sf2.system1.loop_runner import play_round as play_screen_round, two_stage_decide
from sf2.system1 import value_table as VT
from sf2.quorum import reliability as QR
from sf2.quorum.config import QuorumConfig
from sf2.quorum.decider import quorum_decider
from sf2.system2 import character_prompt, lessons as L, rule_stats, screen_evidence, seed_rules, short_memory as SM
from sf2.system2.lesson_prompt import streak

ME = "chunli"
# the characters we can play AS: only those with a RAM-free move menu in sf2.moves_free (chunli/ryu/ken). The round-1
# categories and round-2 move menu, and the advice vocabulary, come from ME's own moveset (sf2.system1.advice).
SUPPORTED_ME = ("chunli", "ryu", "ken")
MENU_MOVES: List[str] = char_menu_moves(ME)       # Chun-Li's followable (two-stage) vocabulary; == the old constant
DELAY_MIN, DELAY_SPAN = 4, 40


# ------------------------------------------------------------------ seeding
def seed(opp: str, book: Optional[str], me: str = ME) -> L.Registry:
    """The starting short memory: the opponent's verified web/book lines (sf2.system2.seed_rules), [] without a book.
    The book is written for ONE character (its ``me`` field, currently Chun-Li); for any other ME there is no seed, so
    the loop starts blank (logged) rather than feeding ME lines in another character's moveset."""
    if not book:
        return []
    doc, _sha = seed_rules.load_book(book)
    book_me = doc.get("me", seed_rules.DEFAULT_ME)
    if me != book_me:
        print("no seed for %s, starting blank (book is for %s)" % (me, book_me), file=sys.stderr)
        return []
    return seed_rules.seed_lessons(book, opp, me)


# ------------------------------------------------------------------ registry carryover (multi-round play)
def save_registry(reg: L.Registry, path: str) -> None:
    """Write the FULL registry (the loop's ``L.Registry`` -- the verdict's ``registry_end`` / in-play state) so a
    later round can ``--carry`` it and learn ON TOP of these rules. Reuses ``sf2.system2.lessons.dump`` if that ever
    exists; otherwise JSON of the registry list (the registry is a plain list of entry dicts, so this round-trips)."""
    dump = getattr(L, "dump", None)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    if callable(dump):
        dump(reg, path)
        return
    with open(path, "w") as f:
        json.dump(list(reg), f, indent=1)


def load_registry(path: str) -> L.Registry:
    """Read a registry written by ``save_registry`` back into the loop's ``L.Registry`` (list of entry dicts).
    The inverse of ``save_registry``; raises ``ValueError`` if the file is not a list of entries."""
    load = getattr(L, "load", None)
    if callable(load):
        return load(path)
    with open(path) as f:
        reg = json.load(f)
    if not isinstance(reg, list):
        raise ValueError("carry registry %s must be a JSON list of entries, got %s" % (path, type(reg).__name__))
    return reg


def starting_registry(carry: Optional[str], opp: str, book: Optional[str],
                      me: str = ME) -> Tuple[L.Registry, str]:
    """The registry the loop STARTS from. With ``--carry`` and the file present, carry it (learn on top of those
    rules); a missing carry file falls back to the normal book seed (logged). Returns ``(registry, source)`` where
    source is "carry" or "book"."""
    if carry and os.path.exists(carry):
        return load_registry(carry), "carry"
    if carry:
        print("carry file %s absent, seeding from book instead" % carry, file=sys.stderr)
    return seed(opp, book, me), "book"


# ------------------------------------------------------------------ the Qwen update (System 2)
QwenCaller = Callable[[List[Dict], str], str]      # (messages, task) -> raw reply text


def summarize_game(opp: str, reg: L.Registry, rows: Sequence[Dict], last: Sequence[Dict], all_rounds: Sequence[Dict],
                   last_rounds: Sequence[Dict], game_hp: Sequence[float], games_wl: Sequence[Dict],
                   ask_qwen: QwenCaller, me: str = ME) -> Dict:
    """STAGE 1 SCOUT: the code-computed per-game digest (faithful to the rows), with the Scout's prose summary attached
    as ``notes``. The facts are queryable (sf2.system2.character_prompt.digest_facts); the model only narrates them, so
    a down/garbled Scout leaves the facts intact (``notes`` empty, ``scout_error`` set)."""
    digest = character_prompt.digest_facts(me, opp, reg, rows, last, all_rounds, last_rounds, game_hp, games_wl,
                                           char_menu_moves(me))
    msgs = character_prompt.summary_messages(me, opp, reg, rows, last, all_rounds, last_rounds, game_hp, games_wl,
                                             char_menu_moves(me))
    try:
        notes = ask_qwen(msgs, "scout_%s" % opp).strip()
        return dict(digest, notes=notes)
    except Exception as e:                       # Scout down / cut off: the facts still stand, no prose
        return dict(digest, notes="", scout_error="%s: %s" % (type(e).__name__, e))


def ask_claims(opp: str, reg: L.Registry, rows: Sequence[Dict], last: Sequence[Dict], all_rounds: Sequence[Dict],
               last_rounds: Sequence[Dict], refused: Sequence[Dict], stable: Optional[str],
               ask_qwen: QwenCaller, me: str = ME, qwen_mode: str = "two",
               digest: Optional[Dict] = None) -> Tuple[List[Dict], List[str], Optional[str]]:
    """STAGE 2 COACH's claims for this game. ``qwen_mode`` "two" (default) feeds the Scout digest into a strategize
    prompt in one of two modes (escalate when losing: offense only, a defensive answer is dropped mechanically;
    consolidate otherwise); "one" is the OLD single prompt, the A/B fallback. ([], [problem], None) if the call or
    parse fails. parse_claims is unchanged either way."""
    from sf2.system2.qwen import json_reply
    two = qwen_mode == "two" and digest is not None
    mode = character_prompt.coach_mode(digest) if two else None
    if two:
        msgs = character_prompt.strategize_messages(me, opp, reg, digest, char_menu_moves(me), refused, mode)
        task = "coach_%s" % opp
    else:
        msgs = character_prompt.messages(me, opp, reg, rows, last, all_rounds, last_rounds, char_menu_moves(me),
                                         refused, stable)
        task = "loop_%s" % opp
    try:
        raw = ask_qwen(msgs, task)
        claims, problems = character_prompt.parse_claims(json_reply(raw))
        if two:
            claims, dropped = character_prompt.coach_filter(claims, mode)
            problems = list(problems) + dropped
    except Exception as e:                       # Qwen down / cut off / not JSON: no claims this game
        return [], ["%s: %s" % (type(e).__name__, e)], None
    return claims, problems, raw


def update(reg: L.Registry, rows: List[Dict], game: int, game_hp: List[float], games: List[Dict],
           changed: List[bool], opp: str, last: List[Dict], all_rounds: List[Dict], last_rounds: List[Dict],
           refused: List[Dict], ask_qwen: QwenCaller, me: str = ME, qwen_mode: str = "two") -> Tuple[L.Registry, Dict]:
    """One System-2 step (per round): the Coach (Qwen) proposes claims; the SIMPLE live policy
    (sf2.system2.short_memory) decides the short memory -- while losing it swaps one line (drop the weakest
    trying, admit one fresh claim), while winning it freezes. ``games`` is the per-ROUND win/loss list, the
    loss-streak signal. The measurement (lessons.condition_evidence) is the ADVISORY graduation scorer only;
    it never blocks a change. Returns the new registry and a trace record. ``qwen_mode`` "two" = Scout+Coach."""
    before = SM.in_play(reg)
    pre_kept = {r["line"] for r in reg if r["state"] in SM.IMMUNE}
    stable = streak(games, changed)
    digest = summarize_game(opp, reg, rows, last, all_rounds, last_rounds, game_hp, games, ask_qwen, me) \
        if qwen_mode == "two" else None
    claims, problems, raw = ask_claims(opp, reg, rows, last, all_rounds, last_rounds, refused, stable,
                                       ask_qwen, me, qwen_mode, digest)
    moves = set(char_menu_moves(me))
    fol = lambda move, rng: followable(move, rng, me)        # drop/refuse rules text-laya can't play at their range
    # POOL per-rule stats across blocks (Step 1B): fold THIS round's decisions into each in-play rule's stats BEFORE
    # the policy reads them, so MIN_TRIES becomes reachable and a rule can actually graduate. Carried by the registry.
    summary = last_rounds[0] if last_rounds else {}
    reg = [dict(r, stats=rule_stats.tally(r.get("stats"), last, summary, r["claim"]))
           if r["state"] in SM.IN_PLAY_STATES else r for r in reg]
    reg, event = SM.step(reg, games, claims, rows, moves, rotate=game, scorer=L.condition_evidence, followable=fol)
    after = SM.in_play(reg)
    promoted = [r["line"] for r in reg if r["state"] == SM.KEPT and r["line"] not in pre_kept]
    admitted = set(event["added"])
    not_taken = [{"claim": c, "why": "not admitted (one change per round while losing)"}
                 for c in claims if not (SM._valid(c, moves) and L.render(c) in admitted)]
    trace = {"game": game, "mode": character_prompt.coach_mode(digest) if digest else "one", "stable": stable,
             "scout": digest, "claims": claims, "problems": problems,
             "outcome": [{"line": l, "state": "trying"} for l in event["added"]],
             "in_play_before": before, "in_play_after": after,
             "added": event["added"], "removed": event["removed"],
             "promoted": promoted, "retired": [], "streak": event["streak"],
             "violations": [], "reply": raw}
    return reg, dict(trace, refused=not_taken)


# ------------------------------------------------------------------ the loop
PlayFn = Callable[..., Dict]        # like sf2.system1.loop_runner.play_round
ScoreFn = Callable[[str], Optional[Dict]]    # a round dir -> its offline replay score (result, hp, ...), or None


def run_loop(opp: str, cat_advisor, move_advisor, ask_qwen: QwenCaller, *, games: int, rounds: int,
             seed_lines: L.Registry, out: str, play_round_fn: PlayFn, state: bytes, state_id: Dict, emu,
             score_fn: Optional[ScoreFn] = None, reader=None, seed_rng: int = 0, me: str = ME, log=None,
             qwen_mode: str = "two", policy: str = "rules", table: Optional[Dict] = None,
             save_table: Optional[str] = None, explore_rate: float = 0.1,
             explore_tries: int = VT.MIN_TRIES, quorum: Optional[QuorumConfig] = None,
             quorum_state: Optional[Dict] = None, save_quorum: Optional[str] = None, qwen_pick=None,
             credit_horizon: int = 0, credit_gamma: float = 1.0, no_learn: bool = False) -> Dict:
    """Play ``games`` games. Two policies (owner's two-system A/B): ``policy="rules"`` (default) = text-laya follows
    the short memory, rotated by System 2 after each round; ``policy="table"`` = the self-learning value table
    (sf2.system1.value_table) picks the move and is CREDITED from each round's outcomes (no Qwen, no short memory).
    Everything heavy is injected: ``cat_advisor``/``move_advisor`` (text laya), ``ask_qwen``, ``play_round_fn``,
    ``emu``, ``score_fn``. For the table policy, ``table`` is the carried table (blank if None) and ``save_table`` a
    path to persist it to after the run (so it GROWS across blocks like the registry). Returns the verdict."""
    os.makedirs(out, exist_ok=True)
    trace = open(os.path.join(out, "trace.jsonl"), "w") if log is None else log
    reg: L.Registry = list(seed_lines)
    tbl: Dict = table if table is not None else VT.blank()
    qcfg: QuorumConfig = quorum if quorum is not None else QuorumConfig()       # policy="quorum" only
    qrel: Dict = quorum_state if quorum_state is not None else QR.blank()
    qsources: Dict[str, int] = {}
    qsplit = [0, 0]                                   # [decisions with no quorum, decisions with a quorum record]
    rng = random.Random(seed_rng)
    explore = random.Random((seed_rng << 1) ^ 0x5F3759DF)      # a separate stream for the table's exploration
    trace.write(json.dumps({"event": "seed", "opp": opp, "lines": SM.in_play(reg),
                            "rules": [{"line": r["line"], "state": r["state"], "source": r.get("evidence", {}).get("source")}
                                      for r in reg]}) + "\n")
    trace.flush()
    all_rows: List[Dict] = []
    all_rounds: List[Dict] = []
    game_hp: List[float] = []
    games_wl: List[Dict] = []
    changed: List[bool] = []
    refused: List[Dict] = []
    round_hp: List[float] = []        # per-ROUND margin / win-loss drive per-round reflection
    round_wl: List[Dict] = []
    idx = 0                           # monotonic reflection unit (one per round) for churn timing
    for g in range(games):
        this_rounds: List[Dict] = []
        for r in range(rounds):
            rd = os.path.join(out, "g%02d_r%d" % (g, r))
            delay = DELAY_MIN + rng.randrange(DELAY_SPAN)
            if policy == "table":                        # the value table picks the move (no short memory, no Qwen)
                play_round_fn(emu, cat_advisor, move_advisor, me, opp, state, state_id, delay, [], rd,
                              reader=reader, decide=VT.decider(tbl, me, explore))
            elif policy == "quorum":                     # the bee quorum: laya + 3 flavours + the table vote (sf2/quorum)
                lines = SM.in_play(reg)
                base = lambda m, _l=lines: two_stage_decide(cat_advisor, move_advisor, me, m, _l)
                play_round_fn(emu, cat_advisor, move_advisor, me, opp, state, state_id, delay, lines, rd,
                              reader=reader, decide=quorum_decider(tbl, qrel, me, explore, base, cat_advisor,
                                                                   move_advisor, qcfg, qwen_pick))
            elif policy == "hybrid":                     # text-laya plays; the table OVERRIDES where it is confident
                lines = SM.in_play(reg)
                base = lambda m, _l=lines: two_stage_decide(cat_advisor, move_advisor, me, m, _l)
                play_round_fn(emu, cat_advisor, move_advisor, me, opp, state, state_id, delay, lines, rd,
                              reader=reader, decide=VT.hybrid_decider(tbl, me, explore, base,
                                                                      min_tries=explore_tries, explore=explore_rate))
            else:
                lines = SM.in_play(reg)                  # FRESH each round: play with the latest short memory
                play_round_fn(emu, cat_advisor, move_advisor, me, opp, state, state_id, delay, lines, rd, reader=reader)
            emu = emu.new_round()
            replay = score_fn(rd) if score_fn else None
            decisions = screen_evidence.read_decisions(rd)
            end_bars = None if replay else screen_evidence.read_end_bars(rd)   # --no-score: the KO from the last read
            drows, summary = screen_evidence.round_evidence(g, me, opp, decisions, replay, end_bars=end_bars)
            all_rows += drows
            all_rounds.append(summary)
            this_rounds.append(summary)
            for d, row in zip(decisions, drows):
                trace.write(json.dumps({"event": "decision", "game": g, "round": r, "k": d.get("k"),
                                        "words": d.get("advice_text"), "situation": d.get("situation"),
                                        "lines": d.get("lines"), "category": d.get("category"),
                                        "action": d.get("action"), "rule": d.get("rule"),
                                        "follows_rule": d.get("follows_rule"), "source": d.get("source"),
                                        "quorum": d.get("quorum"),
                                        "dealt": row["dealt"], "taken": row["taken"]}) + "\n")
            trace.write(json.dumps({"event": "round", "game": g, "round": r, "result": summary["result"],
                                    "hp": summary["hp"], "dealt": summary["dealt"], "taken": summary["taken"],
                                    "source": summary["source"]}) + "\n")
            trace.flush()
            if policy in ("table", "hybrid", "quorum") and not no_learn:  # CREDIT the table (skipped in --no-learn frozen eval)
                tbl = VT.credit(tbl, drows, horizon=credit_horizon, gamma=credit_gamma)
                if policy == "quorum":                   # ...and each voter's reliability, from the same outcomes
                    QR.credit(qrel, decisions, drows, qcfg)
                    rsrc: Dict[str, int] = {}
                    for d in decisions:
                        rsrc[d.get("source") or "?"] = rsrc.get(d.get("source") or "?", 0) + 1
                        qsources[d.get("source") or "?"] = qsources.get(d.get("source") or "?", 0) + 1
                        if d.get("quorum"):
                            qsplit[0] += d["quorum"].get("quorum_move") is None
                            qsplit[1] += 1
                    trace.write(json.dumps({"event": "quorum", "game": g, "round": r, "mode": qcfg.mode,
                                            "sources": rsrc, "when_cells": len(qrel["rel"])}) + "\n")
                trace.write(json.dumps({"event": "table", "game": g, "round": r,
                                        "cells": len(tbl["cells"]), "splits": len(tbl.get("depth", {}))}) + "\n")
                trace.flush()
                if policy == "table":
                    continue                             # pure table: no Qwen/short-memory; hybrid falls through to it
            if no_learn:                                 # frozen EVAL: no table credit (above) AND no memory rotation
                idx += 1
                continue
            # SYSTEM 2 AFTER EACH ROUND: reflect and rotate the short memory so the NEXT round can adapt.
            round_hp.append(summary["dealt"] - summary["taken"])
            round_wl.append({"won": 1 if summary["result"] == "win" else 0,
                             "lost": 0 if summary["result"] == "win" else 1})
            reg, qtrace = update(reg, all_rows, idx, round_hp, round_wl, changed, opp, drows, all_rounds,
                                 [summary], refused, ask_qwen, me, qwen_mode)
            refused = qtrace.pop("refused")
            changed.append(bool(qtrace["added"] or qtrace["removed"] or qtrace["promoted"] or qtrace["retired"]))
            if qtrace.get("scout") is not None:
                trace.write(json.dumps(dict(qtrace["scout"], event="scout", game=g, round=r)) + "\n")
            trace.write(json.dumps(dict(qtrace, event="qwen", game=g, round=r)) + "\n")
            trace.flush()
            idx += 1
        game_hp.append(sum(s["dealt"] - s["taken"] for s in this_rounds) / max(1, len(this_rounds)))
        games_wl.append({"won": sum(s["result"] == "win" for s in this_rounds),
                         "lost": sum(s["result"] != "win" for s in this_rounds)})
    verdict = {"opp": opp, "games": games, "rounds": rounds, "policy": policy,
               "seed_lines": [r["line"] for r in seed_lines],
               "in_play_end": SM.in_play(reg), "registry_end": reg, "decisions": len(all_rows),
               "game_hp": game_hp, "games": games_wl, "violations": L.violations(reg, all_rows)}
    if policy == "quorum":
        verdict["quorum"] = {"mode": qcfg.mode, "config": qcfg.to_dict(), "sources": qsources,
                             "escalation_rate": qsplit[0] / max(1, qsplit[1]),   # share of decisions with no quorum
                             "decisions": qsplit[1]}
        if save_quorum:
            tmp = save_quorum + ".tmp"
            with open(tmp, "w") as f:
                json.dump(qrel, f)
            os.replace(tmp, save_quorum)
    if policy in ("table", "hybrid", "quorum"):
        verdict["table_cells"] = len(tbl["cells"])
        verdict["table_splits"] = len(tbl.get("depth", {}))
        if save_table:                                        # persist so the table GROWS across blocks (like career_reg)
            tmp = save_table + ".tmp"
            with open(tmp, "w") as f:
                json.dump(tbl, f)
            os.replace(tmp, save_table)
    with open(os.path.join(out, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1)
    if log is None:
        trace.close()
    return verdict


# ------------------------------------------------------------------ live wiring (main)
COACH_TEMPERATURE = 0.7      # Stage 2 Coach is creative (strategize); Stage 1 Scout stays at config's 0.0 (grounded)


def _real_qwen() -> QwenCaller:
    """The live Qwen caller. The Coach (task "coach_*") runs hotter so it explores offense; the Scout ("scout_*") and
    the old single prompt ("loop_*") keep the config temperature (0.0), so the summary stays grounded."""
    from sf2.system2.qwen import chat

    def call(messages: List[Dict], task: str) -> str:
        if task.startswith("coach"):
            return chat(messages, task, temperature=COACH_TEMPERATURE)
        return chat(messages, task)
    return call


def _real_score(port: int, rom: Optional[str]) -> ScoreFn:
    """The offline replay scorer: one round dir -> its score (sf2.system1.screen_replay via a fresh headless Mesen)."""
    from sf2.eval.runner import open_fight
    from sf2.system1.screen_replay import score_round

    def score(round_dir: str) -> Optional[Dict]:
        with open(os.path.join(round_dir, os.pardir, "run.json")) as f:
            run = json.load(f)
        me = run.get("me", ME)
        with open(run["state"]["path"], "rb") as f:
            state = f.read()
        with open_fight(me, run["opp"], port, state=state) as (b, _state):
            return score_round(b, state, round_dir, me, run["opp"])
    return score


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--me", default=ME, choices=SUPPORTED_ME,
                    help="the character to play AS (default chunli; only characters with a move menu in sf2.moves_free)")
    ap.add_argument("--opp", default="honda")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--book", default=os.path.join("lessons", "book.json"))
    ap.add_argument("--carry", default=None,
                    help="start from this saved registry (learn ON TOP of it) instead of seeding from the book; "
                         "if the file is absent, fall back to the book seed")
    ap.add_argument("--save-registry", dest="save_registry", default=None,
                    help="after the run, write the final registry here so the next round can --carry it")
    ap.add_argument("--watch", action="store_true", help="open a VISIBLE Mesen window to watch the match (default: headless)")
    ap.add_argument("--speed", type=int, default=100, help="--watch emulation speed percent (e.g. 100, 150)")
    ap.add_argument("--cat-advisor", default=os.path.join("runs", "text_laya", "cat_v3"),
                    help="round-1 CATEGORY checkpoint")
    ap.add_argument("--move-advisor", default=os.path.join("runs", "text_laya", "move_v2"),
                    help="round-2 MOVE checkpoint (point BOTH flags at junk/models/advice_v2 to compare the old "
                         "single model)")
    ap.add_argument("--qwen-mode", dest="qwen_mode", default="two", choices=("one", "two"),
                    help="two (default): Stage 1 Scout summarizes, Stage 2 Coach strategizes (escalate when losing); "
                         "one: the OLD single prompt, kept as the A/B fallback")
    ap.add_argument("--policy", default="rules", choices=("rules", "table", "hybrid", "quorum"),
                    help="rules (default): text-laya + short memory + Qwen; table: the self-learning value table "
                         "decides alone (no Qwen/text-laya); hybrid: text-laya plays and the value table OVERRIDES "
                         "where it is confident a move is clearly good (text-laya + table, the owner's pick)")
    ap.add_argument("--carry-table", dest="carry_table", default=None,
                    help="policy table: start the value table from this JSON and keep GROWING it (pass again to resume)")
    ap.add_argument("--save-table", dest="save_table", default=None,
                    help="policy table: write the value table here after the run")
    ap.add_argument("--explore", dest="explore_rate", type=float, default=0.1,
                    help="hybrid: how often to try an under-sampled move (table exploration rate; higher = more "
                         "exploration, richer table, lower win-rate during collection)")
    ap.add_argument("--explore-tries", dest="explore_tries", type=int, default=VT.MIN_TRIES,
                    help="hybrid: a move with fewer than this many tries in a cell still counts as under-sampled "
                         "(the exploration COVERAGE target). Raise it to re-open exploration in cells that are "
                         "already saturated past the default -- the lever when --explore alone no longer fires.")
    ap.add_argument("--quorum-config", dest="quorum_config", default=None,
                    help="policy quorum: a QuorumConfig JSON (sf2/quorum/config.py; the genome scripts/quorum_evolve.py tunes)")
    ap.add_argument("--quorum-mode", dest="quorum_mode", default=None, choices=("shadow", "candidates", "vote"),
                    help="policy quorum: override the config's mode (shadow = log only, candidates, vote)")
    ap.add_argument("--quorum-qwen", dest="quorum_qwen", action="store_true",
                    help="policy quorum, mode vote: send split votes to Qwen instead of falling back to text-laya")
    ap.add_argument("--carry-quorum", dest="carry_quorum", default=None,
                    help="policy quorum: start the voter-reliability state from this JSON")
    ap.add_argument("--save-quorum", dest="save_quorum", default=None,
                    help="policy quorum: write the voter-reliability state here after the run")
    ap.add_argument("--horizon", dest="credit_horizon", type=int, default=0,
                    help="table/hybrid credit: n-step horizon (backlog B1). >0 credits a decision with the discounted "
                         "return over the next N decisions (fixes setup-move myopia); 0 = one-step reward (default)")
    ap.add_argument("--gamma", dest="credit_gamma", type=float, default=1.0,
                    help="table/hybrid credit: discount for the n-step return (only used with --horizon>0)")
    ap.add_argument("--no-learn", dest="no_learn", action="store_true",
                    help="frozen EVAL: play the carried table but do NOT credit/update it (use with --explore 0)")
    ap.add_argument("--shared-text-laya", action="store_true")
    ap.add_argument("--no-score", action="store_true", help="score rounds from the screen only (skip the offline replay)")
    ap.add_argument("--port", type=int, default=PORTS["system1"][0] + PORTS["system1"][1] - 1)
    ap.add_argument("--replay-port", type=int, default=PORTS["replay"][0] + PORTS["replay"][1] - 1)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--out", default=None)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    from sf2.system1.advisor import Advisor
    from sf2.system1.screen_emu import open_screen

    me = args.me
    state_path = os.path.join("states", "p1_%s_vs_%s.state" % (me, args.opp))
    with open(state_path, "rb") as f:
        state = f.read()
    state_id = {"path": state_path, "sha256": hashlib.sha256(state).hexdigest()}
    out = args.out or os.path.join("rollouts", "loop_screen", "%s_%s_vs_%s" % (time.strftime("%Y%m%d_%H%M%S"), me,
                                                                               args.opp))
    os.makedirs(out, exist_ok=True)
    table_in = None
    if args.policy in ("table", "hybrid", "quorum") and args.carry_table and os.path.exists(args.carry_table):
        with open(args.carry_table) as f:
            table_in = json.load(f)
    qcfg = QuorumConfig.load(args.quorum_config) if args.quorum_config else QuorumConfig()
    if args.quorum_mode:
        qcfg.mode = args.quorum_mode
    if args.quorum_qwen:
        qcfg.qwen = True
    qcfg.validate()
    qstate = None
    if args.policy == "quorum" and args.carry_quorum and os.path.exists(args.carry_quorum):
        with open(args.carry_quorum) as f:
            qstate = json.load(f)
    if args.policy == "table":
        seed_lines, seed_source = [], "table"
    else:                                                    # rules AND hybrid seed text-laya's short memory
        seed_lines, seed_source = starting_registry(args.carry, args.opp, args.book, me)
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump({"arm": args.policy, "me": me, "opp": args.opp, "games": args.games, "rounds": args.rounds,
                   "seed": args.seed, "state": state_id, "book": args.book, "policy": args.policy,
                   "carry": args.carry, "save_registry": args.save_registry, "seed_source": seed_source,
                   "cat_advisor": args.cat_advisor, "move_advisor": args.move_advisor,
                   "qwen": "off" if args.policy == "table" else "on", "qwen_mode": args.qwen_mode,
                   "quorum": qcfg.to_dict() if args.policy == "quorum" else None}, f, indent=1)
    shared = {"shared": True} if args.shared_text_laya else {}
    score_fn = None if args.no_score else _real_score(args.replay_port, args.rom)
    if args.policy == "table":
        # the table decides every move and is credited from outcomes -- text-laya advisors and Qwen are NOT used
        with open_screen(args.port, args.rom, show_window=args.watch, speed=args.speed) as emu:
            verdict = run_loop(args.opp, None, None, lambda *a, **k: "", games=args.games, rounds=args.rounds,
                               seed_lines=seed_lines, out=out, play_round_fn=play_screen_round,
                               state=state, state_id=state_id, emu=emu, score_fn=score_fn, seed_rng=args.seed, me=me,
                               policy="table", table=table_in, save_table=args.save_table)
    else:
        # rules OR hybrid: text-laya plays (two Advisor instances). hybrid also carries/credits the value table.
        with Advisor(args.cat_advisor, **shared) as cat_advisor, Advisor(args.move_advisor, **shared) as move_advisor, \
                open_screen(args.port, args.rom, show_window=args.watch, speed=args.speed) as emu:
            qwen = _real_qwen()
            qwen_pick = None
            if args.policy == "quorum" and qcfg.qwen:            # System 2 on split votes (sf2/quorum/escalate.py)
                from sf2.quorum.escalate import make_qwen_pick
                qwen_pick = make_qwen_pick(qwen)
            verdict = run_loop(args.opp, cat_advisor, move_advisor, qwen, games=args.games, rounds=args.rounds,
                               seed_lines=seed_lines, out=out, play_round_fn=play_screen_round,
                               state=state, state_id=state_id, emu=emu, score_fn=score_fn, seed_rng=args.seed, me=me,
                               qwen_mode=args.qwen_mode, policy=args.policy, table=table_in, save_table=args.save_table,
                               explore_rate=args.explore_rate, explore_tries=args.explore_tries,
                               quorum=qcfg, quorum_state=qstate, save_quorum=args.save_quorum, qwen_pick=qwen_pick,
                               credit_horizon=args.credit_horizon, credit_gamma=args.credit_gamma,
                               no_learn=args.no_learn)
    if args.policy in ("table", "hybrid", "quorum") and args.save_table:
        print("saved table", args.save_table)
    if args.save_registry:
        save_registry(verdict["registry_end"], args.save_registry)
        print("saved registry", args.save_registry)
    print(json.dumps(verdict, indent=1))
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
