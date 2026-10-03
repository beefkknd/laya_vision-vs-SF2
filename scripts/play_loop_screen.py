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
from sf2.system1.action_menu import CATEGORIES, CATEGORY_ORDER
from sf2.system1.loop_runner import play_round as play_screen_round
from sf2.system2 import character_prompt, lessons as L, screen_evidence, seed_rules
from sf2.system2.lesson_prompt import streak

ME = "chunli"
MENU_MOVES: List[str] = [m for cat in CATEGORY_ORDER for m in CATEGORIES[cat]]   # the followable (two-stage) vocabulary
DELAY_MIN, DELAY_SPAN = 4, 40


# ------------------------------------------------------------------ seeding
def seed(opp: str, book: Optional[str], me: str = ME) -> L.Registry:
    """The starting short memory: the opponent's verified web/book lines (sf2.system2.seed_rules), [] without a book."""
    return seed_rules.seed_lessons(book, opp, me) if book else []


# ------------------------------------------------------------------ the Qwen update (System 2)
QwenCaller = Callable[[List[Dict], str], str]      # (messages, task) -> raw reply text


def ask_claims(opp: str, reg: L.Registry, rows: Sequence[Dict], last: Sequence[Dict], all_rounds: Sequence[Dict],
               last_rounds: Sequence[Dict], refused: Sequence[Dict], stable: Optional[str],
               ask_qwen: QwenCaller, me: str = ME) -> Tuple[List[Dict], List[str], Optional[str]]:
    """Qwen's claims for this game (sf2.system2.character_prompt); ([], [problem], None) if the call or parse fails."""
    from sf2.system2.qwen import json_reply
    msgs = character_prompt.messages(me, opp, reg, rows, last, all_rounds, last_rounds, MENU_MOVES, refused, stable)
    try:
        raw = ask_qwen(msgs, "loop_%s" % opp)
        claims, problems = character_prompt.parse_claims(json_reply(raw))
    except Exception as e:                       # Qwen down / cut off / not JSON: no claims this game
        return [], ["%s: %s" % (type(e).__name__, e)], None
    return claims, problems, raw


def update(reg: L.Registry, rows: List[Dict], game: int, game_hp: List[float], games: List[Dict],
           changed: List[bool], opp: str, last: List[Dict], all_rounds: List[Dict], last_rounds: List[Dict],
           refused: List[Dict], ask_qwen: QwenCaller, me: str = ME) -> Tuple[L.Registry, Dict]:
    """One System-2 step: review the registry on the evidence so far, ask Qwen, judge its claims. ``rows`` = all her
    decisions so far, ``last`` = last game's decisions, ``all_rounds`` / ``last_rounds`` = the round summaries.
    Returns the new registry and a trace record (claims, verdicts, the short-memory diff)."""
    before = L.in_play(reg)
    reg = L.review(reg, rows, game, game_hp)
    stable = streak(games, changed)
    claims, problems, raw = ask_claims(opp, reg, rows, last, all_rounds, last_rounds, refused, stable,
                                       ask_qwen, me)
    reg, outcome = L.propose(reg, claims, rows, game, moves=MENU_MOVES)
    after = L.in_play(reg)
    trace = {"game": game, "stable": stable, "claims": claims, "problems": problems,
             "outcome": [{"line": o.get("line", o.get("claim")), "state": o["state"], "why": o.get("why")}
                         for o in outcome],
             "in_play_before": before, "in_play_after": after,
             "added": [l for l in after if l not in before], "removed": [l for l in before if l not in after],
             "violations": L.violations(reg, rows), "reply": raw}
    new_refused = [o for o in outcome if o["state"] == "refused"]
    return reg, dict(trace, refused=new_refused)


# ------------------------------------------------------------------ the loop
PlayFn = Callable[..., Dict]        # like sf2.system1.loop_runner.play_round
ScoreFn = Callable[[str], Optional[Dict]]    # a round dir -> its offline replay score (result, hp, ...), or None


def run_loop(opp: str, cat_advisor, move_advisor, ask_qwen: QwenCaller, *, games: int, rounds: int,
             seed_lines: L.Registry, out: str, play_round_fn: PlayFn, state: bytes, state_id: Dict, emu,
             score_fn: Optional[ScoreFn] = None, reader=None, seed_rng: int = 0, me: str = ME, log=None) -> Dict:
    """Play ``games`` games, rotating the short memory through System 2 after each. Everything heavy is injected:
    ``cat_advisor`` / ``move_advisor`` (the two text laya checkpoints: round-1 category, round-2 move), ``ask_qwen``,
    ``play_round_fn`` (plays one round, writes its record), ``emu`` (a handle with ``new_round()``), ``score_fn`` (the
    offline replay, or None to score from the screen). Returns the verdict."""
    os.makedirs(out, exist_ok=True)
    trace = open(os.path.join(out, "trace.jsonl"), "w") if log is None else log
    reg: L.Registry = list(seed_lines)
    rng = random.Random(seed_rng)
    trace.write(json.dumps({"event": "seed", "opp": opp, "lines": L.in_play(reg),
                            "rules": [{"line": r["line"], "state": r["state"], "source": r.get("evidence", {}).get("source")}
                                      for r in reg]}) + "\n")
    trace.flush()
    all_rows: List[Dict] = []
    all_rounds: List[Dict] = []
    game_hp: List[float] = []
    games_wl: List[Dict] = []
    changed: List[bool] = []
    refused: List[Dict] = []
    for g in range(games):
        lines = L.in_play(reg)
        round_dirs, replays = [], []
        this_rows: List[Dict] = []
        this_rounds: List[Dict] = []
        for r in range(rounds):
            rd = os.path.join(out, "g%02d_r%d" % (g, r))
            delay = DELAY_MIN + rng.randrange(DELAY_SPAN)
            play_round_fn(emu, cat_advisor, move_advisor, me, opp, state, state_id, delay, lines, rd, reader=reader)
            emu = emu.new_round()
            replay = score_fn(rd) if score_fn else None
            decisions = screen_evidence.read_decisions(rd)
            drows, summary = screen_evidence.round_evidence(g, me, opp, decisions, replay)
            this_rows += drows
            this_rounds.append(summary)
            round_dirs.append(rd)
            replays.append(replay)
            for d, row in zip(decisions, drows):
                trace.write(json.dumps({"event": "decision", "game": g, "round": r, "k": d.get("k"),
                                        "words": d.get("advice_text"), "situation": d.get("situation"),
                                        "lines": d.get("lines"), "category": d.get("category"),
                                        "action": d.get("action"), "rule": d.get("rule"),
                                        "follows_rule": d.get("follows_rule"),
                                        "dealt": row["dealt"], "taken": row["taken"]}) + "\n")
            trace.write(json.dumps({"event": "round", "game": g, "round": r, "result": summary["result"],
                                    "hp": summary["hp"], "dealt": summary["dealt"], "taken": summary["taken"],
                                    "source": summary["source"]}) + "\n")
            trace.flush()
        all_rows += this_rows
        all_rounds += this_rounds
        game_hp.append(sum(s["dealt"] - s["taken"] for s in this_rounds) / max(1, len(this_rounds)))
        games_wl.append({"won": sum(s["result"] == "win" for s in this_rounds),
                         "lost": sum(s["result"] != "win" for s in this_rounds)})
        reg, qtrace = update(reg, all_rows, g, game_hp, games_wl, changed, opp, this_rows, all_rounds, this_rounds,
                             refused, ask_qwen, me)
        refused = qtrace.pop("refused")
        changed.append(bool(qtrace["added"] or qtrace["removed"]))
        trace.write(json.dumps(dict(qtrace, event="qwen")) + "\n")
        trace.flush()
    verdict = {"opp": opp, "games": games, "rounds": rounds, "seed_lines": [r["line"] for r in seed_lines],
               "in_play_end": L.in_play(reg), "registry_end": reg, "decisions": len(all_rows),
               "game_hp": game_hp, "games": games_wl, "violations": L.violations(reg, all_rows)}
    with open(os.path.join(out, "verdict.json"), "w") as f:
        json.dump(verdict, f, indent=1)
    if log is None:
        trace.close()
    return verdict


# ------------------------------------------------------------------ live wiring (main)
def _real_qwen() -> QwenCaller:
    from sf2.system2.qwen import chat
    return lambda messages, task: chat(messages, task)


def _real_score(port: int, rom: Optional[str]) -> ScoreFn:
    """The offline replay scorer: one round dir -> its score (sf2.system1.screen_replay via a fresh headless Mesen)."""
    from sf2.eval.runner import open_fight
    from sf2.system1.screen_replay import score_round

    def score(round_dir: str) -> Optional[Dict]:
        with open(os.path.join(round_dir, os.pardir, "run.json")) as f:
            run = json.load(f)
        with open(run["state"]["path"], "rb") as f:
            state = f.read()
        with open_fight(ME, run["opp"], port, state=state) as (b, _state):
            return score_round(b, state, round_dir, ME, run["opp"])
    return score


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opp", default="honda")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--book", default=os.path.join("lessons", "book.json"))
    ap.add_argument("--cat-advisor", default=os.path.join("runs", "text_laya", "cat_v1"),
                    help="round-1 CATEGORY checkpoint")
    ap.add_argument("--move-advisor", default=os.path.join("runs", "text_laya", "move_v1"),
                    help="round-2 MOVE checkpoint (point BOTH flags at runs/text_laya/advice_v2 to compare the old "
                         "single model)")
    ap.add_argument("--shared-text-laya", action="store_true")
    ap.add_argument("--no-score", action="store_true", help="score rounds from the screen only (skip the offline replay)")
    ap.add_argument("--port", type=int, default=PORTS["system1"][0] + PORTS["system1"][1] - 1)
    ap.add_argument("--replay-port", type=int, default=PORTS["replay"][0] + PORTS["replay"][1] - 1)
    ap.add_argument("--rom", default=os.environ.get("SF2_ROM"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from sf2.system1.advisor import Advisor
    from sf2.system1.screen_emu import open_screen

    state_path = os.path.join("states", "p1_%s_vs_%s.state" % (ME, args.opp))
    with open(state_path, "rb") as f:
        state = f.read()
    state_id = {"path": state_path, "sha256": hashlib.sha256(state).hexdigest()}
    out = args.out or os.path.join("rollouts", "loop_screen", "%s_%s" % (time.strftime("%Y%m%d_%H%M%S"), args.opp))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "run.json"), "w") as f:
        json.dump({"arm": "loop", "me": ME, "opp": args.opp, "games": args.games, "rounds": args.rounds,
                   "seed": args.seed, "state": state_id, "book": args.book,
                   "cat_advisor": args.cat_advisor, "move_advisor": args.move_advisor,
                   "qwen": "on"}, f, indent=1)
    shared = {"shared": True} if args.shared_text_laya else {}
    score_fn = None if args.no_score else _real_score(args.replay_port, args.rom)
    # Two Advisor instances - one per checkpoint. In shared mode each gets its own socket (shared_laya.socket_path keys
    # off the checkpoint), so this is two shared servers on different sockets; otherwise two helper subprocesses.
    with Advisor(args.cat_advisor, **shared) as cat_advisor, Advisor(args.move_advisor, **shared) as move_advisor, \
            open_screen(args.port, args.rom) as emu:
        verdict = run_loop(args.opp, cat_advisor, move_advisor, _real_qwen(), games=args.games, rounds=args.rounds,
                           seed_lines=seed(args.opp, args.book), out=out, play_round_fn=play_screen_round,
                           state=state, state_id=state_id, emu=emu, score_fn=score_fn, seed_rng=args.seed)
    print(json.dumps(verdict, indent=1))
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
