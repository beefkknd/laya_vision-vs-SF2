"""G3: what Qwen observes, with NO RAM in play - the screen record rebuilt into the exact row shape the lesson code
consumes (sf2.system2.lessons.condition_evidence / cause, sf2.system2.character_prompt.threats / if_you_see).

The problem (README G3): the lesson evidence was RAM end to end (sf2.system1.game_log.action_entry): damage from RAM
life drops, his move from sf2.system1.opp_moves, hit/whiff/blocked and round results all from RAM. None of that may be
read while a game is played. This adapter rebuilds the SAME fields from the screen-only round record
(sf2.system1.loop_runner.play_round's decisions.jsonl) plus, for the round result and hp only, the OFFLINE replay
(sf2.system1.screen_replay via scripts/replay_score.py - RAM read after play, the allowed path). It reads no RAM itself.

Per decision (the fields lessons / character_prompt read, and where each comes from):
    range          screen  - the reader's distance (decision["situation"][0] / moment.dx)
    action         screen  - the move text laya played (the pick)
    opp_air        screen  - his sprite was in the air (moment.his_air)
    opp_state      screen  - mapped from the reader's label so opp_doing(row) == the reader's "doing" word, below
    dealt / taken  screen  - HEALTH-BAR drops across the decision (his / my drawn hp now minus at the next decision)
    kind           screen  - movement / defense / attack, from the move's menu category
    actual         screen  - None: the screen cannot see hit / whiff / blocked (cause() then reads a damaging
                             attack as "stuffed"); a frame-truth outcome would need the replay and is not wired
    opp_move       screen  - None: the screen cannot name a special (fireball aside); threat() falls back to opp_air
    opp_shot       screen  - a fireball was coming at her at the decision (moment.fireball)
    opp_reaction   screen  - (): the screen gives no state SEQUENCE across the window; threat() uses opp_air instead
Per round (frame truth, from the OFFLINE replay):
    result, hp, dealt, taken  - sf2.system1.screen_replay.score_round (RAM, after play); without a replay the round
                                falls back to the screen: result "unknown", dealt / taken summed from the decisions.

The one known gap (reported, owner/orchestrator to resolve): book / seed lines name a generic "throw", while the
two-stage menu names "throw_F+hp" / "throw_F+mp"; a "throw" line is inert in the runner until the vocabularies are
reconciled. Moves named in both vocabularies (c.mk, lightning_legs, cl.hp, throw_F+hp, ...) flow end to end.
"""
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

from ..system1.action_menu import category_of

# the reader's "doing" word (sf2.system1.screen_words.Moment.doing) -> (opp_state, opp_air) so that
# sf2.system1.advice.opp_doing(row) reproduces the same word the screen already decided.
DOING_STATE: Dict[str, Tuple[str, bool]] = {
    "jumping": ("jump", True), "crouching": ("crouch", False), "attacking": ("attack", False),
    "stunned": ("hit_stun", False), "standing": ("stand", False),
}
# the menu category -> the game-log kind cause() keys on (sf2.system1.game_log.action_entry).
KIND = {"move": "movement", "block": "defense"}


def kind_of(action: str) -> str:
    return KIND.get(category_of(action), "attack")


def _moment(dec: Dict) -> Dict:
    return dec.get("moment", {})


def decision_row(game: int, me: str, opp: str, dec: Dict, nxt: Optional[Dict]) -> Dict:
    """One decision as a lesson-shaped row. ``nxt``: the next moment-like dict ({"my_life", "his_life"}) the health
    drops are measured to (the next decision, or the round-end life from the replay); None -> no drop (0)."""
    m = _moment(dec)
    doing = m.get("doing", "standing")
    opp_state, opp_air = DOING_STATE.get(doing, ("stand", bool(m.get("his_air"))))
    my_now, his_now = int(m.get("my_life", 0)), int(m.get("his_life", 0))
    my_next = int(nxt.get("my_life", my_now)) if nxt else my_now
    his_next = int(nxt.get("his_life", his_now)) if nxt else his_now
    rng = (dec.get("situation") or [None])[0]
    return {
        "game": game, "frame": dec.get("k"), "me": me, "opp": opp,
        "side": m.get("side"), "gap": abs(int(m.get("dx", 0))), "range": rng,
        "my_life": my_now, "opp_life": his_now,
        "opp_state": opp_state, "opp_air": opp_air,
        "action": dec.get("action"), "kind": kind_of(dec.get("action")) if dec.get("action") else "attack",
        "actual": None,                                   # screen cannot see hit / whiff / blocked
        "dealt": max(0, his_now - his_next), "taken": max(0, my_now - my_next),
        "opp_reaction": [], "opp_move": None, "opp_shot": bool(m.get("fireball")),
        "my_life_after": my_next, "opp_life_after": his_next,
        "follows_rule": dec.get("follows_rule"), "rule": dec.get("rule"),
    }


def round_result(my_life_end: int, his_life_end: int) -> str:
    """Win / loss / draw from the two drawn HEALTH BARS at round over (life points, 0 = empty bar). A KO is the empty
    bar: her bar empty -> loss, his bar empty -> win (double KO or both alive with equal life -> draw). Otherwise the
    fighter with more life wins. Never "unknown" - the screen always draws both bars."""
    if his_life_end <= 0 < my_life_end:
        return "win"
    if my_life_end <= 0 < his_life_end:
        return "loss"
    if my_life_end > his_life_end:
        return "win"
    if my_life_end < his_life_end:
        return "loss"
    return "draw"


def round_evidence(game: int, me: str, opp: str, decisions: Sequence[Dict],
                   replay: Optional[Dict] = None) -> Tuple[List[Dict], Dict]:
    """(the per-decision rows, the round summary) for one round. ``decisions``: the round's decision records
    (loop_runner decisions.jsonl). ``replay``: scripts/replay_score.py's per-round score (result, hp, dealt, taken,
    my_life_end, opp_life_end) - the frame-truth round fields; None falls back to the screen."""
    decisions = list(decisions)
    end = None
    if replay is not None:
        end = {"my_life": replay.get("my_life_end", 0), "his_life": replay.get("opp_life_end", 0)}
    rows: List[Dict] = []
    for i, d in enumerate(decisions):
        nxt = _moment(decisions[i + 1]) if i + 1 < len(decisions) else end
        rows.append(decision_row(game, me, opp, d, nxt))
    s_dealt = sum(r["dealt"] for r in rows)
    s_taken = sum(r["taken"] for r in rows)
    if replay is not None:
        summary = {"game": game, "result": replay.get("result", "unknown"),
                   "dealt": replay.get("dealt", s_dealt), "taken": replay.get("taken", s_taken),
                   "hp": replay.get("hp", s_dealt - s_taken), "source": "replay", "decisions": len(rows)}
    else:
        # No RAM referee: decide the result from the drawn health bars at the last decision (the bars the screen read
        # closest to round over - his_life_end 0 means his bar went empty, a KO). hp / dealt / taken stay the screen sums.
        result = round_result(rows[-1]["my_life_after"], rows[-1]["opp_life_after"]) if rows else "unknown"
        summary = {"game": game, "result": result, "dealt": s_dealt, "taken": s_taken,
                   "hp": s_dealt - s_taken, "source": "screen", "decisions": len(rows)}
    return rows, summary


def read_decisions(round_dir: str) -> List[Dict]:
    """The decision records a round wrote (loop_runner.play_round <round>/decisions.jsonl)."""
    path = os.path.join(round_dir, "decisions.jsonl")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def game_evidence(game: int, me: str, opp: str, round_dirs: Sequence[str],
                  replays: Optional[Sequence[Optional[Dict]]] = None) -> Tuple[List[Dict], List[Dict]]:
    """(all decision rows, all round summaries) for one game = its rounds. ``round_dirs``: each round's output dir;
    ``replays``: the matching per-round replay scores (or None each)."""
    replays = list(replays) if replays is not None else [None] * len(round_dirs)
    rows: List[Dict] = []
    summaries: List[Dict] = []
    for rd, rep in zip(round_dirs, replays):
        r, s = round_evidence(game, me, opp, read_decisions(rd), rep)
        rows += r
        summaries.append(s)
    return rows, summaries
