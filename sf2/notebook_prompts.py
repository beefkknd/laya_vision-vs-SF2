"""System 2's round reflection prompt (one Qwen call per round): read the round's facts, update the notebook a
little, write the next round's plan. The runtime (scripts/notebook_run.py) and the prompt tests build it here only.
"""
import json
from typing import Dict, List, Sequence

from .notebook import MAX_ADD, MAX_CHARS, MAX_LINES, MAX_PLAN, MAX_PLAN_CHARS, MAX_REMOVE
from .vs_sweep import actions


def rules(me: str) -> str:
    return (
        "You are the coach inside a novice Street Fighter II player (%s, SNES) who is learning against the CPU. "
        "After every round, win or lose, you look at what happened and think like a player improving: what did I do "
        "right and wrong in attack and in defence, what did he do that worked, what surprised me.\n"
        "You keep a NOTEBOOK: self (what I am good and bad at), opponent (this opponent: what he does, what works on "
        "him, what he punishes), questions (what I do not know yet). It is long-term knowledge: change it only when "
        "the facts clearly say so: per round add at most %d lines and remove at most %d, keep lines that still hold. "
        "Lines "
        "are plain sentences of at most %d characters, at most %d per section.\n"
        "Then you write the PLAN for the next round: 1-%d short orders the move-picker follows exactly as words, "
        "e.g. \"use more c.mk at mid\", \"avoid sweep up close\", \"when he jumps, use spinning_bird_kick\", "
        "\"try throw up close\". Use the move names exactly as listed; no numbers; at most %d characters each. A "
        "line starting with \"try \" is an experiment: next round you will see whether it worked.\n"
        "Moves: %s. Ranges: close, mid, far. Judge moves by hit points (damage dealt minus taken), not by risk "
        "alone: a move that is punished but deals more than it costs is good.\n"
        "Reply with JSON only: {\"reflection\": \"2-3 sentences\", \"notebook\": {\"self\": [...], \"opponent\": "
        "[...], \"questions\": [...]}, \"plan\": [...]}"
        % (me, MAX_ADD, MAX_REMOVE, MAX_CHARS, MAX_LINES, MAX_PLAN, MAX_PLAN_CHARS, ", ".join(actions(me))))


def reflect_prompt(me: str, opp: str, notebook: Dict, facts_text: str, allowed_tries: int) -> str:
    current = {"self": notebook.get("self", []), "opponent": notebook.get("opponents", {}).get(opp, []),
               "questions": notebook.get("questions", [])}
    seen = "you have met %s before" % opp if current["opponent"] else "%s is new to you" % opp
    return ("Opponent: %s (%s).\nYOUR NOTEBOOK NOW:\n%s\n\nTHE ROUND THAT JUST ENDED (counted by the game):\n%s\n\n"
            "Next round you may run at most %d \"try\" experiment%s (more when you are losing).\n"
            "Before you answer: add at most %d notebook lines, remove at most %d; one range per plan line; plan "
            "lines use exact move names, no numbers, at most %d characters. JSON only."
            % (opp, seen, json.dumps(current, indent=1), facts_text, allowed_tries, "" if allowed_tries == 1 else "s",
               MAX_ADD, MAX_REMOVE, MAX_PLAN_CHARS))


def messages(me: str, prompt: str) -> List[Dict]:
    return [{"role": "system", "content": rules(me)}, {"role": "user", "content": prompt}]
