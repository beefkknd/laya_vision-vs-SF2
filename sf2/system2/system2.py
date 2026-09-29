"""System 2: Qwen reads System 1's game log and writes the two memories (sf2/system2/memory.py).

    review(me, logs)            rewrite the long-term playbook: lessons about me that hold across opponents
    populate(me, opp, logs)     write the short memory for the next games against ``opp`` (it goes into laya's prompt)

Qwen sees a digest of the log (move x range results, the opponent's habits, what hit him while he attacked or jumped,
game results), never the raw thousands of lines. Each lesson it writes names the fact it rests on (``claim``), and the
evidence is computed HERE from the log, never taken from Qwen: a lesson the numbers do not support is rejected, with the
reason logged. If Qwen's reply is unusable twice, the previous memory is kept.
"""
import json
import re
from typing import Dict, List, Optional, Tuple

from .memory import KINDS, check
from .qwen import chat, json_reply
from .prompts import (CLAIMS, MAX_PLAYBOOK, MAX_PROMPT_LESSONS, MAX_TEXT, MIN_TRIES, messages, populate_prompt,
                      review_prompt)
from ..vocab import RANGES
from ..data.vs_sweep import actions

# ------------------------------------------------------------------------------------------------ evidence (the truth)
def evidence(acts: List[Dict], claim: str, action: Optional[str], rng: Optional[str]) -> Dict:
    """tries / count / rate / refs for a claim, computed from the log actions."""
    kind, _, state = claim.partition(":")
    pool = [a for a in acts if rng in (None, a["range"])]
    if kind == "habit":
        tries, hits = pool, [a for a in pool if a["opp_state"] == state]
    else:
        tries = [a for a in pool if a["kind"] == "attack" and a["action"] == action
                 and (kind != "counter" or a["opp_state"] == state)]
        test = {"lands": lambda a: a["actual"] == "hit", "counter": lambda a: a["actual"] == "hit",
                "whiffs": lambda a: a["actual"] == "whiff", "blocked": lambda a: a["actual"] == "blocked",
                "punished": lambda a: a["i_was_hit"]}[kind]
        hits = [a for a in tries if test(a)]
    return {"tries": len(tries), "count": len(hits), "rate": round(len(hits) / len(tries), 2) if tries else 0.0,
            "refs": ["%s/g%df%d" % (a.get("log", ""), a["game"], a["frame"]) for a in hits[:5]]}


def supported(les: Dict, ev: Dict) -> Optional[str]:
    """Why the numbers do not back this lesson (None: they do)."""
    kind, claim, rate = les["kind"], les["claim"], ev["rate"]
    if ev["tries"] < MIN_TRIES:
        return "only %d tries" % ev["tries"]
    if kind == "use_more" and not (claim in ("lands", "counter:attack", "counter:jump") and rate >= 0.4):
        return "use_more needs a lands/counter claim at >= 40%% (is %s %.0f%%)" % (claim, 100 * rate)
    if kind == "avoid" and not ((claim == "lands" and rate <= 0.35) or (claim == "whiffs" and rate >= 0.5)
                                or (claim == "punished" and rate >= 0.25) or (claim == "blocked" and rate >= 0.3)):
        return "avoid is not backed: %s %.0f%%" % (claim, 100 * rate)
    if kind == "opponent_habit" and not (claim.startswith("habit:") and rate >= 0.15):
        return "habit needs a habit claim at >= 15%% (is %s %.0f%%)" % (claim, 100 * rate)
    if kind == "counter" and not claim.startswith("counter:"):
        return "counter needs a counter claim"
    return None


# ------------------------------------------------------------------------------------------------ digest (what Qwen reads)
def fits_laya(mem) -> bool:
    """A stored memory still meets today's rules (e.g. one kept from an earlier session or written by an older
    builder): every lesson names a claim, fits laya's prompt (short, no numbers) and no move or claim repeats."""
    seen = set()
    for les in (mem or {}).get("lessons", []):
        text = les.get("text") or ""
        key = (les.get("action"), les.get("kind")) if les.get("action") else (les.get("claim"), les.get("range"))
        if les.get("claim") not in CLAIMS or len(text) > MAX_TEXT or re.search(r"\d", text) or key in seen:
            return False
        seen.add(key)
    return bool(mem and mem.get("lessons"))


OWN_MIN_TRIES = 6          # own-move claims use this opponent's games from this many tries on, else all games


def evidence_for(les: Dict, acts: List[Dict], fallback: Optional[List[Dict]]) -> Dict:
    """Habit / counter claims are about the opponent: only his games. Claims about my own moves use his games once
    they hold OWN_MIN_TRIES tries, else every game so far (``fallback``), marked scope 'all opponents'."""
    ev = evidence(acts, les["claim"], les["action"], les["range"])
    if fallback is None or les["claim"].startswith(("habit:", "counter:")) or ev["tries"] >= OWN_MIN_TRIES:
        return dict(ev, scope="this opponent" if fallback is not None else "all games")
    return dict(evidence(fallback, les["claim"], les["action"], les["range"]), scope="all opponents")


def vet(lessons: List[Dict], acts: List[Dict], limit: int,
        fallback: Optional[List[Dict]] = None) -> Tuple[List[Dict], List[Dict]]:
    """Qwen's lessons, in its order, that fit laya's prompt and the log: (kept with evidence, rejected with why).
    Rejected: unknown kind / claim / range, too long, numbers in the text, not backed by the log, or a second
    lesson on a move already covered (one lesson per move and kind; laya gets no repeats)."""
    kept, rejected, seen = [], [], set()
    for les in lessons:
        les = {k: les.get(k) for k in ("text", "kind", "action", "range", "claim")}
        text = les["text"] if isinstance(les["text"], str) else ""
        key = (les["action"], les["kind"]) if les["action"] else (les["claim"], les["range"])
        if les["claim"] not in CLAIMS or les["kind"] not in KINDS or les["range"] not in RANGES + (None,):
            why = "unknown kind / claim / range"
        elif not text or len(text) > MAX_TEXT:
            why = "text must be 1-%d characters for laya's prompt (is %d)" % (MAX_TEXT, len(text))
        elif re.search(r"\d", text):
            why = "no numbers in the text: laya cannot use them (the evidence carries them)"
        elif key in seen:
            why = "repeats a lesson already given for %s" % (les["action"] or les["claim"])
        else:
            ev = evidence_for(les, acts, fallback)
            why = supported(les, ev)
            if not why:
                kept.append(dict(les, evidence=ev))
                seen.add(key)
                continue
            rejected.append({"lesson": les, "why": why, "evidence": ev})
            continue
        rejected.append({"lesson": les, "why": why})
    return kept[:limit], rejected


# ------------------------------------------------------------------------------------------------ Qwen
def _ask(me: str, task: str, prompt: str, acts: List[Dict], limit: int,
         fallback: Optional[List[Dict]] = None) -> Tuple[Optional[List[Dict]], Dict]:
    """Qwen's lessons with verified evidence (None if unusable twice), plus a report of what was kept / rejected."""
    msgs = messages(me, prompt)
    report: Dict = {"task": task, "attempts": []}
    for _ in range(2):
        try:
            reply = json_reply(chat(msgs, task))
        except Exception as e:  # noqa: BLE001 - a bad reply is retried once, then the old memory stays
            report["attempts"].append({"error": "%s: %s" % (type(e).__name__, e)})
            msgs.append({"role": "user", "content": "Your reply was not usable (%s). Reply with the JSON only." % e})
            continue
        kept, rejected = vet(reply.get("lessons", []), acts, limit, fallback)
        report["attempts"].append({"notes": reply.get("notes"), "kept": len(kept), "rejected": rejected})
        if kept and not check({"lessons": kept}, list(actions(me))):
            report["notes"] = reply.get("notes")
            return kept, report
        msgs.append({"role": "assistant", "content": json.dumps(reply)})
        msgs.append({"role": "user", "content": "None of those lessons held up against the log: %s. Write lessons "
                     "the digest's numbers support." % json.dumps(rejected)[:3000]})
    return None, report


def review(me: str, playbook: Optional[Dict], by_opp: Dict[str, Tuple[List[Dict], List[Dict]]]):
    """New playbook lessons from every opponent's log so far: (lessons or None, report)."""
    everything = [a for acts, _ in by_opp.values() for a in acts]
    return _ask(me, "review_%s" % me, review_prompt(me, playbook, by_opp), everything, MAX_PLAYBOOK)


def populate(me: str, opp: str, playbook: Optional[Dict], vs_opp: Tuple[List[Dict], List[Dict]],
             everything: List[Dict], recent: Optional[List[Dict]] = None, current: Optional[Dict] = None,
             what: str = "the last round"):
    """The short memory for the next games against ``opp``: (lessons or None, report). With ``recent`` (the actions of
    the round or game just played) and ``current`` (the short memory it was played with), Qwen REVISES the memory from
    what just happened instead of rewriting it from the lifetime totals alone."""
    prompt = populate_prompt(me, opp, playbook, vs_opp, everything, recent, current, what)
    return _ask(me, "populate_%s_vs_%s" % (me, opp), prompt, vs_opp[0], MAX_PROMPT_LESSONS, fallback=everything)
