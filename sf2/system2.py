"""System 2: Qwen reads System 1's game log and writes the two memories (sf2/memory.py).

    review(me, logs)            rewrite the long-term playbook: lessons about me that hold across opponents
    populate(me, opp, logs)     write the short memory for the next games against ``opp`` (it goes into laya's prompt)

Qwen sees a digest of the log (move x range results, the opponent's habits, what hit him while he attacked or jumped,
game results), never the raw thousands of lines. Each lesson it writes names the fact it rests on (``claim``), and the
evidence is computed HERE from the log, never taken from Qwen: a lesson the numbers do not support is rejected, with the
reason logged. If Qwen's reply is unusable twice, the previous memory is kept.
"""
import collections
import json
import re
from typing import Dict, List, Optional, Tuple

from .memory import KINDS, MAX_PROMPT_LESSONS, check
from .qwen import chat, json_reply
from .vs_sweep import actions

MAX_PLAYBOOK = 8
MIN_TRIES = 3
MAX_TEXT = 60              # characters per lesson: it goes into laya's prompt
RANGES = ("close", "mid", "far")
CLAIMS = ("lands", "whiffs", "blocked", "punished", "habit:attack", "habit:jump", "habit:guard", "counter:attack",
          "counter:jump")


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
def digest(acts: List[Dict], games: List[Dict], title: str) -> str:
    lines = ["## %s: %d games (%s), %d actions" % (
        title, len(games), ", ".join("%s %d" % kv for kv in collections.Counter(g["result"] for g in games).items()),
        len(acts))]
    if games:
        lines.append("damage per game: dealt %.0f, taken %.0f" % (sum(g["dealt"] for g in games) / len(games),
                                                                  sum(g["taken"] for g in games) / len(games)))
    moves = collections.defaultdict(list)
    for a in acts:
        if a["kind"] == "attack":
            moves[(a["action"], a["range"])].append(a)
    lines.append("move @ range: tries, lands%, whiffs%, blocked%, punished% (I was hit before my next turn)")
    for (m, r), g in sorted(moves.items(), key=lambda kv: -len(kv[1])):
        if len(g) >= MIN_TRIES:
            pct = lambda f: 100 * sum(1 for a in g if f(a)) / len(g)  # noqa: E731
            lines.append("  %s @ %s: %d, %.0f%%, %.0f%%, %.0f%%, %.0f%%" % (
                m, r, len(g), pct(lambda a: a["actual"] == "hit"), pct(lambda a: a["actual"] == "whiff"),
                pct(lambda a: a["actual"] == "blocked"), pct(lambda a: a["i_was_hit"])))
    lines.append("opponent at my turns, by range (share of turns):")
    for r in RANGES:
        g = [a for a in acts if a["range"] == r]
        if g:
            c = collections.Counter(a["opp_state"] for a in g)
            lines.append("  %s (%d turns): %s" % (r, len(g), ", ".join("%s %.0f%%" % (s, 100 * n / len(g))
                                                                       for s, n in c.most_common(4))))
    for state in ("attack", "jump"):
        g = [a for a in acts if a["kind"] == "attack" and a["opp_state"] == state]
        if g:
            c = collections.defaultdict(list)
            for a in g:
                c[a["action"]].append(a["actual"] == "hit")
            lines.append("when he was in %s: %s" % (state, ", ".join("%s lands %d/%d" % (m, sum(v), len(v))
                                                                   for m, v in sorted(c.items(), key=lambda kv: -len(kv[1])))))
    return "\n".join(lines)


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
def _rules(me: str) -> str:
    return (
        "You are System 2, the coach of an AI playing %s in Street Fighter II (SNES, World Warrior) against the CPU. "
        "System 1 is a vision model that picks one move every turn; your lessons steer it.\n"
        "Moves %s can do: %s.\nRanges (gap between the fighters): close < 55 px <= mid < 120 px <= far.\n"
        "System 1 is laya, a small vision model: it reads your lessons as plain text in its prompt, next to a RAM "
        "note, and knows the moves only by the exact names above. So write FOR IT: one short plain instruction per "
        "lesson (<= %d characters), the move name exactly as listed, no numbers or percentages (the evidence "
        "carries those), and never two lessons about the same move: when a lesson holds at every range, give range "
        "null and say so once, e.g. \"avoid spinning_bird_kick: he punishes it at every range\", "
        "\"use more lp up close\", \"when he jumps in, answer with spinning_bird_kick\".\n"
        "Every lesson must rest on numbers in the digest and name them:\n"
        "  kind: one of %s\n  action: one of the moves above, or null (habits)\n  range: close, mid, far, or null\n"
        "  claim: one of %s (lands/whiffs/blocked/punished: what my move did; habit:X: the opponent was in X at my "
        "turn; counter:X: my move landed while he was in X)\n"
        "Lessons the numbers do not support are thrown away, so do not guess. Reply with JSON only: "
        "{\"lessons\": [{\"text\", \"kind\", \"action\", \"range\", \"claim\"}], \"notes\": \"one line: what changed and why\"}"
        % (me, me, ", ".join(actions(me)), MAX_TEXT, ", ".join(KINDS), ", ".join(CLAIMS)))


def _ask(me: str, task: str, prompt: str, acts: List[Dict], limit: int,
         fallback: Optional[List[Dict]] = None) -> Tuple[Optional[List[Dict]], Dict]:
    """Qwen's lessons with verified evidence (None if unusable twice), plus a report of what was kept / rejected."""
    msgs = [{"role": "system", "content": _rules(me)}, {"role": "user", "content": prompt}]
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
    prompt = ("Rewrite %s's long-term PLAYBOOK: up to %d lessons about %s's own moves and approach that hold across "
              "opponents (keep the good ones, fix or drop what the new games contradict).\n\nCurrent playbook:\n%s\n\n"
              "%s\n\n%s" % (me, MAX_PLAYBOOK, me, _show(playbook),
                            digest(everything, [g for _, gs in by_opp.values() for g in gs], "all opponents"),
                            "\n\n".join(digest(a, g, "vs " + o) for o, (a, g) in sorted(by_opp.items()))))
    return _ask(me, "review_%s" % me, prompt, everything, MAX_PLAYBOOK)


def populate(me: str, opp: str, playbook: Optional[Dict], vs_opp: Tuple[List[Dict], List[Dict]],
             everything: List[Dict]):
    """The short memory for the next games against ``opp``: (lessons or None, report)."""
    acts, games = vs_opp
    past = digest(acts, games, "past games vs " + opp) if acts else "No games against %s yet: use the playbook and " \
        "the all-opponent digest for my own moves (habit and counter claims need games against him)." % opp
    prompt = ("Write the SHORT MEMORY for %s's next games against %s: up to %d lessons, the most useful first; they go "
              "straight into laya's prompt and are the only memory it sees. Prefer what is specific to %s; one lesson "
              "per move; merge a move's ranges into one lesson when it holds at every range.\n\nPlaybook:\n%s\n\n%s\n\n%s" % (
                  me, opp, MAX_PROMPT_LESSONS, opp, _show(playbook), past,
                  digest(everything, [], "my moves against all opponents")))
    return _ask(me, "populate_%s_vs_%s" % (me, opp), prompt, acts, MAX_PROMPT_LESSONS, fallback=everything)


def _show(mem: Optional[Dict]) -> str:
    if not mem or not mem.get("lessons"):
        return "(empty)"
    return "\n".join("- %s  [%s %s: %d/%d]" % (x["text"], x["kind"], x.get("claim", ""), x["evidence"]["count"],
                                                x["evidence"]["tries"]) for x in mem["lessons"])
