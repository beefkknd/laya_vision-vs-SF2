"""Everything System 2 sends Qwen, in one place: the rules (system prompt), the log digest, the report of what System
1 did with each lesson, and the task prompts. The runtime (sf2/system2.py, via scripts/learn_loop.py) and the prompt
tests (scripts/check_system2_prompts.py) both build their chats here, so what is tested is exactly what runs."""
import collections
from typing import Dict, List, Optional, Tuple

from .memory import KINDS, MAX_PROMPT_LESSONS
from .vocab import RANGES
from .vs_sweep import actions

MAX_PLAYBOOK = 8
MIN_TRIES = 3
MAX_TEXT = 60              # characters per lesson: it goes into laya's prompt
CLAIMS = ("lands", "whiffs", "blocked", "punished", "habit:attack", "habit:jump", "habit:guard", "counter:attack",
          "counter:jump")


def digest(acts: List[Dict], games: List[Dict], title: str) -> str:
    lines = ["## %s: %d rounds (%s), %d actions" % (
        title, len(games), ", ".join("%s %d" % kv for kv in collections.Counter(g["result"] for g in games).items()),
        len(acts))]
    if games:
        lines.append("damage per round: dealt %.0f, taken %.0f" % (sum(g["dealt"] for g in games) / len(games),
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


def rules(me: str) -> str:
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


def messages(me: str, prompt: str) -> List[Dict]:
    """The chat System 2 sends Qwen: the rules for ``me``, then the task prompt (loop and prompt tests share it)."""
    return [{"role": "system", "content": rules(me)}, {"role": "user", "content": prompt}]


# said last, right before Qwen answers: the numbers in the digest above pull its wording toward them
FORMAT_REMINDER = ("Before you answer: every lesson text is plain words for laya - at most %d characters, NO numbers "
                   "or percentages (the evidence carries them), the move names exactly as listed. JSON only."
                   % MAX_TEXT)


def review_prompt(me: str, playbook: Optional[Dict], by_opp: Dict[str, Tuple[List[Dict], List[Dict]]]) -> str:
    everything = [a for acts, _ in by_opp.values() for a in acts]
    return ("Rewrite %s's long-term PLAYBOOK: up to %d lessons about %s's own moves and approach that hold across "
            "opponents (keep the good ones, fix or drop what the new games contradict).\n\nCurrent playbook:\n%s\n\n"
            "%s\n\n%s" % (me, MAX_PLAYBOOK, me, show(playbook),
                          digest(everything, [g for _, gs in by_opp.values() for g in gs], "all opponents"),
                          "\n\n".join(digest(a, g, "vs " + o) for o, (a, g) in sorted(by_opp.items())))
            + "\n\n" + FORMAT_REMINDER)


def followed(recent: List[Dict], current: Optional[Dict]) -> List[str]:
    """For each lesson in the current short memory: did System 1 act on it in the recent actions, and how did it go."""
    out = []
    for les in (current or {}).get("lessons", []):
        act, rng = les.get("action"), les.get("range")
        if not act:
            continue
        used = [a for a in recent if a["action"] == act and rng in (None, a["range"])]
        where = "%s%s" % (act, " at " + rng if rng else "")
        if les["kind"] == "avoid":
            verdict = "FOLLOWED" if not used else "IGNORED (used anyway)"
            out.append("- %s \"%s\": %s used %d times%s" % (verdict, les["text"], where, len(used), _how(used)))
        else:
            verdict = "IGNORED (never used)" if not used else "FOLLOWED"
            out.append("- %s \"%s\": %s used %d times%s" % (verdict, les["text"], where, len(used), _how(used)))
    return out


def _how(used: List[Dict]) -> str:
    if not used:
        return ""
    c = collections.Counter(a["actual"] for a in used)
    return " (%s; punished %d)" % (", ".join("%s %d" % kv for kv in c.most_common()), sum(a["i_was_hit"] for a in used))


def populate_prompt(me: str, opp: str, playbook: Optional[Dict], vs_opp: Tuple[List[Dict], List[Dict]],
                    everything: List[Dict], recent: Optional[List[Dict]] = None, current: Optional[Dict] = None,
                    what: str = "the last round") -> str:
    acts, games = vs_opp
    past = digest(acts, games, "all games vs " + opp) if acts else (
        "No games against %s yet. So NO lesson about him: no opponent_habit or counter lesson, nothing starting "
        "\"when he ...\" (the playbook's counter lessons were learned against other opponents: do not copy them). "
        "Write only lessons about %s's own moves (claims lands, whiffs, blocked, punished) from the playbook and the "
        "all-opponent digest." % (opp, me))
    task = ("Write the SHORT MEMORY for %s's next games against %s" % (me, opp)) if not current else (
        "REVISE the SHORT MEMORY %s played %s with against %s. %s just ended; see below what %s actually did with "
        "each lesson. Rules: a lesson marked IGNORED must NOT come back in the same words - reword it more directly "
        "in plain words (the move and the moment, still no numbers) or replace it with a move System 1 will use; keep a FOLLOWED lesson that "
        "still holds; drop or change one the recent play contradicts; add what the recent play shows is new"
        % (me, what, opp, what[0].upper() + what[1:], me))
    now = ""
    if recent:
        rep = followed(recent, current)
        banned = [line.split('"')[1] for line in rep if " IGNORED " in " " + line[2:]]
        now = "\n\n%s\n%s%s" % (
            digest(recent, [], what + " vs " + opp),
            ("What System 1 did with the current short memory:\n" + "\n".join(rep)) if rep else "",
            ("\nThese texts were IGNORED and may not appear again word for word: %s"
             % "; ".join('"%s"' % b for b in banned)) if banned else "")
    return ("%s: up to %d lessons, the most useful first; they go straight into laya's prompt and are the only memory "
            "it sees. Prefer what is specific to %s; one lesson per move; merge a move's ranges into one lesson when "
            "it holds at every range.\n\nCurrent short memory:\n%s\n\nPlaybook:\n%s%s\n\n%s\n\n%s\n\n%s" % (
                task, MAX_PROMPT_LESSONS, opp, show(current), show(playbook), now, past,
                digest(everything, [], "my moves against all opponents"), FORMAT_REMINDER))


def show(mem: Optional[Dict]) -> str:
    if not mem or not mem.get("lessons"):
        return "(empty)"
    return "\n".join("- %s  [%s %s: %d/%d]" % (x["text"], x["kind"], x.get("claim", ""), x["evidence"]["count"],
                                                x["evidence"]["tries"]) for x in mem["lessons"])
