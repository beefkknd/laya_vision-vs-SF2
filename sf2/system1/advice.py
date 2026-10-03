"""Text laya's job: turn Qwen's short memory into one move. laya-vision watches the screen and rates the moves; text
laya reads the moment in words, the rated shortlist and the advice, and picks one.

The same functions build the text at training time and at play time (the words must be byte-identical):

    situation_text   "He is up close and jumping. My bar is half, his bar is high."
    option_text      "likely works" / "may work" / "likely fails" (laya-vision's score, in words), "walk in" (forward)
    advice_text      "Advice: use more lp up close; avoid sweep."
    question         the choice question over the shortlist

``answers`` is the label rule. What the advice is worth is not text laya's business (Qwen judges that from the games);
text laya must FOLLOW it faithfully:

  1. a lesson applies only when its conditions hold (distance: up close / at mid / at far; what he is doing: when he
     jumps / crouches / attacks / stands);
  2. an applying negative lesson ("avoid", "never", "stop", "less", "don't") rules its move out;
  3. an applying hard lesson ("always", "only") picks its move;
  4. an applying soft lesson ("use more", "prefer", "go for", "when he jumps, use") picks its move unless laya-vision
     says it likely fails right now;
  5. otherwise the move(s) laya-vision rates best, among those not ruled out; if every one likely fails, walk in.
Every answer this returns is correct; a training target spreads over all of them.
"""
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..moves_free import menu as _menu
from ..vocab import BARS, OPP_STATES, RANGE_WORDS, RANGES
from .action_menu import CATEGORIES, CATEGORY_ORDER, DEFAULT_MOVE, category_of

FORWARD = "forward"

# G4: the fireball condition in text laya's grammar. The TOKEN is the word "fireball": the screen words say the exact
# clause ``FIREBALL_CLAUSE`` when the reader sees a projectile coming (sf2.system1.screen_words), and a lesson's
# fireball condition is read off the same word (``FIRE`` below), so "block_low when a fireball comes at mid" parses.
FIREBALL_CLAUSE = "A fireball is coming."

WORKS, MAY, FAILS = "likely works", "may work", "likely fails"
RATING = {WORKS: 2, MAY: 1, FAILS: 0}
WORKS_AT, MAY_AT = 0.5, 0.3          # laya-vision's score for an attack, P(hit) -> words
# A block's score is P(blocked), on another scale (2026-09-29, 131k decisions vs Ken/Ryu/Honda): a block scored 0.1-0.2
# actually blocked 40% of the time and 0.2+ 59-65%, while an attack scored 0.5+ actually hit 25-44%. With one threshold
# a block was never rated above "likely fails" (independent review, docs/reviews/2026-09-29_dr_fable.md).
BLOCK_WORKS_AT, BLOCK_MAY_AT = 0.2, 0.1
BLOCK_MOVES = ("block_high", "block_low")
# The lookup table's scale (sf2.data.value_oracle, docs/prereg_2x2.md): a move's expected net (hp dealt - taken until
# the next decision), rated RELATIVE TO WALKING IN (forward's value in the same table row): at or below it "likely
# fails", above it "may work", NET_WORKS_MARGIN or more above it "likely works". Text laya's rule walks in when nothing
# rates above "likely fails", which then means exactly "nothing beats walking in". (An absolute scale, >= 3.0 / >= 1.8,
# matched to runs/all8's word mix, made text laya walk in on 67 of 97 decisions vs Ryu in the 2026-09-30 smoke although
# forward was never the table's best there.) Blocks use the same scale (a net is a net).
NET_WORKS_MARGIN = 3.0
SCALES = ("p_hit", "net")

INSTRUCTIONS = "Which move do I do now? Follow the advice when it fits this moment."


def rating(score: float, move: Optional[str] = None, scale: str = "p_hit", walk: float = 0.0) -> str:
    """A move's score in words. ``scale`` "p_hit": laya-vision's score, a block (``move``) on its own scale; without
    ``move`` the attack scale (what scripts/build_advice_data.py built text laya's training data with). "net": the
    lookup table's expected net in hp, every move (blocks too) on one scale, relative to ``walk`` (forward's value in
    the same situation)."""
    if scale not in SCALES:
        raise ValueError("unknown rating scale %r (one of %s)" % (scale, ", ".join(SCALES)))
    if scale == "net":
        return WORKS if score >= walk + NET_WORKS_MARGIN else MAY if score > walk else FAILS
    else:
        works, may = (BLOCK_WORKS_AT, BLOCK_MAY_AT) if move in BLOCK_MOVES else (WORKS_AT, MAY_AT)
    return WORKS if score >= works else MAY if score >= may else FAILS


def opp_doing(entry: Dict) -> str:
    """What the opponent is doing, from a game-log action entry (sf2.game_log.action_entry)."""
    if entry.get("opp_air"):
        return "jumping"
    return {"crouch": "crouching", "attack": "attacking", "special": "attacking", "hit_stun": "stunned"}.get(
        entry.get("opp_state"), "standing")


def situation_text(rng: str, doing: str, my_bar: str, opp_bar: str, fireball: bool = False) -> str:
    if rng not in RANGES or doing not in OPP_STATES or my_bar not in BARS or opp_bar not in BARS:
        raise ValueError("bad situation %r %r %r %r" % (rng, doing, my_bar, opp_bar))
    base = "He is %s and %s. My bar is %s, his bar is %s." % (RANGE_WORDS[rng], doing, my_bar, opp_bar)
    return (base + " " + FIREBALL_CLAUSE) if fireball else base


def option_text(move: str, rated: Optional[str]) -> str:
    return "walk in" if move == FORWARD else rated


def advice_text(lessons: Sequence[str]) -> str:
    return ("Advice: " + "; ".join(lessons) + ".") if lessons else "Advice: none."


def question(options: Dict[str, str]) -> Dict:
    """``options``: move -> laya-vision's rating words (FORWARD's is ignored)."""
    return {"type": "choice", "instructions": INSTRUCTIONS,
            "criteria": {m: option_text(m, r) for m, r in options.items()}}


def prompt(situation: str, lessons: Sequence[str]) -> str:
    return situation + " " + advice_text(lessons)


# ---- reading a lesson's text (the label must follow what the TEXT says, since that is all text laya reads) ----

NEG = re.compile(r"\b(avoid|never|stop|drop|skip|don't|do not|less|fewer|no|quit|punish(es|ed)?)\b", re.I)
HARD = re.compile(r"\b(always|only)\b", re.I)
HE = r"(he|ryu|ken|blanka|guile|chun-?li|honda|e\.? honda|zangief|dhalsim)"      # the opponent, by pronoun or name
WHEN = {"jumping": re.compile(r"\bwhen %s (jumps|is jumping|is in the air)\b|\bjumping\b" % HE, re.I),
        "crouching": re.compile(r"\bwhen %s (crouches|is crouching|ducks)\b" % HE, re.I),
        "attacking": re.compile(r"\bwhen %s (attacks|is attacking|presses)\b" % HE, re.I),
        "standing": re.compile(r"\bwhen %s (stands|is standing)\b" % HE, re.I),
        "stunned": re.compile(r"\bwhen %s is (stunned|in hit_stun|reeling)\b|\bin hit_stun\b" % HE, re.I)}
WHERE = {"close": re.compile(r"\b(close|next to)\b", re.I),
         "mid": re.compile(r"\b(mid|middle)\b", re.I),
         "far": re.compile(r"\b(far|long range)\b", re.I)}
FIRE = re.compile(r"\bfireballs?\b", re.I)      # G4: the lesson conditions on an incoming fireball (identical token)
HE_DOES = re.compile(r"\b(he|him|ryu|ken|blanka|guile|chunli|chun-li|honda|zangief|dhalsim)\s+(is\s+)?\w+", re.I)
# The throw alias (owner 2026-10-02): a lesson may say the generic word "throw"/"grab" rather than the menu's
# canonical throw-move name (throw_F+hp). When no concrete move is named, a generic throw word resolves to the
# character's throw move (throw_F+hp preferred) so the lesson is FOLLOWABLE, not inert. Canonical names stay the menu's.
THROW_WORD = re.compile(r"\b(throw|throws|throwing|grab|grabs|grabbing|toss|tosses)\b", re.I)


def throw_move(moves: Sequence[str]) -> Optional[str]:
    """The concrete throw move a generic "throw"/"grab" aliases to: throw_F+hp if offered, else the first throw_*
    move, else a literal "throw" if the menu still carries one; None when this menu has no throw."""
    if "throw_F+hp" in moves:
        return "throw_F+hp"
    throws = sorted(m for m in moves if m.startswith("throw_"))
    if throws:
        return throws[0]
    return "throw" if "throw" in moves else None


@dataclass(frozen=True)
class Lesson:
    text: str
    move: Optional[str]          # None: no move to do (e.g. an opponent habit)
    polarity: str                # "soft" | "hard" | "neg" | "none"
    where: Optional[str]         # a range, or None: anywhere
    when: Optional[str]          # an opponent state, or None: whatever he does
    fireball: bool = False       # G4: the lesson holds only while a fireball is coming (default: regardless)

    def applies(self, rng: str, doing: str, fireball: bool = False) -> bool:
        return ((self.where is None or self.where == rng) and (self.when is None or self.when == doing)
                and (not self.fireball or fireball))


def parse(text: str, moves: Sequence[str]) -> Lesson:
    """A lesson as its words say it. Qwen writes "<what to do>: <why>": the move and its polarity are read from the
    part before the first colon (the why may say "he does not punish it"), the conditions from the whole lesson
    ("use c.mk at mid: it lands when he stands"). The move is the character's move named there (longest name
    first, so "c.mk" is not read as "mk"; "throw" only when no other move is named, as in "throw c.mk"; what HE does,
    "he jumps", is not her move); none named -> no move."""
    what = text.split(":", 1)[0]
    cleaned = HE_DOES.sub(" ", what)        # drop "he jumps" / "him up" so his action is not read as my move
    named = [m for m in sorted(moves, key=len, reverse=True)
             if re.search(r"(?<![\w.])%s(ing|s)?(?![\w])" % re.escape(m), cleaned)]
    if len(named) > 1 and "throw" in named:
        named.remove("throw")
    move = named[0] if named else None
    if move is None and THROW_WORD.search(cleaned):   # throw alias: a bare "throw"/"grab" -> the concrete throw move
        move = throw_move(moves)
    if move is None:
        pol = "none"
    elif NEG.search(what):
        pol = "neg"
    elif HARD.search(what):
        pol = "hard"
    else:
        pol = "soft"
    where = [r for r, pat in WHERE.items() if pat.search(text)]      # conditions: anywhere in the lesson
    when = [s for s, pat in WHEN.items() if pat.search(text)]
    if len(where) > 1 or len(when) > 1:
        raise ValueError("lesson has conflicting conditions: %r" % text)
    return Lesson(text, move, pol, where[0] if where else None, when[0] if when else None, bool(FIRE.search(text)))


def read(text: str, moves: Sequence[str]) -> Lesson:
    """``parse`` for play time: a lesson it cannot read (e.g. two ranges in one line) counts as naming no move, so
    System 1 and System 2's checks skip it instead of crashing the game."""
    try:
        return parse(text, moves)
    except ValueError:
        return Lesson(text, None, "none", None, None)


def answers(rng: str, doing: str, options: Dict[str, str], lessons: Sequence[Lesson]) -> Tuple[List[str], str]:
    """The correct move(s) and which rule decided. ``options``: move -> rating words (FORWARD always offered)."""
    if FORWARD not in options:
        raise ValueError("the shortlist must offer %r" % FORWARD)
    live = [les for les in lessons if les.move in options and les.applies(rng, doing)]
    out = {les.move for les in live if les.polarity == "neg"}
    hard = sorted({les.move for les in live if les.polarity == "hard"} - out)
    if hard:
        return hard, "hard"
    soft = sorted({les.move for les in live if les.polarity == "soft" and options[les.move] != FAILS} - out)
    if soft:
        return soft, "soft"
    moves = [m for m in options if m != FORWARD and m not in out]
    best = max((RATING[options[m]] for m in moves), default=0)
    if best > 0:
        return sorted(m for m in moves if RATING[options[m]] == best), "vision"
    if FORWARD not in out:
        return [FORWARD], "walk"
    return sorted(moves) or [FORWARD], "nothing_left"


# ---- G1: the two-stage menu (sf2.system1.action_menu). No table, no ratings: round 1 picks a CATEGORY, round 2 the
# MOVE inside it. Text laya follows an applying advice line; nothing applies -> the hardcoded default (block). The
# situation prunes the stance, so round 2 only ever offers the moves that stance can do. ----

CAT_INSTRUCTIONS = "Which kind of move do I do now? Follow the advice when it fits this moment."
STANCES = ("standing", "close", "crouch", "air")
GROUNDED_STANCES = ("standing", "close", "crouch")
# how a stance picks the move variant: standing -> s.* , grounded+close -> cl.* , crouch -> c.* , air -> j./jf.*
STANCE_PREFIXES = {"standing": ("s.",), "close": ("cl.",), "crouch": ("c.",), "air": ("j.", "jf.")}
NORMAL_PREFIXES = ("jf.", "cl.", "j.", "s.", "c.")      # longest/ambiguous first so "cl."/"jf." win over "c."/"j."


def stance_of(posture: str, rng: str) -> str:
    """The stance that prunes the menu, from my posture (stand / crouch / air) and the range (grounded+far -> standing,
    grounded+close -> close)."""
    if posture == "air":
        return "air"
    if posture == "crouch":
        return "crouch"
    if posture == "stand":
        return "close" if rng == "close" else "standing"
    raise ValueError("unknown posture %r" % posture)


def _prefix(move: str) -> Optional[str]:
    for p in NORMAL_PREFIXES:
        if move.startswith(p):
            return p
    return None


def moves_in_stance(category: str, stance: str, categories: Optional[Dict[str, Sequence[str]]] = None) -> List[str]:
    """``category``'s move names that this stance can actually do: a prefixed normal/combo by its prefix, a throw only
    up close, and the other grounded moves (movement, block, special) only while grounded. ``categories`` is the
    character's category->moves map (default: the module's CATEGORIES, i.e. Chun-Li), so one rule serves all 8."""
    categories = categories if categories is not None else CATEGORIES
    if stance not in STANCES:
        raise ValueError("unknown stance %r" % stance)
    out = []
    for m in categories[category]:
        p = _prefix(m)
        if p is not None:
            if p in STANCE_PREFIXES[stance]:
                out.append(m)
        elif category == "throw":
            if stance == "close":
                out.append(m)
        elif stance in GROUNDED_STANCES:
            out.append(m)
    return out


def available_moves(stance: str, categories: Optional[Dict[str, Sequence[str]]] = None) -> set:
    categories = categories if categories is not None else CATEGORIES
    return {m for cat in CATEGORY_ORDER for m in moves_in_stance(cat, stance, categories)}


# The menu action_menu.CATEGORIES hardcodes for Chun-Li, derived here for ANY character whose RAM-free move menu
# sf2.moves_free knows (chunli / ryu / ken). This is the ONE reusable source of the per-character category map:
# scripts/build_advice_data.char_categories mirrors the same mapping over all 8 characters (it cannot import from a
# play-path module via the hard gate), and char_categories("chunli") is byte-identical to action_menu.CATEGORIES, so
# Chun-Li's labels and menu are unchanged.
KIND_TO_CAT = {"movement": "move", "block": "block", "throw": "throw", "special": "special", "combo": "combo"}


def char_categories(char: str) -> Dict[str, List[str]]:
    """``char``'s category -> move-names map (the 7 categories, in CATEGORY_ORDER), from its RAM-free move menu
    (sf2.moves_free.menu, which raises ValueError for a character it has no menu for). Normals split into punch/kick
    by their button; everything else by kind. char_categories("chunli") equals action_menu.CATEGORIES exactly."""
    cats: Dict[str, List[str]] = {c: [] for c in CATEGORY_ORDER}
    for m in _menu(char):
        if m.kind == "normal":
            cat = "punch" if m.name.rsplit(".", 1)[-1].endswith("p") else "kick"
        else:
            cat = KIND_TO_CAT[m.kind]
        cats[cat].append(m.name)
    return cats


def char_menu_moves(char: str) -> List[str]:
    """``char``'s flat two-stage move vocabulary (CATEGORY_ORDER), the names advice lines are parsed against."""
    cats = char_categories(char)
    return [m for c in CATEGORY_ORDER for m in cats[c]]


def category_question() -> Dict:
    """Round 1: pick one of the 7 categories (unrated; the words + advice decide)."""
    return {"type": "choice", "instructions": CAT_INSTRUCTIONS, "criteria": {c: c for c in CATEGORY_ORDER}}


def move_question(moves: Sequence[str]) -> Dict:
    """Round 2: pick one move from the chosen category, pruned to the stance (empty -> the default move)."""
    moves = list(moves) or [DEFAULT_MOVE]
    return {"type": "choice", "instructions": INSTRUCTIONS, "criteria": {m: m for m in moves}}


def chosen_moves(rng: str, doing: str, stance: str, lessons: Sequence[Lesson],
                 fireball: bool = False, categories: Optional[Dict[str, Sequence[str]]] = None) -> Tuple[List[str],
                                                                                                         str]:
    """The move(s) the advice picks from the stance-pruned (unrated) menu, and which rule decided: an applying hard
    lesson, else an applying soft lesson, else the hardcoded default (block). A negative lesson rules its move out.
    With no ratings a move is chosen only when a lesson names it; otherwise it is the default. ``categories`` is the
    character's menu (default: Chun-Li's CATEGORIES)."""
    categories = categories if categories is not None else CATEGORIES
    avail = available_moves(stance, categories)
    live = [les for les in lessons if les.move in avail and les.applies(rng, doing, fireball)]
    out = {les.move for les in live if les.polarity == "neg"}
    hard = sorted({les.move for les in live if les.polarity == "hard"} - out)
    if hard:
        return hard, "hard"
    soft = sorted({les.move for les in live if les.polarity == "soft"} - out)
    if soft:
        return soft, "soft"
    return [DEFAULT_MOVE], "default"


def two_stage(rng: str, doing: str, stance: str, lessons: Sequence[Lesson], fireball: bool = False,
              categories: Optional[Dict[str, Sequence[str]]] = None) -> Tuple[List[str], List[str], str]:
    """Labels for the two-stage menu: (round-1 category answers, round-2 move answers, the rule that decided). Round 2
    is scoped to one category, so when the picks span categories the caller asks round 2 per category. ``categories``
    is the character's menu (default: Chun-Li's CATEGORIES); the category of each picked move is read from it, so a
    character whose moves action_menu.category_of does not know (e.g. Ryu's hadoken) still resolves."""
    categories = categories if categories is not None else CATEGORIES
    moves, rule = chosen_moves(rng, doing, stance, lessons, fireball, categories)
    cat_of = {m: c for c, names in categories.items() for m in names}
    cats = sorted({cat_of.get(m) or category_of(m) for m in moves})
    return cats, moves, rule
