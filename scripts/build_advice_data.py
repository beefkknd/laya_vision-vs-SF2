"""ONE shared two-stage training dataset for text laya (sf2.system1.advice + sf2.system1.action_menu), covering ALL 8
characters (ryu, honda, blanka, guile, ken, chunli, zangief, dhalsim). It feeds BOTH fine-tunes:

    round "cat"   which of the 7 categories (move/punch/kick/block/throw/special/combo) — options are the bare names,
                  the SAME 7 for every character (round 1 is character-agnostic);
    round "move"  which move inside the chosen category — options are that category's moves PRUNED to the stance for
                  THIS character (advice.moves_in_stance over the character's own menu), plus block_high always offered.

The label follows the advice (advice.two_stage): an applying hard line, else an applying soft line, else the DEFAULT
(block / block_high). A line applies only when its conditions hold (range / what he does / a fireball) AND its move is
one this stance can do. The lever this dataset turns is CONDITION-OFF: a positive line whose condition does NOT hold
teaches the default (block), never the advised category — the Honda-smoke misfire (docs/plan_two_finetune_textlaya.md).

Per-character menus: the shared structural moves (movement, 30 normals, 2 blocks) are character-agnostic and taken
from the shared sf2.moves_free builders / action_menu; the throws, specials and combos are each character's own —
from sf2.moves_free.menu for ryu/ken/chunli, and from sf2.data.actions_free.SPECIALS (the all-8 action vocabulary) for
the other five. char_categories("chunli") is byte-identical to action_menu.CATEGORIES (the old default), so Chun-Li's
labels are unchanged.

    .venv/bin/python scripts/build_advice_data.py                 # -> test_data/advice_v4/{train,val,test}.jsonl
    .venv/bin/python scripts/build_advice_data.py --per-char 2000

Held-out polarity wordings (TEST_WORDS) appear ONLY in the test split, so test measures following the MEANING of a
phrasing never trained on. Log: logs/advice/build_v4.log.
"""
import argparse
import collections
import json
import os
import random
from typing import Dict, List, Sequence, Tuple

import _path  # noqa: F401
from sf2.data.actions_free import MOVEMENT as ACT_MOVEMENT
from sf2.data.actions_free import NORMALS as ACT_NORMALS
from sf2.data.actions_free import SPECIALS as ACT_SPECIALS
from sf2.moves_free import blocks as _free_blocks
from sf2.moves_free import menu as _free_menu
from sf2.moves_free import movement as _free_movement
from sf2.moves_free import normals as _free_normals
from sf2.moves_free import throws as _free_throws
from sf2.system1.action_menu import CATEGORIES, CATEGORY_ORDER, DEFAULT_MOVE
from sf2.system1.advice import (available_moves, category_question, move_question, moves_in_stance, parse, prompt,
                                 situation_text, stance_of, two_stage)
from sf2.vocab import BARS, FIGHTERS, OPP_STATES, RANGES

OUT = "test_data/advice_v4"
POSTURES = ("stand", "crouch", "air")
# the five OFFENSE categories (move + block are defensive); the generator balances condition-ON offense rules across
# these so the category model learns to FOLLOW the offense category when an offense rule applies, instead of defaulting
# to block. Each offense category's (posture, forced-range) choices are the stances in which that category has moves:
# a throw needs the close stance (stand + close range); specials are grounded; combos live in crouch/air.
OFFENSE_CATS = ("punch", "kick", "special", "throw", "combo")
CAT_CHOICES = {"punch": (("stand", None), ("crouch", None), ("air", None)),
               "kick": (("stand", None), ("crouch", None), ("air", None)),
               "special": (("stand", None), ("crouch", None)),
               "throw": (("stand", "close"),),
               "combo": (("crouch", None), ("air", None))}
# moves_free carries the full prefixed menu (incl. specials + combos) for these; the other five it does not.
FREE_MENU_CHARS = ("ryu", "ken", "chunli")
# the _BASE normal-ish keys in actions_free.SPECIALS[char] that are NOT specials (crouch normals / sweep / throw)
ACT_NON_SPECIAL = {"c.lk", "c.mk", "sweep", "c.hp", "throw"} | set(ACT_MOVEMENT) | set(ACT_NORMALS)
KIND_TO_CAT = {"movement": "move", "block": "block", "throw": "throw", "special": "special", "combo": "combo"}


# ------------------------------------------------------------------------------------- per-character action menu
def _menu_pairs(char: str) -> List[Tuple[str, str]]:
    """(move name, kind) for every move this character has. ryu/ken/chunli come straight from moves_free; the other
    five are the shared structural moves + throw_F+hp + their own specials (named as the all-8 action vocabulary
    sf2.data.actions_free keeps them)."""
    if char in FREE_MENU_CHARS:
        return [(m.name, m.kind) for m in _free_menu(char)]
    base = _free_movement() + _free_normals() + _free_blocks() + _free_throws(["hp"])
    pairs = [(m.name, m.kind) for m in base]
    pairs += [(name, "special") for name in ACT_SPECIALS[char] if name not in ACT_NON_SPECIAL]
    return pairs


def char_categories(char: str) -> Dict[str, List[str]]:
    """This character's category -> move-names map (the 7 categories, in CATEGORY_ORDER). Normals split into
    punch/kick by their button; everything else by kind. For "chunli" this equals action_menu.CATEGORIES exactly."""
    cats: Dict[str, List[str]] = {c: [] for c in CATEGORY_ORDER}
    for name, kind in _menu_pairs(char):
        if kind == "normal":
            cat = "punch" if name.rsplit(".", 1)[-1].endswith("p") else "kick"
        else:
            cat = KIND_TO_CAT[kind]
        cats[cat].append(name)
    return cats


def char_moves(cats: Dict[str, List[str]]) -> List[str]:
    return [m for c in CATEGORY_ORDER for m in cats[c]]


_ALL_CAT_OF: Dict[str, str] = {m: c for char in FIGHTERS for c, names in char_categories(char).items() for m in names}


def category_of(move: str) -> str:
    """The category of a move across ALL 8 characters' menus (action_menu.category_of only knows Chun-Li's)."""
    if move not in _ALL_CAT_OF:
        raise ValueError("move %r is in no category" % move)
    return _ALL_CAT_OF[move]


# ------------------------------------------------------------------------------------- wordings
WHERE_WORDS = {"close": ["up close", "at close range", "next to him"],
               "mid": ["at mid", "at mid range", "in the middle"], "far": ["far away", "at far", "at long range"]}
WHEN_WORDS = {"jumping": "when he jumps", "crouching": "when he crouches", "attacking": "when he attacks",
              "standing": "when he stands", "stunned": "when he is stunned"}
FIRE_WORDS = ["when a fireball comes", "when a fireball is coming", "against a fireball", "when a fireball approaches"]
TRAIN_WORDS = {
    "soft": ["use more {m}", "prefer {m}", "go for {m}", "use {m}", "hit him with {m}"],
    "hard": ["always use {m}", "only use {m}"],
    "neg": ["avoid {m}", "never use {m}", "stop using {m}", "use less {m}", "don't use {m}", "no {m}"],
}
TEST_WORDS = {           # never in training: does it generalise the polarity of a new phrasing?
    "soft": ["lean on {m}", "try {m} more often"],
    "hard": ["always go for {m}"],
    "neg": ["fewer {m}", "quit using {m}", "do not use {m}"],
}
REASONS = {"soft": ["it lands", "he does not punish it", "it works"], "hard": ["it is the only thing that works"],
           "neg": ["it whiffs", "he punishes it", "it misses"]}
HABITS = ["he jumps a lot {w}, be ready", "he crouches a lot {w}", "he attacks a lot {w}"]
# generic throw words a lesson may use instead of the canonical name throw_F+hp (the throw-alias, advice.throw_move)
GENERIC_THROW = ["throw", "throw him", "grab", "grab him", "throw him down"]

# REBALANCED for advice_v4. v3 was ~66% "default -> block", which bled block into condition-ON offense cases (the
# category model then defaulted to block even when a soft offense rule applied). Here the default (block) share is cut
# to ~40%: still heavy enough to keep condition_off strong (the spam fix), but no longer dominating. The freed weight
# goes to condition-ON offense rules (hard/soft), which now target the 5 offense categories balanced (see _pick_offense).
CASES = {"hard": 0.14, "soft": 0.40, "fireball": 0.10, "plain": 0.03, "neg": 0.06, "default": 0.05, "condition_off": 0.22}
EXPECT = {"hard": {"hard"}, "soft": {"soft"}, "neg": {"default"}, "condition_off": {"default"}, "plain": {"default"},
          "default": {"default"}, "fireball": {"hard", "soft", "default"}}


def HAS_GENERIC_THROW(line: str) -> bool:
    """A lesson line that leans on the generic word "throw"/"grab" rather than the canonical throw_F+hp name."""
    low = line.lower()
    return ("throw_f+" not in low) and any(w in low for w in ("throw", "grab", "toss"))


def rand_sit(rng: random.Random) -> Tuple[str, str, str, str]:
    return rng.choice(RANGES), rng.choice(OPP_STATES), rng.choice(BARS), rng.choice(BARS)


def _phrase(rng: random.Random, move: str, generic: bool) -> str:
    """How a move is named in a line: its canonical name, or (for throw_F+hp, when ``generic``) a bare throw word."""
    if generic and move == "throw_F+hp":
        return rng.choice(GENERIC_THROW)
    return move


def line(rng: random.Random, move: str, pol: str, words: Dict, where: str = None, when: str = None,
         fire: bool = False, generic: bool = False) -> str:
    """One advice line: the polarity phrasing of ``move`` plus any conditions, messily ordered, sometimes a reason."""
    text = rng.choice(words[pol]).format(m=_phrase(rng, move, generic))
    conds = []
    if when:
        conds.append(WHEN_WORDS[when])
    if where:
        conds.append(rng.choice(WHERE_WORDS[where]))
    if fire:
        conds.append(rng.choice(FIRE_WORDS))
    rng.shuffle(conds)
    lead = next((c for c in conds if c.startswith("when")), None) if rng.random() < 0.5 else None
    if lead:
        rest = [c for c in conds if c is not lead]
        text = lead + ", " + text + ((" " + " ".join(rest)) if rest else "")
    elif conds:
        text = text + " " + " ".join(conds)
    if rng.random() < 0.4:
        text += ": " + rng.choice(REASONS[pol])
    return text


def distractors(rng: random.Random, moves: Sequence[str], target: str) -> List[str]:
    """0-2 lines that never change the answer: opponent habits, and negatives on OTHER moves (with no ratings a move
    is only ever chosen by a positive line naming it, and the default is never gated by a negative)."""
    out = []
    for _ in range(rng.choice([0, 0, 1, 1, 2])):
        if rng.random() < 0.5:
            out.append(rng.choice(HABITS).format(w=rng.choice(sum(WHERE_WORDS.values(), []))))
        else:
            other = rng.choice([m for m in moves if m != target])
            out.append(rng.choice(TRAIN_WORDS["neg"]).format(m=other))
    return out


def hold(rng: random.Random, p: float) -> Tuple[Dict, str]:
    return (TEST_WORDS, "held_out") if rng.random() < p else (TRAIN_WORDS, "trained")


def _cat_and_target(rng: random.Random, cats: Dict[str, List[str]], stance: str) -> Tuple[str, str, bool]:
    """A category available in this stance and one of its moves (uniform over categories, to balance the matrix).
    Sometimes the throw category with the canonical throw_F+hp phrased generically (exercises the throw alias)."""
    avail_cats = [c for c in CATEGORY_ORDER if moves_in_stance(c, stance, cats)]
    if "throw" in avail_cats and "throw_F+hp" in cats["throw"] and rng.random() < 0.18:
        return "throw", "throw_F+hp", True
    cat = rng.choice(avail_cats)
    return cat, rng.choice(moves_in_stance(cat, stance, cats)), False


def _pick_offense(rng: random.Random, cats: Dict[str, List[str]], rng_: str) -> Tuple[str, str, str, str, bool]:
    """Balanced over the 5 OFFENSE categories: pick a category uniformly, then a posture (and, for a throw, the close
    range) that makes it available for THIS character. Returns (category, posture, forced-range or rng_, target move,
    generic-throw flag). Falls back to a punch/kick (always available) if a character lacks the drawn category."""
    for _ in range(24):
        # categories are drawn uniformly; a draw the character/stance cannot realise simply re-rolls (so non-combo
        # characters never produce a combo). combo therefore lands lower than the others -- it exists on only 3 of the
        # 8 fighters -- and is NOT oversampled on purpose: oversampling it concentrates ryu/ken's single air combo and
        # lets the metadata shortcut read the move from stance=air (the close/air block dilution below is the pair fix).
        cat = rng.choice(OFFENSE_CATS)
        posture, forced = rng.choice(CAT_CHOICES[cat])
        use_rng = forced if forced is not None else rng_
        stance = stance_of(posture, use_rng)
        pool = moves_in_stance(cat, stance, cats)
        if pool:
            if cat == "throw" and "throw_F+hp" in pool and rng.random() < 0.5:
                return cat, posture, use_rng, "throw_F+hp", rng.random() < 0.6
            return cat, posture, use_rng, rng.choice(pool), False
    cat = "punch" if rng.random() < 0.5 else "kick"
    posture = rng.choice(("stand", "crouch", "air"))
    pool = moves_in_stance(cat, stance_of(posture, rng_), cats)
    return cat, posture, rng_, rng.choice(pool), False


def make(rng: random.Random, char: str, cats: Dict[str, List[str]], moves: List[str], case: str, hold_p: float) -> Dict:
    """One decision for ``char`` realised as the intended ``case``: situation, stance, fireball flag, advice lines.
    For condition-ON offense cases (hard/soft/fireball) the target is a BALANCED offense category 85% of the time,
    with the remaining 15% falling back to any category so move/block stay reachable and legitimate block rules remain.
    condition_off always names an offense move (offense phrasing, condition off -> still block: the exact spam fix)."""
    rng_, doing, my_bar, opp_bar = rand_sit(rng)
    offense_case = case in ("hard", "soft", "fireball", "condition_off")
    use_offense = offense_case and (case == "condition_off" or rng.random() < 0.85)
    off_target = off_generic = None
    if use_offense:
        _ocat, posture, rng_, off_target, off_generic = _pick_offense(rng, cats, rng_)
        stance = stance_of(posture, rng_)
    elif case in ("plain", "neg", "default"):
        # a throw lives ONLY in the close stance and a combo ONLY in crouch/air, so those offense moves would be
        # readable from the stance alone. Route the block-labelled cases to the SAME leaky stances, so block_high
        # (the default/global answer) stays the plurality there and the metadata shortcut cannot read the move.
        roll = rng.random()
        if roll < 0.45:
            rng_, stance = "close", "close"
        elif roll < 0.75:
            stance = "air"
        elif roll < 0.90:
            stance = "crouch"
        else:
            stance = stance_of("stand", rng_)
    else:
        stance = stance_of(rng.choice(POSTURES), rng_)
    avail = sorted(available_moves(stance, cats))
    if not avail:                                            # degenerate stance (no move) -> retriable
        return make(rng, char, cats, moves, case, hold_p)
    fire_sit = False
    lessons: List[str] = []
    tags: List[str] = []

    def words_of():
        w, tag = hold(rng, hold_p)
        tags.append(tag)
        return w

    def cat_and_target():                                    # the balanced offense target when drawn, else any category
        if use_offense:
            return None, off_target, off_generic
        return _cat_and_target(rng, cats, stance)

    if case == "plain":
        lessons = distractors(rng, moves, DEFAULT_MOVE) or [rng.choice(HABITS).format(w=rng.choice(WHERE_WORDS[rng_]))]
        fire_sit = rng.random() < 0.15
    elif case in ("hard", "soft"):
        _cat, target, generic = cat_and_target()
        where = rng_ if rng.random() < 0.5 else None
        when = doing if rng.random() < 0.5 else None
        lessons = [line(rng, target, case, words_of(), where=where, when=when, generic=generic)]
        lessons += distractors(rng, moves, target)
        fire_sit = rng.random() < 0.15
    elif case == "neg":                                      # a soft line names X, a negative line rules X out -> default
        target = rng.choice(avail)
        lessons = [line(rng, target, "soft", words_of()), line(rng, target, "neg", words_of())]
        lessons += distractors(rng, moves, target)
        fire_sit = rng.random() < 0.15
    elif case == "fireball":                                 # a line conditioned on a fireball; present -> follow, else default
        _cat, target, generic = cat_and_target()
        pol = rng.choice(["soft", "hard"])
        fire_sit = rng.random() < 0.5
        where = rng_ if rng.random() < 0.4 else None
        lessons = [line(rng, target, pol, words_of(), where=where, fire=True, generic=generic)]
        lessons += distractors(rng, moves, target)
    elif case == "default":                                  # advice names a move this stance CANNOT do -> default
        off = [m for m in moves if m not in avail]
        target = rng.choice(off) if off else rng.choice(avail)
        pol = rng.choice(["soft", "hard"])
        lessons = [line(rng, target, pol, words_of(), where=rng_ if rng.random() < 0.5 else None)]
        lessons += distractors(rng, moves, target)
        fire_sit = rng.random() < 0.15
    else:                                                    # condition_off: a positive line whose range/state does not hold
        _cat, target, generic = cat_and_target()
        pol = rng.choice(["soft", "hard"])
        where = rng.choice([x for x in RANGES if x != rng_]) if rng.random() < 0.5 else None
        when = rng.choice([x for x in OPP_STATES if x != doing]) if not where else None
        lessons = [line(rng, target, pol, words_of(), where=where, when=when, generic=generic)]
        lessons += distractors(rng, moves, target)
        fire_sit = rng.random() < 0.15

    rng.shuffle(lessons)
    parsed = [parse(t, moves) for t in lessons]
    cats_ans, moves_ans, rule = two_stage(rng_, doing, stance, parsed, fire_sit, cats)
    return {"me": char, "rng": rng_, "doing": doing, "my_bar": my_bar, "opp_bar": opp_bar, "stance": stance,
            "fire": fire_sit, "lessons": lessons, "cats": cats_ans, "moves": moves_ans, "rule": rule, "case": case,
            "words": "held_out" if "held_out" in tags else "trained"}


def rows_of(d: Dict, cats: Dict[str, List[str]]) -> List[Dict]:
    """The round-1 and round-2 rows of one decision (same prompt, different question). block_high is always among
    the round-2 options (the safe move is always available), and "block" is always a round-1 option."""
    sit = situation_text(d["rng"], d["doing"], d["my_bar"], d["opp_bar"], fireball=d["fire"])
    text = prompt(sit, d["lessons"])
    inv = {m: c for c, names in cats.items() for m in names}
    chosen_cat = inv[d["moves"][0]] if d["rule"] != "default" else "block"
    move_answers = [m for m in d["moves"] if inv.get(m) == chosen_cat]
    opts = moves_in_stance(chosen_cat, d["stance"], cats)
    if DEFAULT_MOVE not in opts:                             # block_high always offered in round 2
        opts = opts + [DEFAULT_MOVE]
    base = {"me": d["me"], "text": text, "lessons": d["lessons"], "rule": d["rule"], "case": d["case"],
            "stance": d["stance"], "rng": d["rng"], "doing": d["doing"], "my_bar": d["my_bar"],
            "opp_bar": d["opp_bar"], "fireball": d["fire"], "words": d["words"]}
    return [dict(base, round="cat", question=category_question(), answers=d["cats"]),
            dict(base, round="move", question=move_question(opts), answers=move_answers)]


def build(per_char: int, seed: int) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    cats_by_char = {c: char_categories(c) for c in FIGHTERS}
    moves_by_char = {c: char_moves(cats_by_char[c]) for c in FIGHTERS}
    splits: Dict[str, List[Dict]] = collections.defaultdict(list)
    for split, n, hold_p in (("train", per_char, 0.0), ("val", max(8, per_char // 16), 0.0),
                             ("test", max(16, per_char // 8), 0.5)):
        for char in FIGHTERS:
            cats, moves = cats_by_char[char], moves_by_char[char]
            quota = {c: round(n * s) for c, s in CASES.items()}
            while any(v > 0 for v in quota.values()):
                case = rng.choices(list(quota), weights=[max(0, v) for v in quota.values()])[0]
                for _ in range(30):
                    d = make(rng, char, cats, moves, case, hold_p)
                    if d["rule"] in EXPECT[case]:
                        break
                quota[case] -= 1
                splits[split].extend(rows_of(d, cats))
    for split, rows in splits.items():
        rng.shuffle(rows)
        for i, r in enumerate(rows):
            r["id"] = "%s_%06d" % (split, i)
    return splits


# ------------------------------------------------------------------------------------- the shortcut (no-leakage) check
# A metadata-only predictor is FIT on train and scored on test, so it can only win by signal that GENERALISES — never
# by memorising a one-off situation. The features are the structured situation fields that could plausibly leak the
# answer (character, stance, range, his state, fireball); the advice TEXT and the situation sentence are withheld. If
# this beats the majority baseline by more than 0.05, the label can be read off the metadata and the advice is bypassable.
def _meta_key(r: Dict) -> Tuple:
    return (r["me"], r["stance"], r["rng"], r["doing"], bool(r["fireball"]))


def _global_guess(rows: List[Dict]) -> str:
    tally: Dict[str, int] = collections.Counter()
    for r in rows:
        for a in set(r["answers"]):
            tally[a] += 1
    return max(tally, key=tally.get) if tally else DEFAULT_MOVE


def _acc(rows: List[Dict], guess_of) -> float:
    return sum(guess_of(r) in set(r["answers"]) for r in rows) / len(rows) if rows else 0.0


def _round_scores(train: List[Dict], test: List[Dict]) -> Tuple[float, float]:
    """(chance, meta) on the TEST rows: the majority baseline, and a metadata predictor fit on TRAIN (per-key
    majority, falling back to the global majority when a key is unseen or its guess is not offered here)."""
    gbl = _global_guess(train)
    chance = _acc(test, lambda r: gbl)
    key_guess: Dict[Tuple, str] = {}
    groups: Dict[Tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in train:
        for a in set(r["answers"]):
            groups[_meta_key(r)][a] += 1
    for k, tally in groups.items():
        key_guess[k] = max(tally, key=tally.get)

    def guess_of(r: Dict) -> str:
        offered = set(r["question"]["criteria"])
        g = key_guess.get(_meta_key(r), gbl)
        return g if g in offered else (gbl if gbl in offered else next(iter(offered)))

    return chance, _acc(test, guess_of)


def shortcut_scores(splits: Dict[str, List[Dict]]) -> Tuple[float, float, float, float]:
    """(cat_chance, cat_meta, move_chance, move_meta): majority baseline vs the train-fit metadata-only predictor,
    scored on the held-out test split. A clean dataset keeps meta <= chance + 0.05 for BOTH rounds."""
    def rnd(split: str, name: str) -> List[Dict]:
        return [r for r in splits[split] for _ in (0,) if r["round"] == name]
    cat_chance, cat_meta = _round_scores(rnd("train", "cat"), rnd("test", "cat"))
    move_chance, move_meta = _round_scores(rnd("train", "move"), rnd("test", "move"))
    return cat_chance, cat_meta, move_chance, move_meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-char", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    splits = build(args.per_char, args.seed)
    os.makedirs(OUT, exist_ok=True)
    os.makedirs("logs/advice", exist_ok=True)
    allrows = [r for rs in splits.values() for r in rs]
    cat_chance, cat_meta, move_chance, move_meta = shortcut_scores(splits)
    with open("logs/advice/build_v4.log", "w") as log:
        def emit(s: str) -> None:
            print(s)
            log.write(s + "\n")
        for split, rows in splits.items():
            with open(os.path.join(OUT, split + ".jsonl"), "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            by_char = collections.Counter(r["me"] for r in rows)
            emit("%-5s %6d rows  rounds %s" % (split, len(rows), dict(collections.Counter(r["round"] for r in rows))))
            emit("      chars %s" % dict(by_char))
            emit("      cases %s" % dict(collections.Counter(r["case"] for r in rows)))
            emit("      rule  %s  held_out %d" % (dict(collections.Counter(r["rule"] for r in rows)),
                                                  sum(r["words"] == "held_out" for r in rows)))
        block_share = sum(r["rule"] == "default" for r in allrows) / len(allrows)
        cond_off = sum(r["case"] == "condition_off" for r in allrows) / len(allrows)
        emit("TOTAL %6d rows  block-target(rule=default) %.3f  condition_off case %.3f" %
             (len(allrows), block_share, cond_off))
        emit("SHORTCUT cat: chance %.3f meta %.3f (gap %.3f)  move: chance %.3f meta %.3f (gap %.3f)" %
             (cat_chance, cat_meta, cat_meta - cat_chance, move_chance, move_meta, move_meta - move_chance))


if __name__ == "__main__":
    main()
